#include <metal_stdlib>
using namespace metal;

// The Neural Engine's side of a product C[m, n] = A[m, k] · W[k, n], packed and unpacked on the GPU so the CPU
// stays free for the AMX. The Neural Engine computes in fp16 and flushes values under fp16's smallest normal,
// so (scheme from Siliconed):
//   each weight column j is lifted by λ_j = 2^-floor(log2 max_k |W_kj|), its absmax into [1, 2): exact;
//   each activation row i by σ_i = 2^floor(log2(2^15 / (max_k |A_ik| · L))), L = max_j Σ_k |W_kj| λ_j,
//   so every output |Y_ij| <= max|A_i| σ_i L <= 2^15 stays under fp16's 65 504;
//   C_ij = Y_ij / (σ_i λ_j) in fp32.
// The product runs in tiles of at most 4 096 × 4 096 (one Core ML program each, the Neural Engine's fast range);
// tiles along K are summed here in fp32.

// fp32 -> fp16 (check-ane).
kernel void pack_half(device const float *src [[buffer(0)]], device half *dst [[buffer(1)]],
                      constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) dst[i] = half(src[i]);
}

struct AneShape {
    uint k;        // full K of the product
    uint n;        // full N
    uint k0, kPart;  // this tile's K range
    uint n0, nPart;  // this tile's N range
    uint row0;     // first row of A / C the Neural Engine takes
    uint rows;     // its rows
    uint kParts;   // tiles along K (unpack)
};

// Per column of W [k, n]: λ_j and the lifted column's L1 norm. One thread per column (coalesced rows).
kernel void ane_columns(device const float *w [[buffer(0)]], device float *lambda [[buffer(1)]],
                        device float *norm [[buffer(2)]], constant AneShape &s [[buffer(3)]],
                        uint j [[thread_position_in_grid]]) {
    if (j >= s.n) return;
    float peak = 0, sum = 0;
    for (uint r = 0; r < s.k; r++) {
        float v = fabs(w[(ulong)r * s.n + j]);
        peak = max(peak, v);
        sum += v;
    }
    float l = (peak > 0 && isfinite(peak)) ? exp2(-floor(log2(peak))) : 1.0f;
    lambda[j] = l;
    norm[j] = sum * l;
}

// L = max_j norm[j], into state[0]. One threadgroup of 1 024.
kernel void ane_norm(device const float *norm [[buffer(0)]], device float *state [[buffer(1)]],
                     constant AneShape &s [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[32];
    float m = 0;
    for (uint j = tid; j < s.n; j += 1024) m = max(m, norm[j]);
    m = simd_max(m);
    if (tid % 32 == 0) shared[tid / 32] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint i = 0; i < 32; i++) total = max(total, shared[i]);
        state[0] = total;
    }
}

// One weight tile, lifted, into its surface [kPart, nPart] fp16.
kernel void ane_pack_weight(device const float *w [[buffer(0)]], device const float *lambda [[buffer(1)]],
                            device half *tile [[buffer(2)]], constant AneShape &s [[buffer(3)]],
                            uint2 g [[thread_position_in_grid]]) {
    uint jj = g.x, kk = g.y;
    if (jj >= s.nPart || kk >= s.kPart) return;
    uint j = s.n0 + jj;
    tile[(ulong)kk * s.nPart + jj] = half(w[(ulong)(s.k0 + kk) * s.n + j] * lambda[j]);
}

// σ_i of each Neural Engine row. One threadgroup of 256 per row.
kernel void ane_row_scales(device const float *a [[buffer(0)]], device const float *state [[buffer(1)]],
                           device float *sigma [[buffer(2)]], constant AneShape &s [[buffer(3)]],
                           uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[8];
    device const float *row = a + (ulong)(s.row0 + i) * s.k;
    float m = 0;
    for (uint c = tid; c < s.k; c += 256) m = max(m, fabs(row[c]));
    m = simd_max(m);
    if (tid % 32 == 0) shared[tid / 32] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float peak = 0;
        for (uint q = 0; q < 8; q++) peak = max(peak, shared[q]);
        float bound = peak * state[0];
        sigma[i] = (bound > 0 && isfinite(bound)) ? exp2(floor(log2(32768.0f / bound))) : 1.0f;
    }
}

// The Neural Engine's rows of A, scaled, into a tile's input surface [rows, kPart] fp16.
kernel void ane_pack_rows(device const float *a [[buffer(0)]], device const float *sigma [[buffer(1)]],
                          device half *x [[buffer(2)]], constant AneShape &s [[buffer(3)]],
                          uint2 g [[thread_position_in_grid]]) {
    uint kk = g.x, i = g.y;
    if (kk >= s.kPart || i >= s.rows) return;
    x[(ulong)i * s.kPart + kk] = half(a[(ulong)(s.row0 + i) * s.k + s.k0 + kk] * sigma[i]);
}

// C rows from the outputs of the K tiles of one N range (up to 3), summed in fp32, unscaled.
kernel void ane_unpack(device const half *y0 [[buffer(0)]], device const half *y1 [[buffer(1)]],
                       device const half *y2 [[buffer(2)]], device const float *lambda [[buffer(3)]],
                       device const float *sigma [[buffer(4)]], device float *c [[buffer(5)]],
                       constant AneShape &s [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
    uint jj = g.x, i = g.y;
    if (jj >= s.nPart || i >= s.rows) return;
    ulong at = (ulong)i * s.nPart + jj;
    float sum = float(y0[at]);
    if (s.kParts > 1) sum += float(y1[at]);
    if (s.kParts > 2) sum += float(y2[at]);
    uint j = s.n0 + jj;
    c[(ulong)(s.row0 + i) * s.n + j] = (sum / lambda[j]) / sigma[i];
}
