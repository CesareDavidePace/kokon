#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Fused scaled dot-product attention, fp32, head_dim 128, on the layout the products write: q, k, v and out are
// rows of `stride` floats (all heads side by side), head h at columns [128 h, 128 h + 128).
//
// A threadgroup takes 64 query rows of one head (8 SIMD groups of 8 rows) and walks the keys 32 at a time with an
// online softmax (running max and sum per row), so the [queries, keys] score matrix never exists in memory. Each
// key tile, then value tile, is loaded once into threadgroup memory by all 256 threads (coalesced rows) and shared
// by the 8 SIMD groups; products on 8×8 SIMD-group matrices.
//
// causal: the queries are the last `queries` of the keys; query i sees keys j <= keys - queries + i (a text
// segment of Qwen-Image 2.1's block-causal mask). Rows past `queries` are computed but never stored and keys past
// `keys` are masked: the caller guarantees those rows exist (up to 63 past the end) and are finite.
//
// Barriers are threadgroup barriers: with simdgroup barriers alone the lanes' writes to threadgroup memory were
// sometimes not seen by the SIMD-group loads that followed (1 run in 4 wrong, M2 Pro, macOS 27, AttentionTests).

constant uint BQ = 64, BK = 32, D = 128, GROUPS = 8;

struct FlashParams {
    uint queries;
    uint keys;
    uint causal;
    uint stride;
    float scale;     // 1/sqrt(head_dim) * log2(e): scores go straight to exp2
    uint headBase;
};

kernel void flash_attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                            device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
                            constant FlashParams &p [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tile[BK * D];                 // 16 KB: the key tile, then the value tile
    threadgroup float stageAll[GROUPS][8 * BK];     // 8 KB: each SIMD group's scores / probabilities
    threadgroup float diagAll[GROUPS][64];          // 2 KB: each SIMD group's per-row scale, as a diagonal
    threadgroup float *stage = stageAll[sg];
    threadgroup float *diag = diagAll[sg];

    const uint head = p.headBase + tg.y;
    const uint q0 = tg.x * BQ + sg * 8;
    const uint stride = p.stride;
    device const float *kh = k + head * D;
    device const float *vh = v + head * D;

    simdgroup_float8x8 Q[16], O[16];
    for (uint d = 0; d < 16; d++) {
        simdgroup_load(Q[d], q + (ulong)q0 * stride + head * D + 8 * d, stride);
        O[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    diag[lane * 2] = 0.0f;
    diag[lane * 2 + 1] = 0.0f;

    // Each lane owns 8 scores of one row: row lane / 4, columns (lane % 4) * 8 ... + 8.
    const uint row = lane / 4, cbase = (lane % 4) * 8;
    const uint qi = q0 + row;
    const uint offset = p.keys - p.queries;
    const int limit = p.causal ? int(offset + qi) : int(p.keys) - 1;
    float m = -INFINITY, l = 0.0f;

    uint kEnd = p.keys;
    if (p.causal) kEnd = min(p.keys, offset + min(tg.x * BQ + BQ - 1, p.queries - 1) + 1);

    for (uint j0 = 0; j0 < kEnd; j0 += BK) {
        // Key tile: 32 rows × 128, 256 threads × 4 float4.
        for (uint i = tid; i < BK * D / 4; i += 256) {
            uint r = i / (D / 4), c = i % (D / 4);
            ((threadgroup float4 *)tile)[i] = *((device const float4 *)(kh + (ulong)(j0 + r) * stride) + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[4];
        for (uint c = 0; c < 4; c++) S[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0; d < 16; d++) {
            for (uint c = 0; c < 4; c++) {
                simdgroup_float8x8 Kt;
                simdgroup_load(Kt, tile + 8 * c * D + 8 * d, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[c], Q[d], Kt, S[c]);
            }
        }
        for (uint c = 0; c < 4; c++) simdgroup_store(S[c], stage + 8 * c, BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Value tile into the same memory (everyone is done with the keys), softmax meanwhile.
        for (uint i = tid; i < BK * D / 4; i += 256) {
            uint r = i / (D / 4), c = i % (D / 4);
            ((threadgroup float4 *)tile)[i] = *((device const float4 *)(vh + (ulong)(j0 + r) * stride) + c);
        }
        float s[8];
        float blockMax = -INFINITY;
        for (uint t = 0; t < 8; t++) {
            int j = int(j0 + cbase + t);
            float x = stage[row * BK + cbase + t] * p.scale;
            if (j > limit) x = -INFINITY;
            s[t] = x;
            blockMax = max(blockMax, x);
        }
        blockMax = max(blockMax, simd_shuffle_xor(blockMax, 1));
        blockMax = max(blockMax, simd_shuffle_xor(blockMax, 2));
        float mNew = max(m, blockMax);
        float alpha = (mNew == -INFINITY) ? 1.0f : exp2(m - mNew);
        float sum = 0.0f;
        for (uint t = 0; t < 8; t++) {
            float e = (mNew == -INFINITY) ? 0.0f : exp2(s[t] - mNew);
            stage[row * BK + cbase + t] = e;
            sum += e;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * alpha + sum;
        m = mNew;
        if (lane % 4 == 0) diag[row * 8 + row] = alpha;
        const bool rescale = !simd_all(alpha == 1.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 P[4];
        for (uint c = 0; c < 4; c++) simdgroup_load(P[c], stage + 8 * c, BK);
        if (rescale) {
            simdgroup_float8x8 Dg;
            simdgroup_load(Dg, diag, 8);
            for (uint d = 0; d < 16; d++) {
                simdgroup_float8x8 scaled;
                simdgroup_multiply(scaled, Dg, O[d]);
                O[d] = scaled;
            }
        }
        for (uint d = 0; d < 16; d++) {
            for (uint c = 0; c < 4; c++) {
                simdgroup_float8x8 V;
                simdgroup_load(V, tile + 8 * c * D + 8 * d, D);
                simdgroup_multiply_accumulate(O[d], P[c], V, O[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / l per row, then the valid rows to memory.
    if (lane % 4 == 0) diag[row * 8 + row] = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 Dg;
    simdgroup_load(Dg, diag, 8);
    for (uint part = 0; part < 4; part++) {
        for (uint c = 0; c < 4; c++) {
            simdgroup_float8x8 r;
            simdgroup_multiply(r, Dg, O[part * 4 + c]);
            simdgroup_store(r, stage + 8 * c, BK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qi < p.queries) {
            device float *o = out + (ulong)qi * stride + head * D + part * 32 + cbase;
            for (uint t = 0; t < 8; t++) o[t] = stage[row * BK + cbase + t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
