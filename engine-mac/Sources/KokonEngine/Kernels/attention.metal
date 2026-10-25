#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Fused scaled dot-product attention, fp32, head_dim 128, on the layout the products write: q, k, v and out are
// rows of `stride` floats (all heads side by side), head h at columns [128 h, 128 h + 128).
//
// A threadgroup takes 64 query rows of one head (8 SIMD groups of 8 rows) and walks the keys 64 at a time with an
// online softmax (running max and sum per row), so the [queries, keys] score matrix never exists in memory. Each
// key tile, then its value tile, is loaded once into threadgroup memory by all 256 threads (coalesced rows) and
// shared by the 8 SIMD groups; products on 8×8 SIMD-group matrices.
//
// The softmax stays in registers: each lane holds two elements of every 8×8 matrix, row (lane / 16) * 4 +
// (lane / 2) % 4, columns ((lane / 8) % 2) * 4 + (lane % 2) * 2 and the next (`thread_elements`, the layout of
// Apple's GPUs); the four lanes of a row differ in lane bits 0 and 3, so a row's max and sum take two shuffles, and
// O is rescaled in place.
//
// Measured at 1024² (4 096 queries × 4 128 keys, 32 heads, M2 Pro, `probe-attention`, 2026-10-25): 120 ms; the
// first version (32 keys a tile, scores through threadgroup memory, O rescaled by a diagonal product) 145 ms; with
// separate 32-key K and V tiles 135; 16-key tiles or 4 SIMD groups 1.5-2.6× slower, 16 SIMD groups 178, K and V read
// straight from memory by each SIMD group 520.
//
// causal: the queries are the last `queries` of the keys; query i sees keys j <= keys - queries + i (a text
// segment of Qwen-Image 2.1's block-causal mask). Rows past `queries` are computed but never stored and keys past
// `keys` are masked: the caller guarantees those rows exist (up to 63 past the end) and are finite.
//
// Barriers are threadgroup barriers: with simdgroup barriers alone the lanes' writes to threadgroup memory were
// sometimes not seen by the SIMD-group loads that followed (1 run in 4 wrong, M2 Pro, macOS 27, AttentionTests).

constant uint BQ = 64, BK = 64, D = 128;

struct FlashParams {
    uint queries;
    uint keys;
    uint causal;
    uint stride;
    float scale;     // 1/sqrt(head_dim) * log2(e): scores go straight to exp2
    uint headBase;
};

// A lane's own two elements of an 8×8 matrix (`thread_elements` is typed as the whole matrix).
static inline thread float2 &elements(thread simdgroup_float8x8 &m) {
    return *(thread float2 *)&m.thread_elements();
}

kernel void flash_attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                            device const float *v [[buffer(2)]], device float *out [[buffer(3)]],
                            constant FlashParams &p [[buffer(4)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tile[BK * D];   // 32 KB: the key tile, then the value tile

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

    const uint row = (lane / 16) * 4 + (lane / 2) % 4;
    const uint col = ((lane / 8) % 2) * 4 + (lane % 2) * 2;
    const uint qi = q0 + row;
    const uint offset = p.keys - p.queries;
    const int limit = p.causal ? int(offset + qi) : int(p.keys) - 1;
    float m = -INFINITY, l = 0.0f;

    uint kEnd = p.keys;
    if (p.causal) kEnd = min(p.keys, offset + min(tg.x * BQ + BQ - 1, p.queries - 1) + 1);

    for (uint j0 = 0; j0 < kEnd; j0 += BK) {
        // Key tile: 64 rows × 128, 256 threads × 8 float4.
        for (uint i = tid; i < BK * D / 4; i += 256) {
            uint r = i / (D / 4), c = i % (D / 4);
            ((threadgroup float4 *)tile)[i] = *((device const float4 *)(kh + (ulong)(j0 + r) * stride) + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[8];
        for (uint c = 0; c < 8; c++) S[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0; d < 16; d++) {
            for (uint c = 0; c < 8; c++) {
                simdgroup_float8x8 Kt;
                simdgroup_load(Kt, tile + 8 * c * D + 8 * d, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[c], Q[d], Kt, S[c]);
            }
        }

        // Online softmax on the lane's two scores of each of the 8 matrices.
        float2 s[8];
        float blockMax = -INFINITY;
        for (uint c = 0; c < 8; c++) {
            s[c] = elements(S[c]) * p.scale;
            int j = int(j0 + 8 * c + col);
            if (j > limit) s[c].x = -INFINITY;
            if (j + 1 > limit) s[c].y = -INFINITY;
            blockMax = max(blockMax, max(s[c].x, s[c].y));
        }
        blockMax = max(blockMax, simd_shuffle_xor(blockMax, 1));
        blockMax = max(blockMax, simd_shuffle_xor(blockMax, 8));
        float mNew = max(m, blockMax);
        float alpha = (mNew == -INFINITY) ? 1.0f : exp2(m - mNew);
        float sum = 0.0f;
        for (uint c = 0; c < 8; c++) {
            float2 e = (mNew == -INFINITY) ? float2(0.0f) : exp2(s[c] - mNew);
            elements(S[c]) = e;
            sum += e.x + e.y;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        l = l * alpha + sum;
        m = mNew;
        if (!simd_all(alpha == 1.0f)) {
            for (uint d = 0; d < 16; d++) elements(O[d]) *= alpha;
        }

        // Value tile into the same memory, once every SIMD group is done with the keys.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < BK * D / 4; i += 256) {
            uint r = i / (D / 4), c = i % (D / 4);
            ((threadgroup float4 *)tile)[i] = *((device const float4 *)(vh + (ulong)(j0 + r) * stride) + c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d = 0; d < 16; d++) {
            for (uint c = 0; c < 8; c++) {
                simdgroup_float8x8 V;
                simdgroup_load(V, tile + 8 * c * D + 8 * d, D);
                simdgroup_multiply_accumulate(O[d], S[c], V, O[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / l, the valid rows to memory.
    if (qi < p.queries) {
        const float inv = 1.0f / l;
        device float *o = out + (ulong)qi * stride + head * D + col;
        for (uint d = 0; d < 16; d++) {
            float2 e = elements(O[d]) * inv;
            o[8 * d] = e.x;
            o[8 * d + 1] = e.y;
        }
    }
}
