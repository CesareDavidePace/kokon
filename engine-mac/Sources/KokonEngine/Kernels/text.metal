#include <metal_stdlib>
using namespace metal;

// The Qwen3-VL text decoder's kernels that the DiT's (dit.metal) do not cover, fp32, following transformers'
// modeling_qwen3_vl.py (QwenTextEncoder.swift says which module each one replaces). HeadParams is dit.metal's.

// self_attn.q_norm / k_norm (RMSNorm over head_dim 128, weight) then apply_rotary_pos_emb: rotate_half pairs
// dimension i with i + 64, `rope` is [rows, 64, (cos, sin)]. One SIMD group per (row, head), each lane the
// dimensions lane, lane + 32, lane + 64, lane + 96.
kernel void head_rmsnorm_rope_neox(device float *x [[buffer(0)]], device const float *weight [[buffer(1)]],
                                   device const float *rope [[buffer(2)]], constant HeadParams &p [[buffer(3)]],
                                   uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                                   uint simd [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]]) {
    uint unit = group * groups + simd;   // (row, head)
    if (unit >= p.rows * p.heads) return;
    uint row = unit / p.heads;
    device float *h = x + (ulong)unit * 128;
    float4 v = float4(h[lane], h[32 + lane], h[64 + lane], h[96 + lane]);
    float inv = rsqrt(simd_sum(dot(v, v)) / 128.0f + p.eps);
    v = (v * inv) * float4(weight[lane], weight[32 + lane], weight[64 + lane], weight[96 + lane]);
    device const float *t = rope + (ulong)row * 128;
    float c0 = t[lane * 2], s0 = t[lane * 2 + 1], c1 = t[(32 + lane) * 2], s1 = t[(32 + lane) * 2 + 1];
    h[lane] = v.x * c0 - v.z * s0;
    h[64 + lane] = v.z * c0 + v.x * s0;
    h[32 + lane] = v.y * c1 - v.w * s1;
    h[96 + lane] = v.w * c1 + v.y * s1;
}

struct RepeatParams {
    uint count;    // output floats: rows × heads × 128
    uint heads;    // query heads
    uint groups;   // query heads per key/value head
};

// repeat_kv: rows of `heads / groups` key or value heads widened to `heads`, each head repeated `groups` times.
kernel void repeat_kv(device const float *kv [[buffer(0)]], device float *out [[buffer(1)]],
                      constant RepeatParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    uint width = p.heads * 128;
    uint row = i / width, c = i % width;
    out[i] = kv[(ulong)row * (width / p.groups) + (c / 128 / p.groups) * 128 + c % 128];
}

// x += y: the decoder layer's two residuals.
kernel void add_inplace(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                        constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    x[i] = x[i] + y[i];
}
