#include <metal_stdlib>
using namespace metal;

// The Qwen3-VL vision tower's kernels, fp32, following transformers' modeling_qwen3_vl.py (QwenVisionTower.swift says
// which module each one replaces). Rows are patches (or merged groups of 4 patches) of `columns` floats.

struct AffineParams {
    uint columns;
    float eps;
};

// nn.LayerNorm(columns, eps) with weight and bias: norm1/norm2, the mergers' norms. One threadgroup of 256 per row;
// the mean first, then the variance around it (two passes, as torch's CPU kernel's result).
kernel void layernorm_affine(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                             device const float *weight [[buffer(2)]], device const float *bias [[buffer(3)]],
                             constant AffineParams &p [[buffer(4)]],
                             uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[8];
    device const float *r = x + (ulong)row * p.columns;
    device float *o = out + (ulong)row * p.columns;
    float s = 0;
    for (uint j = tid; j < p.columns; j += 256) s += r[j];
    s = simd_sum(s);
    if (tid % 32 == 0) shared[tid / 32] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0;
    for (uint i = 0; i < 8; i++) mean += shared[i];
    mean /= float(p.columns);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = 0;
    for (uint j = tid; j < p.columns; j += 256) { float d = r[j] - mean; v += d * d; }
    v = simd_sum(v);
    if (tid % 32 == 0) shared[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0;
    for (uint i = 0; i < 8; i++) var += shared[i];
    float inv = rsqrt(var / float(p.columns) + p.eps);
    for (uint j = tid; j < p.columns; j += 256) o[j] = (r[j] - mean) * inv * weight[j] + bias[j];
}

// x[r, c] += b[c]: the Linear layers' bias after the product (count = rows × columns).
kernel void bias_rows(device float *x [[buffer(0)]], device const float *b [[buffer(1)]],
                      constant uint2 &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.x) return;
    x[i] += b[i % p.y];
}

// nn.GELU() (erf), in place, the mergers' activation: 0.5 x erfc(−x / √2), erfc as Numerical Recipes' erfcc (fractional
// error below 1.2e-7 everywhere, so the tail for negative x keeps its relative precision).
kernel void gelu_erf(device float *x [[buffer(0)]], constant uint &count [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float v = x[i];
    float z = -v * 0.7071067811865476f;
    float a = fabs(z), t = 1.0f / (1.0f + 0.5f * a);
    float e = t * precise::exp(-a * a - 1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f +
              t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f +
              t * (-0.82215223f + t * 0.17087277f)))))))));
    float tail = z >= 0 ? e : 2.0f - e;
    x[i] = 0.5f * v * tail;
}

// Qwen3VLVisionAttention's qkv split and apply_rotary_pos_emb_vision: q, k, v as [rows, 16 heads × 72], the layout
// `flash_attention_72` reads and attn.proj's input has. rotate_half pairs dimension i with i + 36; `rope` is
// [rows, 36, (cos, sin)]. One thread per (row, head, j < 36), rotating the pair (j, j + 36).
kernel void vision_rope(device const float *qkv [[buffer(0)]], device float *q [[buffer(1)]],
                        device float *k [[buffer(2)]], device float *v [[buffer(3)]],
                        device const float *rope [[buffer(4)]], constant uint &rows [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    uint j = i % 36, head = (i / 36) % 16, row = i / 576;
    if (row >= rows) return;
    device const float *in = qkv + (ulong)row * 3456 + head * 72;
    ulong o = (ulong)row * 1152 + head * 72;
    float c = rope[((ulong)row * 36 + j) * 2], s = rope[((ulong)row * 36 + j) * 2 + 1];
    float q0 = in[j], q1 = in[j + 36], k0 = in[1152 + j], k1 = in[1152 + j + 36];
    q[o + j] = q0 * c - q1 * s;
    q[o + j + 36] = q1 * c + q0 * s;
    k[o + j] = k0 * c - k1 * s;
    k[o + j + 36] = k1 * c + k0 * s;
    v[o + j] = in[2304 + j];
    v[o + j + 36] = in[2304 + j + 36];
}
