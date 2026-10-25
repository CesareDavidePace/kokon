#include <metal_stdlib>
using namespace metal;

// The Qwen-Image 2.1 DiT's elementwise and per-row kernels, fp32, following diffusers'
// transformer_qwenimage21.py operation by operation (QwenDiT.swift says which module each one replaces).
//
// Rows are joint-sequence rows of `columns` floats. A modulated kernel reads its per-channel vector from
// `prefixMod` for rows < `targetStart` (text and reference images read t = 0) and from `targetMod` after.

constant uint ROW_THREADS = 256;

struct RowParams {
    uint columns;
    uint targetStart;
    float eps;
};

// Sum over the 256 threads of a threadgroup.
static inline float row_sum(float v, threadgroup float *shared, uint tid) {
    v = simd_sum(v);
    if (tid % 32 == 0) shared[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < ROW_THREADS / 32; i++) total += shared[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// out = LayerNorm(x) (no affine) * onePlusScale[band]: img_norm1/2 + _modulate, norm_out.
// One threadgroup per row.
kernel void layernorm_modulate(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                               device const float *prefixMod [[buffer(2)]], device const float *targetMod [[buffer(3)]],
                               constant RowParams &p [[buffer(4)]],
                               uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[ROW_THREADS / 32];
    device const float *r = x + (ulong)row * p.columns;
    float s = 0;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) s += r[j];
    float mean = row_sum(s, shared, tid) / float(p.columns);
    float q = 0;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) { float c = r[j] - mean; q += c * c; }
    float inv = rsqrt(row_sum(q, shared, tid) / float(p.columns) + p.eps);
    device const float *m = row < p.targetStart ? prefixMod : targetMod;
    device float *o = out + (ulong)row * p.columns;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) o[j] = ((r[j] - mean) * inv) * m[j];
}

// QwenImage21ZeroCenterRMSNorm: x * rsqrt(mean(x²) + eps) * (w + 1), `onePlusWeight` precomputed.
kernel void rmsnorm_rows(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                         device const float *onePlusWeight [[buffer(2)]], constant RowParams &p [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[ROW_THREADS / 32];
    device const float *r = x + (ulong)row * p.columns;
    float q = 0;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) q += r[j] * r[j];
    float inv = rsqrt(row_sum(q, shared, tid) / float(p.columns) + p.eps);
    device float *o = out + (ulong)row * p.columns;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) o[j] = (r[j] * inv) * onePlusWeight[j];
}

struct HeadParams {
    uint heads;
    uint rows;
    float eps;
};

// attn.norm_q / norm_k (RMSNorm over head_dim 128, weight) then apply_rotary_emb_qwen(use_real=False): adjacent
// pairs times (cos, sin) as a complex product. `rope` is [rows, 64 pairs, (cos, sin)]. One SIMD group per
// (row, head), each thread two pairs.
kernel void head_rmsnorm_rope(device float *x [[buffer(0)]], device const float *weight [[buffer(1)]],
                              device const float *rope [[buffer(2)]], constant HeadParams &p [[buffer(3)]],
                              uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]], uint groups [[simdgroups_per_threadgroup]]) {
    uint unit = group * groups + simd;   // (row, head)
    if (unit >= p.rows * p.heads) return;
    uint row = unit / p.heads;
    device float *h = x + (ulong)unit * 128;
    float4 v = float4(h[lane * 2], h[lane * 2 + 1], h[64 + lane * 2], h[64 + lane * 2 + 1]);
    float inv = rsqrt(simd_sum(dot(v, v)) / 128.0f + p.eps);
    float4 w = float4(weight[lane * 2], weight[lane * 2 + 1], weight[64 + lane * 2], weight[64 + lane * 2 + 1]);
    v = (v * inv) * w;
    device const float *t = rope + (ulong)row * 128;
    uint p0 = lane, p1 = 32 + lane;   // pair indices
    float c0 = t[p0 * 2], s0 = t[p0 * 2 + 1], c1 = t[p1 * 2], s1 = t[p1 * 2 + 1];
    h[lane * 2] = v.x * c0 - v.y * s0;
    h[lane * 2 + 1] = v.x * s0 + v.y * c0;
    h[64 + lane * 2] = v.z * c1 - v.w * s1;
    h[64 + lane * 2 + 1] = v.z * s1 + v.w * c1;
}

struct ResidualParams {
    uint columns;
    uint targetStart;
    uint count;
};

// x += tanh(gate)[band] * o: the block's two residuals, `tanhGate` precomputed per band.
kernel void gated_residual(device float *x [[buffer(0)]], device const float *o [[buffer(1)]],
                           device const float *prefixGate [[buffer(2)]], device const float *targetGate [[buffer(3)]],
                           constant ResidualParams &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    uint row = i / p.columns, j = i % p.columns;
    float g = (row < p.targetStart ? prefixGate : targetGate)[j];
    x[i] = x[i] + g * o[i];
}

// QwenImage21SwiGLUFeedForward: gate = SiLU(gate) * proj, in place in `gate`.
kernel void swiglu(device float *gate [[buffer(0)]], device const float *proj [[buffer(1)]],
                   constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float g = gate[i];
    gate[i] = (g / (1.0f + exp(-g))) * proj[i];
}

// nn.GELU(approximate="tanh"), in place.
kernel void gelu_tanh(device float *x [[buffer(0)]], constant uint &count [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
