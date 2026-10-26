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

// The fp16 copy of a row kernel's output that the next split product reads (`Coexecution.swift`): rows below
// `halfRows` go to `outHalf` (the GPU's and the AMX's rows, as `pack_half` would cast them), fp32 is written from
// `fullFrom` on (the Neural Engine's rows, which its packing scales from fp32; 0 when every row is read in fp32).
// Rows from `rows` (the real ones) to `halfRows` are the product's padding: zeros in the copy.
struct HalfOut {
    uint rows;
    uint halfRows;
    uint fullFrom;
};

struct NormParams {
    uint columns;
    uint targetStart;
    float eps;
    HalfOut half_out;
};

// out = LayerNorm(x) (no affine) * onePlusScale[band]: img_norm1/2 + _modulate, norm_out.
// One threadgroup per row (max(rows, halfRows) of them).
kernel void layernorm_modulate(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                               device const float *prefixMod [[buffer(2)]], device const float *targetMod [[buffer(3)]],
                               constant NormParams &p [[buffer(4)]], device half *outHalf [[buffer(5)]],
                               uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[ROW_THREADS / 32];
    device half *h = outHalf + (ulong)row * p.columns;
    if (row >= p.half_out.rows) {   // the whole threadgroup: no barrier is skipped by part of it
        for (uint j = tid; j < p.columns; j += ROW_THREADS) h[j] = 0;
        return;
    }
    device const float *r = x + (ulong)row * p.columns;
    float s = 0;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) s += r[j];
    float mean = row_sum(s, shared, tid) / float(p.columns);
    float q = 0;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) { float c = r[j] - mean; q += c * c; }
    float inv = rsqrt(row_sum(q, shared, tid) / float(p.columns) + p.eps);
    device const float *m = row < p.targetStart ? prefixMod : targetMod;
    device float *o = out + (ulong)row * p.columns;
    bool toHalf = row < p.half_out.halfRows, toFull = row >= p.half_out.fullFrom;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) {
        float v = ((r[j] - mean) * inv) * m[j];
        if (toFull) o[j] = v;
        if (toHalf) h[j] = half(v);
    }
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

// QwenImage21SwiGLUFeedForward: gate = SiLU(gate) * proj, in place in `gate`; with a fp16 copy as in
// layernorm_modulate (`columns` floats a row, one thread per value of max(rows, halfRows) rows).
struct SwigluParams {
    uint columns;
    HalfOut half_out;
};

kernel void swiglu(device float *gate [[buffer(0)]], device const float *proj [[buffer(1)]],
                   constant SwigluParams &p [[buffer(2)]], device half *outHalf [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    uint row = i / p.columns;
    if (row >= p.half_out.rows) {
        if (row < p.half_out.halfRows) outHalf[i] = 0;
        return;
    }
    float g = gate[i];
    float v = (g / (1.0f + exp(-g))) * proj[i];
    if (row >= p.half_out.fullFrom) gate[i] = v;
    if (row < p.half_out.halfRows) outHalf[i] = half(v);
}

// nn.GELU(approximate="tanh"), in place.
kernel void gelu_tanh(device float *x [[buffer(0)]], constant uint &count [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

// fp16 -> fp32: the prefix K/V cache read back (QwenDiT.copyPrefix; `pack_half` writes it).
kernel void widen_half(device const half *src [[buffer(0)]], device float *dst [[buffer(1)]],
                       constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) dst[i] = float(src[i]);
}

// The prefix K/V cache in int8 (QwenDiT.cacheFormat q8): one scale per `lanes` × 4 values (8 lanes: 32 values; a
// scale per head of 128 was 2× the error at the same speed), max |x| / 127, codes rounded to nearest. One SIMD group per row and head, 4 values a lane; `units` =
// rows × heads, the grid exactly units × 32 threads.
kernel void pack_q8(device const float4 *src [[buffer(0)]], device char4 *codes [[buffer(1)]],
                    device float *scales [[buffer(2)]], constant uint &units [[buffer(3)]], constant uint &lanes [[buffer(4)]],
                    uint i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if (i / 32 >= units) return;
    float4 x = src[i];
    float m = max(max(fabs(x.x), fabs(x.y)), max(fabs(x.z), fabs(x.w)));
    for (ushort offset = 1; offset < lanes; offset *= 2) m = max(m, simd_shuffle_xor(m, offset));
    codes[i] = char4(rint(x * (m > 0 ? 127 / m : 0)));
    if (lane % lanes == 0) scales[i / lanes] = m / 127;
}

kernel void widen_q8(device float4 *dst [[buffer(0)]], device const char4 *codes [[buffer(1)]],
                     device const float *scales [[buffer(2)]], constant uint &units [[buffer(3)]],
                     constant uint &lanes [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i / 32 < units) dst[i] = float4(codes[i]) * scales[i / lanes];
}

// The Neural Engine's rows of a row kernel's output, written straight into the next split product's input surfaces
// (`Coexecution.swift`, `RowPack`): each row's σ from its largest |value| and the readers' bound L, as `ane_row_scales`
// computes it, then half(v · σ) as `ane_pack_rows` writes it, into `surfaces` tiles of 4 096 columns (the same row
// into each of an N-split product's tiles; a K-split product's parts in turn). Rows from `real` on are the product's
// padding: zeros (σ 1), as packing never-written rows gave.
struct AnePack {
    uint row0;        // the first Neural Engine row, from the kernel's first row
    uint real;        // rows of the kernel that are real, from its first row
    uint surfaces;    // tiles written: 1 to 3
    uint kSplit;      // 1: column j goes to tile j / 4096 (K parts); 0: every tile gets the whole row (N parts)
};

static inline float row_max(float v, threadgroup float *shared, uint tid) {
    v = simd_max(v);
    if (tid % 32 == 0) shared[tid / 32] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = 0;
    for (uint i = 0; i < ROW_THREADS / 32; i++) m = max(m, shared[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

static inline float ane_sigma(float peak, device const float *l0, device const float *l1, device const float *l2) {
    float bound = peak * max(l0[0], max(l1[0], l2[0]));
    return (bound > 0 && isfinite(bound)) ? exp2(floor(log2(32768.0f / bound))) : 1.0f;
}

// layernorm_modulate on the Neural Engine's rows (4 096 columns), packed. One threadgroup per row.
kernel void layernorm_modulate_ane(device const float *x [[buffer(0)]], device const float *prefixMod [[buffer(1)]],
                                   device const float *targetMod [[buffer(2)]], constant NormParams &p [[buffer(3)]],
                                   constant AnePack &a [[buffer(4)]], device float *sigma [[buffer(5)]],
                                   device const float *l0 [[buffer(6)]], device const float *l1 [[buffer(7)]],
                                   device const float *l2 [[buffer(8)]], device half *s0 [[buffer(9)]],
                                   device half *s1 [[buffer(10)]], device half *s2 [[buffer(11)]],
                                   uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[ROW_THREADS / 32];
    uint row = a.row0 + i;
    device half *out[3] = {s0 + (ulong)i * 4096, s1 + (ulong)i * 4096, s2 + (ulong)i * 4096};
    if (row >= a.real) {
        for (uint t = 0; t < a.surfaces; t++) for (uint j = tid; j < 4096; j += ROW_THREADS) out[t][j] = 0;
        if (tid == 0) sigma[i] = 1;
        return;
    }
    device const float *r = x + (ulong)row * p.columns;
    float s = 0;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) s += r[j];
    float mean = row_sum(s, shared, tid) / float(p.columns);
    float q = 0;
    for (uint j = tid; j < p.columns; j += ROW_THREADS) { float c = r[j] - mean; q += c * c; }
    float inv = rsqrt(row_sum(q, shared, tid) / float(p.columns) + p.eps);
    device const float *m = row < p.targetStart ? prefixMod : targetMod;
    float v[16], peak = 0;
    for (uint c = 0; c < 16; c++) {
        uint j = tid + c * ROW_THREADS;
        v[c] = ((r[j] - mean) * inv) * m[j];
        peak = max(peak, fabs(v[c]));
    }
    float sg = ane_sigma(row_max(peak, shared, tid), l0, l1, l2);
    if (tid == 0) sigma[i] = sg;
    for (uint c = 0; c < 16; c++) {
        half h = half(v[c] * sg);
        for (uint t = 0; t < a.surfaces; t++) out[t][tid + c * ROW_THREADS] = h;
    }
}

// swiglu on the Neural Engine's rows (12 288 columns: the MLP out's K parts), packed. One threadgroup per row.
// Padding rows pack the gate's value, which is what `swiglu` leaves in them.
kernel void swiglu_ane(device const float *gate [[buffer(0)]], device const float *proj [[buffer(1)]],
                       constant AnePack &a [[buffer(2)]], device float *sigma [[buffer(3)]],
                       device const float *l0 [[buffer(4)]], device const float *l1 [[buffer(5)]],
                       device const float *l2 [[buffer(6)]], device half *s0 [[buffer(7)]],
                       device half *s1 [[buffer(8)]], device half *s2 [[buffer(9)]],
                       uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[ROW_THREADS / 32];
    uint row = a.row0 + i;
    device const float *g = gate + (ulong)row * 12288, *pr = proj + (ulong)row * 12288;
    float v[48], peak = 0;
    for (uint c = 0; c < 48; c++) {
        uint j = tid + c * ROW_THREADS;
        float x = g[j];
        v[c] = row < a.real ? (x / (1.0f + exp(-x))) * pr[j] : x;
        peak = max(peak, fabs(v[c]));
    }
    float sg = ane_sigma(row_max(peak, shared, tid), l0, l1, l2);
    if (tid == 0) sigma[i] = sg;
    device half *out[3] = {s0 + (ulong)i * 4096, s1 + (ulong)i * 4096, s2 + (ulong)i * 4096};
    for (uint c = 0; c < 48; c++) {
        uint j = tid + c * ROW_THREADS;
        out[j / 4096][j % 4096] = half(v[c] * sg);
    }
}

// swiglu_ane reading gate's and proj's Neural Engine outputs where the products left them (`keepOutput`): each
// value unscaled as `ane_unpack` does ((y / λ) / σ, σ the rows' shared one, then replaced by the MLP out's), the gate
// tiles `g0..g2` and proj tiles `p0..p2` of 4 096 columns each. Value for value the unpacks, `swiglu`, then the packing.
kernel void swiglu_ane_tiles(device const half *g0 [[buffer(0)]], device const half *g1 [[buffer(1)]],
                             device const half *g2 [[buffer(2)]], device const half *p0 [[buffer(3)]],
                             device const half *p1 [[buffer(4)]], device const half *p2 [[buffer(5)]],
                             device const float *lambdaG [[buffer(6)]], device const float *lambdaP [[buffer(7)]],
                             constant AnePack &a [[buffer(8)]], device float *sigma [[buffer(9)]],
                             device const float *l0 [[buffer(10)]], device const float *l1 [[buffer(11)]],
                             device const float *l2 [[buffer(12)]], device half *s0 [[buffer(13)]],
                             device half *s1 [[buffer(14)]], device half *s2 [[buffer(15)]],
                             uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[ROW_THREADS / 32];
    uint row = a.row0 + i;
    float sgIn = sigma[i];   // read before the barriers in row_max; thread 0 rewrites it after them
    device const half *gs[3] = {g0, g1, g2}, *ps[3] = {p0, p1, p2};
    float v[48], peak = 0;
    for (uint c = 0; c < 48; c++) {
        uint j = tid + c * ROW_THREADS, t = j / 4096;
        ulong at = (ulong)i * 4096 + j % 4096;
        float x = (float(gs[t][at]) / lambdaG[j]) / sgIn;
        v[c] = row < a.real ? (x / (1.0f + exp(-x))) * ((float(ps[t][at]) / lambdaP[j]) / sgIn) : x;
        peak = max(peak, fabs(v[c]));
    }
    float sg = ane_sigma(row_max(peak, shared, tid), l0, l1, l2);
    if (tid == 0) sigma[i] = sg;
    device half *out[3] = {s0 + (ulong)i * 4096, s1 + (ulong)i * 4096, s2 + (ulong)i * 4096};
    for (uint c = 0; c < 48; c++) {
        uint j = tid + c * ROW_THREADS;
        out[j / 4096][j % 4096] = half(v[c] * sg);
    }
}

// gated_residual then layernorm_modulate in one pass over each row: x += tanh(gate) · o (written back), then the
// norm of the new x, as the two kernels compute them. Rows from `real` on are padding: no residual, zeros in the fp16
// copy. One threadgroup per row, 4 096 columns.
kernel void residual_layernorm(device float *x [[buffer(0)]], device float *out [[buffer(1)]],
                               device const float *prefixMod [[buffer(2)]], device const float *targetMod [[buffer(3)]],
                               constant NormParams &p [[buffer(4)]], device half *outHalf [[buffer(5)]],
                               device const float *o [[buffer(6)]], device const float *prefixGate [[buffer(7)]],
                               device const float *targetGate [[buffer(8)]],
                               uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[ROW_THREADS / 32];
    device half *h = outHalf + (ulong)row * p.columns;
    if (row >= p.half_out.rows) {
        for (uint j = tid; j < p.columns; j += ROW_THREADS) h[j] = 0;
        return;
    }
    device float *r = x + (ulong)row * p.columns;
    device const float *orow = o + (ulong)row * p.columns;
    device const float *g = row < p.targetStart ? prefixGate : targetGate;
    float v[16], s = 0;
    for (uint c = 0; c < 16; c++) {
        uint j = tid + c * ROW_THREADS;
        v[c] = r[j] + g[j] * orow[j];
        r[j] = v[c];
        s += v[c];
    }
    float mean = row_sum(s, shared, tid) / float(p.columns);
    float q = 0;
    for (uint c = 0; c < 16; c++) { float d = v[c] - mean; q += d * d; }
    float inv = rsqrt(row_sum(q, shared, tid) / float(p.columns) + p.eps);
    device const float *m = row < p.targetStart ? prefixMod : targetMod;
    device float *w = out + (ulong)row * p.columns;
    bool toHalf = row < p.half_out.halfRows, toFull = row >= p.half_out.fullFrom;
    for (uint c = 0; c < 16; c++) {
        uint j = tid + c * ROW_THREADS;
        float y = ((v[c] - mean) * inv) * m[j];
        if (toFull) w[j] = y;
        if (toHalf) h[j] = half(y);
    }
}

// The same on the Neural Engine's rows, o read from the product's output tiles (`y0..y2`: its K parts, summed in
// fp32 and unscaled as `ane_unpack` does, σ the product's, then replaced by the next one's), the norm packed as in
// layernorm_modulate_ane.
kernel void residual_layernorm_ane(device float *x [[buffer(0)]], device const float *prefixMod [[buffer(1)]],
                                   device const float *targetMod [[buffer(2)]], constant NormParams &p [[buffer(3)]],
                                   constant AnePack &a [[buffer(4)]], device float *sigma [[buffer(5)]],
                                   device const float *l0 [[buffer(6)]], device const float *l1 [[buffer(7)]],
                                   device const float *l2 [[buffer(8)]], device half *s0 [[buffer(9)]],
                                   device half *s1 [[buffer(10)]], device half *s2 [[buffer(11)]],
                                   device const half *y0 [[buffer(12)]], device const half *y1 [[buffer(13)]],
                                   device const half *y2 [[buffer(14)]], device const float *lambda [[buffer(15)]],
                                   constant uint &kParts [[buffer(16)]], device const float *prefixGate [[buffer(17)]],
                                   device const float *targetGate [[buffer(18)]],
                                   uint i [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[ROW_THREADS / 32];
    uint row = a.row0 + i;
    device half *out[3] = {s0 + (ulong)i * 4096, s1 + (ulong)i * 4096, s2 + (ulong)i * 4096};
    float sgIn = sigma[i];   // read before the barriers; thread 0 rewrites it after them
    if (row >= a.real) {
        for (uint t = 0; t < a.surfaces; t++) for (uint j = tid; j < 4096; j += ROW_THREADS) out[t][j] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) sigma[i] = 1;
        return;
    }
    device float *r = x + (ulong)row * p.columns;
    device const float *g = row < p.targetStart ? prefixGate : targetGate;
    float v[16], s = 0;
    for (uint c = 0; c < 16; c++) {
        uint j = tid + c * ROW_THREADS;
        ulong at = (ulong)i * 4096 + j;
        float sum = float(y0[at]);
        if (kParts > 1) sum += float(y1[at]);
        if (kParts > 2) sum += float(y2[at]);
        v[c] = r[j] + g[j] * ((sum / lambda[j]) / sgIn);
        r[j] = v[c];
        s += v[c];
    }
    float mean = row_sum(s, shared, tid) / float(p.columns);
    float q = 0;
    for (uint c = 0; c < 16; c++) { float d = v[c] - mean; q += d * d; }
    float inv = rsqrt(row_sum(q, shared, tid) / float(p.columns) + p.eps);
    device const float *m = row < p.targetStart ? prefixMod : targetMod;
    float peak = 0;
    for (uint c = 0; c < 16; c++) {
        v[c] = ((v[c] - mean) * inv) * m[tid + c * ROW_THREADS];
        peak = max(peak, fabs(v[c]));
    }
    float sg = ane_sigma(row_max(peak, shared, tid), l0, l1, l2);
    if (tid == 0) sigma[i] = sg;
    for (uint c = 0; c < 16; c++) {
        half hv = half(v[c] * sg);
        for (uint t = 0; t < a.surfaces; t++) out[t][tid + c * ROW_THREADS] = hv;
    }
}
