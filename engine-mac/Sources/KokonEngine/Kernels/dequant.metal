#include <metal_stdlib>
using namespace metal;

// GGUF weights unpacked on the GPU, ggml's formulas (dequantize_row_* in ggml-quants.c), value for value
// against Dequant.block on the CPU (QuantTests).
//
// A tensor is `rows` ggml rows of `columns` values (a Linear weight: rows = out features, columns = in).
// One thread unpacks one block of one row. Two layouts out:
//   transposed = 0: [rows, columns], the file's order;
//   transposed = 1: [columns, rows], the [k, n] layout MPS multiplies without transposing (2x faster than
//                   transposeRight on K = 10 240, measured by Siliconed). Neighbouring threads take
//                   neighbouring rows of the same block, so their writes are contiguous.

struct DequantParams {
    uint rows;
    uint columns;
    uint blocksPerRow;
    uint rowBytes;
    uint transposed;
};

static inline float load_half(device const uchar *p, uint at) {
    return float(*((device const half *)(p + at)));
}

// get_scale_min_k4
static inline void scale_min(uint j, device const uchar *q, thread float &sc, thread float &m) {
    if (j < 4) {
        sc = float(q[j] & 63);
        m = float(q[j + 4] & 63);
    } else {
        sc = float((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = float((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

template <typename T>
static inline void put(device T *out, uint base, uint stride, uint j, float v) {
    out[base + j * stride] = T(v);
}

template <typename T>
static inline void block_q4_K(device const uchar *x, device T *out, uint base, uint stride) {
    float d = load_half(x, 0), dmin = load_half(x, 2);
    device const uchar *scales = x + 4;
    device const uchar *q = x + 16;
    for (uint s = 0; s < 8; s += 2) {
        float sc1, m1, sc2, m2;
        scale_min(s, scales, sc1, m1);
        scale_min(s + 1, scales, sc2, m2);
        float d1 = d * sc1, n1 = dmin * m1, d2 = d * sc2, n2 = dmin * m2;
        uint o = s * 32;
        for (uint l = 0; l < 32; l++) put(out, base, stride, o + l, d1 * float(q[l] & 0xF) - n1);
        for (uint l = 0; l < 32; l++) put(out, base, stride, o + 32 + l, d2 * float(q[l] >> 4) - n2);
        q += 32;
    }
}

template <typename T>
static inline void block_q5_K(device const uchar *x, device T *out, uint base, uint stride) {
    float d = load_half(x, 0), dmin = load_half(x, 2);
    device const uchar *scales = x + 4;
    device const uchar *qh = x + 16;
    device const uchar *ql = x + 48;
    uchar u1 = 1, u2 = 2;
    for (uint s = 0; s < 8; s += 2) {
        float sc1, m1, sc2, m2;
        scale_min(s, scales, sc1, m1);
        scale_min(s + 1, scales, sc2, m2);
        float d1 = d * sc1, n1 = dmin * m1, d2 = d * sc2, n2 = dmin * m2;
        uint o = s * 32;
        for (uint l = 0; l < 32; l++)
            put(out, base, stride, o + l, d1 * float((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - n1);
        for (uint l = 0; l < 32; l++)
            put(out, base, stride, o + 32 + l, d2 * float((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - n2);
        ql += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
}

template <typename T>
static inline void block_q6_K(device const uchar *x, device T *out, uint base, uint stride) {
    float d = load_half(x, 208);
    device const uchar *ql = x;
    device const uchar *qh = x + 128;
    device const char *sc = (device const char *)(x + 192);
    for (uint n = 0; n < 256; n += 128) {
        for (uint l = 0; l < 32; l++) {
            uint i = l / 16;
            int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            put(out, base, stride, n + l, d * float(sc[i]) * float(q1));
            put(out, base, stride, n + l + 32, d * float(sc[i + 2]) * float(q2));
            put(out, base, stride, n + l + 64, d * float(sc[i + 4]) * float(q3));
            put(out, base, stride, n + l + 96, d * float(sc[i + 6]) * float(q4));
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

template <typename T>
static inline void block_q8_0(device const uchar *x, device T *out, uint base, uint stride) {
    float d = load_half(x, 0);
    device const char *qs = (device const char *)(x + 2);
    for (uint j = 0; j < 32; j++) put(out, base, stride, j, float(qs[j]) * d);
}

template <typename T>
static inline void block_q4_0(device const uchar *x, device T *out, uint base, uint stride) {
    float d = load_half(x, 0);
    device const uchar *qs = x + 2;
    for (uint j = 0; j < 16; j++) {
        put(out, base, stride, j, float(int(qs[j] & 0xF) - 8) * d);
        put(out, base, stride, j + 16, float(int(qs[j] >> 4) - 8) * d);
    }
}

template <typename T>
static inline void block_f32(device const uchar *x, device T *out, uint base, uint stride) {
    put(out, base, stride, 0, *((device const float *)x));
}

template <typename T>
static inline void block_f16(device const uchar *x, device T *out, uint base, uint stride) {
    put(out, base, stride, 0, load_half(x, 0));
}

template <typename T>
static inline void block_bf16(device const uchar *x, device T *out, uint base, uint stride) {
    put(out, base, stride, 0, as_type<float>(uint(*((device const ushort *)x)) << 16));
}

// Thread `gid` -> (row, block). Transposed: rows vary fastest, so a SIMD group writes one contiguous run.
#define DEQUANT_KERNEL(NAME, BLOCK, VALUES, BYTES, T)                                                     \
    kernel void NAME(device const uchar *src [[buffer(0)]], device T *dst [[buffer(1)]],                  \
                     constant DequantParams &p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {     \
        if (gid >= p.rows * p.blocksPerRow) return;                                                       \
        uint r, b;                                                                                        \
        if (p.transposed) { r = gid % p.rows; b = gid / p.rows; }                                         \
        else { r = gid / p.blocksPerRow; b = gid % p.blocksPerRow; }                                      \
        device const uchar *x = src + r * p.rowBytes + b * (BYTES);                                       \
        uint base = p.transposed ? (b * (VALUES)) * p.rows + r : r * p.columns + b * (VALUES);            \
        uint stride = p.transposed ? p.rows : 1;                                                          \
        BLOCK<T>(x, dst, base, stride);                                                                   \
    }

DEQUANT_KERNEL(dequant_q4_K_f32, block_q4_K, 256, 144, float)
DEQUANT_KERNEL(dequant_q5_K_f32, block_q5_K, 256, 176, float)
DEQUANT_KERNEL(dequant_q6_K_f32, block_q6_K, 256, 210, float)
DEQUANT_KERNEL(dequant_q8_0_f32, block_q8_0, 32, 34, float)
DEQUANT_KERNEL(dequant_q4_0_f32, block_q4_0, 32, 18, float)
DEQUANT_KERNEL(dequant_f32_f32, block_f32, 1, 4, float)
DEQUANT_KERNEL(dequant_f16_f32, block_f16, 1, 2, float)
DEQUANT_KERNEL(dequant_bf16_f32, block_bf16, 1, 2, float)

DEQUANT_KERNEL(dequant_q4_K_f16, block_q4_K, 256, 144, half)
DEQUANT_KERNEL(dequant_q5_K_f16, block_q5_K, 256, 176, half)
DEQUANT_KERNEL(dequant_q6_K_f16, block_q6_K, 256, 210, half)
DEQUANT_KERNEL(dequant_q8_0_f16, block_q8_0, 32, 34, half)
DEQUANT_KERNEL(dequant_q4_0_f16, block_q4_0, 32, 18, half)
DEQUANT_KERNEL(dequant_f32_f16, block_f32, 1, 4, half)
DEQUANT_KERNEL(dequant_f16_f16, block_f16, 1, 2, half)
DEQUANT_KERNEL(dequant_bf16_f16, block_bf16, 1, 2, half)
