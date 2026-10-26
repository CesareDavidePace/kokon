// The VAE decoder's tiles (QwenVAE.swift): a window of channel planes copied between the whole tensor and a tile,
// on the GPU so that every tile of a decode goes in one command buffer.

struct WindowCopy {
    uint channels, rows, cols;
    uint srcH, srcW, srcY, srcX;
    uint dstH, dstW, dstY, dstX;
};

/// dst[c, dstY + y, dstX + x] = src[c, srcY + y, srcX + x] for y < rows, x < cols, c < channels.
kernel void window_copy(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                        constant WindowCopy &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.cols || gid.y >= p.rows || gid.z >= p.channels) return;
    ulong s = (ulong(gid.z) * p.srcH + p.srcY + gid.y) * p.srcW + p.srcX + gid.x;
    ulong d = (ulong(gid.z) * p.dstH + p.dstY + gid.y) * p.dstW + p.dstX + gid.x;
    dst[d] = src[s];
}

/// QwenImage21RMS_norm then SiLU, one pixel a thread over the `p.x` channel planes of `p.y` pixels: x / max(‖x‖,
/// 1e-12) · √C · γ, then v · sigmoid(v). `x` and `out` may be the same buffer.
kernel void vae_norm_silu(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                          device const float *gamma [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.y) return;
    float sum = 0;
    for (uint c = 0; c < p.x; c++) { float v = x[ulong(c) * p.y + i]; sum += v * v; }
    float scale = sqrt(float(p.x)) / max(sqrt(sum), 1e-12f);
    for (uint c = 0; c < p.x; c++) {
        ulong at = ulong(c) * p.y + i;
        float v = x[at] * scale * gamma[c];
        out[at] = v / (1.0f + exp(-v));
    }
}

/// Nearest 2× upsampling of `p.x` planes of `p.y` × `p.z`.
kernel void vae_nearest_up(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                           constant uint3 &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= 2 * p.z || gid.y >= 2 * p.y || gid.z >= p.x) return;
    out[(ulong(gid.z) * 2 * p.y + gid.y) * 2 * p.z + gid.x] = x[(ulong(gid.z) * p.y + gid.y / 2) * p.z + gid.x / 2];
}

struct DupUp { uint inC, outC, ft, h, w; };

/// QwenImage21DupUp3D on a first chunk, added to the upsampler's output `out` [outC, 2h, 2w]: output channel o's
/// pixel (2y + i, 2x + j) repeats input channel q / repeats at (y, x), q = ((o · ft + ft − 1) · 2 + i) · 2 + j.
kernel void vae_dup_up_add(device float *out [[buffer(0)]], device const float *x [[buffer(1)]],
                           constant DupUp &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= 2 * p.w || gid.y >= 2 * p.h || gid.z >= p.outC) return;
    uint repeats = p.outC * p.ft * 4 / p.inC;
    uint q = ((gid.z * p.ft + p.ft - 1) * 2 + (gid.y & 1)) * 2 + (gid.x & 1);
    out[(ulong(gid.z) * 2 * p.h + gid.y) * 2 * p.w + gid.x] += x[(ulong(q / repeats) * p.h + gid.y / 2) * p.w + gid.x / 2];
}

struct AvgDown { uint inC, outC, ft, fs, h, w; };

/// QwenImage21AvgDown3D on a first chunk, added to the down block's output `out` [outC, h, w]: the input [inC, fs·h,
/// fs·w] seen as inC · ft · fs² channels ((c · ft + t) · fs + i) · fs + j, of which `group` consecutive ones average
/// into each output channel. With ft = 2 the single frame is padded by a zero frame before it (t = 0), which counts in
/// the average.
kernel void vae_avg_down_add(device float *out [[buffer(0)]], device const float *x [[buffer(1)]],
                             constant AvgDown &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= p.outC) return;
    uint group = p.inC * p.ft * p.fs * p.fs / p.outC;
    float sum = 0;
    for (uint k = 0; k < group; k++) {
        uint index = gid.z * group + k;
        uint j = index % p.fs, i = (index / p.fs) % p.fs, t = (index / (p.fs * p.fs)) % p.ft, c = index / (p.fs * p.fs * p.ft);
        if (t + 1 < p.ft) continue;   // the padded frame
        sum += x[(ulong(c) * p.fs * p.h + gid.y * p.fs + i) * p.fs * p.w + gid.x * p.fs + j];
    }
    out[(ulong(gid.z) * p.h + gid.y) * p.w + gid.x] += sum / float(group);
}

/// QwenImage21RMS_norm alone (the attention block's), one pixel a thread: x / max(‖x‖, 1e-12) · √C · γ.
kernel void vae_norm(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                     device const float *gamma [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.y) return;
    float sum = 0;
    for (uint c = 0; c < p.x; c++) { float v = x[ulong(c) * p.y + i]; sum += v * v; }
    float scale = sqrt(float(p.x)) / max(sqrt(sum), 1e-12f);
    for (uint c = 0; c < p.x; c++) {
        ulong at = ulong(c) * p.y + i;
        out[at] = x[at] * scale * gamma[c];
    }
}

/// The normalized latent back to the VAE's: x · std + mean per channel (`p.x` channels of `p.y` pixels).
kernel void vae_latent_affine(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                              device const float *std [[buffer(2)]], device const float *mean [[buffer(3)]],
                              constant uint2 &p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.y || gid.y >= p.x) return;
    ulong at = ulong(gid.y) * p.y + gid.x;
    out[at] = x[at] * std[gid.y] + mean[gid.y];
}
