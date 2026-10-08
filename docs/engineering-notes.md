# Engineering notes

How Kokon works inside, and the backend facts its design depends on. For using the app, see the
[README](../README.md).

## Architecture

```
Kokon.exe (Tauri 2 + React)      kokon-mcp.exe (MCP, stdio)      scripts
        │  loopback HTTP + SSE, token               │                        │
        └──────────────────────────────┬────────────┴────────────────────────┘
                                       ▼
                     kokon-service.exe  —  stable API  http://127.0.0.1:39151/api/v1
                     (engine: planner · preprocessing · job queue · gallery)
                                       │  native async API /sdcpp/v1/*
                                       ▼
                     sd-server.exe (stable-diffusion.cpp)  127.0.0.1:39150
```

| Crate / package | Responsibility |
| --- | --- |
| `crates/qls-core/src/geometry.rs` | Pure dimension math: 32-alignment, working canvas, outpaint/inpaint geometry |
| `imaging.rs` | EXIF orientation, alpha-safe resize, padding, masks, compositing, PNG metadata/ICC |
| `planner.rs` | Mode detection, resolution policy, upscale targets, VRAM budget |
| `pipeline.rs` | Builds backend requests, post-processes results (crop back, ESRGAN tiling, composite) |
| `sdcpp.rs` | Inference adapter — the only code that knows sd-server's schema |
| `process.rs` | sd-server launch/reuse/stop, log capture, truthful progress parsing |
| `engine.rs` | Async job queue, cancellation, OOM retry, saving |
| `store.rs` | SQLite gallery + content-addressed inputs |
| `paths.rs` | Path validation (no UNC/device/ADS/traversal, never overwrite) |
| `catalog.rs`, `installer.rs` | Pinned downloads (URLs, sizes, SHA-256), hardware presets, parallel resumable downloader |
| `crates/qls-service` | Headless service and stable HTTP API; single instance; lifecycle |
| `src-tauri`, `src/` | Desktop shell (Rust) and UI (React + TypeScript) |
| `mcp/` | MCP server (TypeScript, official SDK) |

Internal names (`qls-core`, `qls-service`, `QLS_*` environment variables) date from the project's
earlier name, Qwen Local Studio, and are kept on purpose.

## Verified backend facts (stable-diffusion.cpp commit `3f8527a`)

- Native async API: `POST /sdcpp/v1/img_gen` → `GET /sdcpp/v1/jobs/{id}`; cancel works only for
  queued jobs (`cancel_generating: false`). Cancelling a running job therefore restarts the
  server when this app launched it.
- Qwen-Image 2.1 editing uses `ref_images` (labelled `<image1>`, `<image2>`… in order);
  references and canvas must be multiples of 32. Inpainting uses `init_image` + binary
  `mask_image` (white = regenerate). For Qwen 2.1 that mask is only a latent blend (it
  discards changes outside it; the model never sees it), so the app also sends the area
  marked in color on a second reference and prompts "the area marked in red in `<image2>`"
  (the official 2.1 way to point at a region). Removals blur the content out of `<image1>`
  first so the model rebuilds background instead of copying the object back.
- Recommended settings follow the official Qwen-Image 2.1 release (QwenLM/Qwen-Image-2.1
  README, diffusers `QwenImage21Pipeline`): `euler`, **no CFG** (1.0, "meant to be sampled
  without guidance"), 40 steps, native 2K sizes; sd-server picks the same resolution-dependent
  flow schedule (exponential shift 0.5–0.9, terminal 0.02). sd.cpp's own docs use CFG 6, which
  runs a second, unconditional model pass every step; the app used that until 2026-10-07.
  At CFG 1 the negative prompt is ignored.
  Presets: Fast 20 steps at ~1 MP (a shortcut, not official); Balanced 40 steps at ~1 MP;
  Quality 40 steps at the official 2K buckets (2048², 2752×1536…; edits share that budget
  with their reference images, so they land near 2 MP on 16 GB).
- Default launch: `--backend cuda0 --params-backend te=cpu --fa --sage-attn --vae-tiling
  --vae-tile-size 768 --vae-tile-overlap 0.125 --max-vram -1` plus `--hires-upscalers-dir`
  when the upscalers folder exists (text-encoder weights in RAM, computed on the GPU). Measured
  on an RTX 5070 Ti: running the text encoder fully on the CPU made a one-image edit take 240 s
  versus 61 s; SageAttention makes 2K steps ~32 % faster; untiled 2K VAE decodes OOM and fall
  back to 256 px tiles (~22 s), 768 px tiles take ~4 s.
- Request objects are merged field by field over the server's launch options, so omitted
  fields keep the CLI values (per-request VAE tiling is only sent to turn it off).
- `img_gen` accepts a `hires` object (second pass: upscale with a built-in filter or an ESRGAN
  model from `--hires-upscalers-dir`, then re-sample at `denoising_strength`); reference
  latents and inpaint masks are carried into that pass. `cache_mode`/`cache_option`
  (e.g. `easycache`, `threshold=0.2`) work per request. `POST /sdcpp/v1/upscale`
  (`{image, upscaler}`) runs ESRGAN synchronously, up to 8192 px per side.
- Qwen-Image 2.1 always decodes an alpha channel (≈1 % of pixels at alpha 247–254 on opaque
  images); outputs are made opaque unless transparency is requested.

## Resolution and upscaling policy

Every job runs a single diffusion pass at the working size (Auto: ~1 MP for Fast/Balanced,
native 2K for Quality; or the 1K/1.5K/2K/Source/Custom choice), capped by the VRAM token budget
that the target shares with its reference images. The earlier two-pass mode (≈1 MP render →
ESRGAN → low-denoise refine at 2K) was removed on 2026-10-06: in A/B tests the refine made
edits look worse than a single pass.

Whenever a result is larger than what the model rendered (edits restored to the source size,
outpaint, sizes above the VRAM budget) it is enlarged with ESRGAN ×4 and then resized down to
the exact size; Lanczos is only used for shrinking or factors below 1.1×.

The **Upscale** action has two methods: **Quick** (ESRGAN only) and **Detailed** (a low-denoise
0.3 img2img pass at about 2K, then ESRGAN to the target).

## HDR

The source's ISO 21496-1 gain map is reused and faded out where the edit changed brightness by
more than ~¼ stop (gone at 1 stop). Reading the gain map needs `ffmpeg` 7.1+ on PATH; without
it, or when the result's framing differs from the source (outpaint, other aspect ratios), the
result is saved in SDR and the job's notes say why.

## Downloads

Sources, sizes and SHA-256 sums are pinned in `crates/qls-core/src/catalog.rs`. Files ≥ 16 MB
download over 8 parallel connections in 8 MB pieces (HTTP Range, positional writes into a
preallocated `.part`, finished pieces recorded in `.part.segments` so pause or a crash can
resume). When nothing is left to start, idle connections also fetch the piece with the most
left; the first to finish wins. A server without range support falls back to one connection.
A file that fails twice, or a link that moved, is reported with its download page and target
path so it can be fetched by hand.

## Settings not shown in the UI

In `%APPDATA%\Kokon\config.json`: `hardwarePreset` (`12gb|16gb|24gb|32gb`, written by the
setup screen), `vramOverrideMib` (plan sizes as if the GPU had this much VRAM; 0 = detect;
for testing), `refineMode` (`fast|best`, Detailed upscale effort; `off` is legacy),
`upscalersDir` (null = default folder), `upscalerModel` (`4x-UltraSharp`; configs on the old
`RealESRGAN_x4plus` default move to it), `upscaleTileSize` (0 = backend default 128),
`prefixCacheType`, `eagerLoad`, `sageAttention`.

## Testing smaller cards on a bigger one

`scripts\dev\vram-ballast.py --card-mib 12282` holds the VRAM a 12 GB card does not have (it
writes the memory every 0.5 s so Windows keeps it resident), and `vramOverrideMib` makes the
planner size images for 12 GB. `scripts\dev\test-12gb.ps1` does both and runs text to image,
edit, inpaint, outpaint, multi-reference and upscale through an isolated service, recording
time, working size, peak VRAM, memory spilled to shared GPU memory and out-of-memory retries
into `docs\test-results\`. Bandwidth and compute stay those of the bigger card, so times are
optimistic: a real 12 GB card (RTX 3060/4070) has to confirm them.
