# Qwen Local Studio

Local image generation and editing for Windows, built on
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and Qwen-Image 2.1.
Everything runs on this PC: no telemetry, no uploads, no cloud calls. All services
listen on `127.0.0.1` only.

- Text to image, edit, multi-reference, inpaint, outpaint (expand), upscale
- One diffusion pass per image, then ESRGAN (UltraSharp by default) instead of plain
  resizing whenever the result is enlarged to its final size
- Automatic mode, resolution and memory decisions; advanced controls stay hidden
- Persistent local gallery with metadata (also embedded in the PNGs)
- MCP server for Claude Code and other MCP hosts, plus a stable local HTTP API

## Architecture

```
Qwen Local Studio.exe (Tauri 2 + React)      qls-mcp.exe (MCP, stdio)      scripts
        │  loopback HTTP + SSE, token               │                        │
        └──────────────────────────────┬────────────┴────────────────────────┘
                                       ▼
                     qls-service.exe  —  stable API  http://127.0.0.1:39151/api/v1
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
| `crates/qls-service` | Headless service and stable HTTP API; single instance; lifecycle |
| `src-tauri`, `src/` | Desktop shell (Rust) and UI (React + TypeScript) |
| `mcp/` | MCP server (TypeScript, official SDK) |

### Verified backend facts (stable-diffusion.cpp commit `3f8527a`)

- Native async API: `POST /sdcpp/v1/img_gen` → `GET /sdcpp/v1/jobs/{id}`; cancel works only for
  queued jobs (`cancel_generating: false`). Cancelling a running job therefore restarts the
  server when this app launched it.
- Qwen-Image 2.1 editing uses `ref_images` (labelled `<image1>`, `<image2>`… in order);
  references and canvas must be multiples of 32. Inpainting uses `init_image` + binary
  `mask_image` (white = regenerate).
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

## Using the app

Drop, paste (Ctrl+V) or add images, describe the result, press Enter.

- **No image** → text to image (aspect from the prompt or the aspect menu)
- **One image** → edit; the original size and aspect ratio are kept
- **Several images** → combine; refer to them as image 1, image 2… (drag to reorder)
- **Brush** → paint a mask, then describe what goes there (inpainting)
- **Expand** → drag the edges or type pixel/percent values (outpainting)

Results offer Edit, Variation, Use as reference, Inpaint, Expand, Upscale, Favorite, Save, Copy
and Show in Explorer. The gallery (grid icon) reopens any previous generation with its recipe.

### Large images and upscaling

Every job runs a single diffusion pass at the working size (Auto: ~1 MP for Fast/Balanced,
native 2K for Quality; or the 1K/1.5K/2K/Source/Custom choice), capped by the VRAM token budget
that the target shares with its reference images. The earlier two-pass mode (≈1 MP render →
ESRGAN → low-denoise refine at 2K) was removed on 2026-10-06: in A/B tests the refine made
edits look worse than a single pass.

Whenever a result is larger than what the model rendered (edits restored to the source size,
outpaint, sizes above the VRAM budget) it is enlarged with ESRGAN ×4 and then resized down to the exact
size; Lanczos is only used for shrinking or factors below 1.1×.

The **Upscale** button (on results and inputs) offers 2×, 4K, the original source size or a
custom width, with two methods:

- **Quick**: ESRGAN only, a few seconds; adds no new content.
- **Detailed**: re-renders fine detail with a low-denoise (0.3) img2img pass at about 2K, then
  ESRGAN to the target. About a minute.

The result is saved as a new gallery item linked to its source. The Details panel shows the
pipeline (e.g. `1376×768 → 5120×2880 (UltraSharp)`).

**HEIC and HDR.** HEIC/HEIF photos can be opened like any other image (picker, drag & drop);
they are decoded by Windows' HEIF codec, so Microsoft's "HEIF Image Extensions" and "HEVC Video
Extensions" must be installed. iPhone photos with an HDR gain map (ISO 21496-1, iOS 18+) show a
**Keep HDR** switch in Advanced (on by default): edits, inpainting and upscales of such a photo are
saved as an **Ultra HDR JPEG** (SDR image + gain map), which Chrome/Edge, Android and recent
Apple systems show in HDR and every other viewer as a normal JPEG. The model itself works in SDR;
the source's gain map is reused and faded out where the edit changed brightness by more than
~¼ stop (gone at 1 stop). Reading the gain map needs `ffmpeg` (7.1+) on PATH; without it, or
when the result's framing differs from the source (outpaint, other aspect ratios), the result is
saved in SDR and the job's notes say why.

**Upscaler model.** Settings → Models offers two ESRGAN models (64 MB each), used for
enlarging results and for the Upscale button: **UltraSharp** (default,
`4x-UltraSharp.pth`) and **Superscale-SP** (`4x_NMKD-Superscale-SP_178000_G.pth`). Both keep
fine texture that RealESRGAN_x4plus smooths away. Put them in
`<stable-diffusion.cpp folder>\models\upscalers\`. The app does not download models: get them
from <https://huggingface.co/uwg/upscaler/tree/main/ESRGAN>. UltraSharp is licensed
CC BY-NC-SA 4.0 (non-commercial). Without the selected model everything still works, with
Lanczos instead of ESRGAN; the setup screen reports whether the model was found.

Data locations:

| What | Where |
| --- | --- |
| Images | `%USERPROFILE%\Pictures\Qwen Local Studio\<date>\` (configurable) |
| Settings | `%APPDATA%\Qwen Local Studio\config.json` |
| Gallery DB, inputs, thumbnails, logs | `%LOCALAPPDATA%\QwenLocalStudio\` |
| API port + token | `%LOCALAPPDATA%\QwenLocalStudio\service.json` |

## MCP (Claude Code)

The installer places `qls-mcp.exe` next to the app. Register it once (user scope, available in
every project):

```powershell
claude mcp add --scope user qwen-local -- "$env:LOCALAPPDATA\Qwen Local Studio\qls-mcp.exe"
```

Equivalent JSON form:

```powershell
claude mcp add-json --scope user qwen-local '{"type":"stdio","command":"C:\\Users\\<you>\\AppData\\Local\\Qwen Local Studio\\qls-mcp.exe","args":[]}'
```

Project-scoped `.mcp.json` example:

```json
{
  "mcpServers": {
    "qwen-local": {
      "type": "stdio",
      "command": "C:\\Users\\<you>\\AppData\\Local\\Qwen Local Studio\\qls-mcp.exe",
      "args": []
    }
  }
}
```

Tools: `generate_image`, `edit_image`, `edit_images`, `outpaint_image`, `inpaint_image`,
`upscale_image`, `get_image_info`, `get_generator_status`. Every result includes the absolute
path of the saved file, its size, seed and generation metadata (including the
ESRGAN pipeline when used). `upscale_image` takes `scale` (default 2) or `width`/`height`
(up to 8192 px) and `method` (`quick` | `detailed`).

Notes:

- The app does not need to be open. The MCP server starts the service, which starts or reuses
  sd-server (first call ≈ 1 min to load the model). Without the desktop window, the model is
  unloaded after 15 idle minutes (Settings → General).
- Paths may be absolute or relative to the project (`CLAUDE_PROJECT_DIR`). Reads and writes
  are limited to the project folder, the app's output folder, and folders listed in
  `QLS_MCP_ALLOWED_DIRS` (semicolon-separated). Existing files are never overwritten.
- Images are **not** returned to the model by default (they stay local). Pass
  `include_preview: true` to include a small JPEG preview.
- Claude Code's default stdio tool timeout is 30 minutes (`MCP_TOOL_TIMEOUT`). Progress is
  reported through MCP progress notifications.

## Stable local API

Base URL: `http://127.0.0.1:39151/api/v1`. Every call except `/ping` needs
`Authorization: Bearer <token>`, where the token comes from `service.json`. Requests carrying a
browser `Origin` header are refused.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/ping` | Liveness (no auth) |
| GET | `/status` | Server state, model, GPU/VRAM, active job, queue |
| POST | `/generate`, `/edit`, `/inpaint`, `/outpaint`, `/upscale` | Simple operations; wait for the result unless `?wait=false` |
| POST | `/jobs` | Full request (see `GenerationRequest` in `planner.rs`) |
| GET | `/jobs/{id}`, `/jobs/{id}/wait?timeout=60` | Job status / long-poll |
| POST | `/jobs/{id}/cancel` | Cancel |
| POST | `/plan` | Dry run: resolved mode, working/output size, notes |
| GET | `/events` | Server-sent events (jobs, server, gallery) |
| GET | `/gallery`, `/gallery/{id}`, `/gallery/{id}/image` | Gallery |
| POST | `/image-info` | Size, aspect, type, embedded generation metadata |

Example (PowerShell):

```powershell
$i = Get-Content "$env:LOCALAPPDATA\QwenLocalStudio\service.json" | ConvertFrom-Json
$h = @{ Authorization = "Bearer $($i.token)" }
Invoke-RestMethod -Method Post "http://127.0.0.1:$($i.port)/api/v1/generate" -Headers $h `
  -ContentType 'application/json' -Body '{"prompt":"a lighthouse at dusk, 16:9","quality":"fast"}'
```

Simple-operation fields: `prompt`, `image` / `images[]`, `mask`, `width`, `height`,
`aspect_ratio`, `quality` (`fast|balanced|quality`), `seed`, `output_path`,
`preserve_dimensions`, `left/right/top/bottom` + `units` (`pixels|percent`) for outpainting,
`scale` and `method` (`quick|detailed`) for upscaling. Full requests (`/jobs`) also accept
`"mode": "upscale"` with `"upscale": {"method", "scale" | "width"/"height"}` and
`"advanced": {"refine": "fast|best"}` (effort of a Detailed upscale).

New settings in `config.json`: `refineMode` (`fast|best`, Detailed upscale effort; `off` is legacy), `upscalersDir` (null = default
folder), `upscalerModel` (`4x-UltraSharp`; configs on the old `RealESRGAN_x4plus` default move to it), `upscaleTileSize` (0 = backend default 128).

## Building

Requirements: Rust (stable, MSVC), Node 22+, VS Build Tools, WebView2.

```powershell
cargo test -p qls-core            # core unit tests
npx vitest run                    # UI logic tests
cd mcp; npx vitest run; cd ..     # MCP tests (includes a stdio protocol test)
npx tauri dev                     # run the desktop app in development
powershell -ExecutionPolicy Bypass -File scripts\build-release.ps1   # installer
```
