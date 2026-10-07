# Speed & upscaling: results log and implementation plan

Status as of 2026-10-06. Hardware: RTX 5070 Ti 16 GB, sd-server/sd-cli in `C:\AI\sdcpp`
(stable-diffusion.cpp commit 3f8527a), Qwen Image 2.1 UC Q4_K_M + Qwen3VL-8B Q4_K_M TE.
Benchmark script: `C:\AI\sdcpp\bench\bench-2k.ps1` (`-Only a,b`), outputs + logs in `C:\AI\sdcpp\bench\out`.

## 1. Results so far

### 1.1 Text-to-image 2048x1152, 20 steps, cfg 6, euler, seed 42 (sd-cli, incl. model load)

| Variant | Total | s/step | Notes |
|---|---|---|---|
| `base` (`--fa`) | 139 s | 5.63 | Over-sharpened / "oil painting" look at native 2K |
| `sage` (+`--sage-attn`) | 104 s | 3.81 | Near-identical image to base |
| `easycache` (threshold=0.2) | 90 s | 5.57 | Skips 10/20 steps |
| `sage_ec` | 64 s | 3.80 | Same composition, slightly softer |
| `hires` (1024x576 → x2 latent, 10 steps, denoise 0.5) | 87 s | 3.86 | Different, more photographic |
| `hires_esrgan` (1376x768 → ESRGAN → 2048x1152, 8 steps, denoise 0.35) | 104 s | — | Best look; but decode wasted 22.8 s |
| `hires_vt768` (same + `--vae-tiling --vae-tile-size 768 --vae-tile-overlap 0.125`) | **86.8 s** | — | Best quality/time |
| `hires_vt1024` (tile 1024) | 88.8 s | — | Decode 6.1 s (768: 3.9 s) |
| `hires_fast` (tile 1024 + EasyCache + 6 refine steps) | **57.5 s** | — | Very close to vt768, slightly softer fine detail |

Breakdown `hires_vt768`: base 20 steps @1376x768 33.5 s (1.63 s/it with sage) · hires prep
(decode + ESRGAN + re-encode) ~16 s · 8 refine steps @2K 30.8 s (3.8 s/it) · decode 3.9 s.
Breakdown `hires_fast`: base 17.4 s (EasyCache skipped 10/20) · refine 15.5 s (skipped 2/6) · prep ~15 s · decode 6.2 s.

### 1.2 Key findings

- **SageAttention** (`--sage-attn`): ~32% faster steps at 2K, no visible quality cost. Works on Blackwell (sm_120).
- **VAE decode at 2K always OOMs** untiled (needs ~18 GB with DiT weights resident), then the backend
  retries with 256 px tiles → ~22 s per 2K image. 768 px tiles + overlap 0.125 → ~4 s. `--vae-tiling`
  alone keeps 256 px tiles and also slows encode (+8 s), so the tile size matters.
- **ESRGAN (RealESRGAN_x4plus.pth)** via `sd-cli -M upscale`: 1376x768 → 5504x3072 in 13 s *including*
  process start + model load; output clearly sharper/cleaner than Lanczos (slightly "painted" fine texture).
  Model at `C:\AI\sdcpp\models\upscalers\RealESRGAN_x4plus.pth`; as a hires upscaler its name is the
  file stem (`--hires-upscaler RealESRGAN_x4plus` with `--hires-upscalers-dir`). Verified working.
- **Native 2K looks over-processed**; 1 MP base + 2K low-denoise refine looks natural. The model is best near 1 MP.
- **Edit CFG**: at cfg 6 edits can collapse to near-black (Spider-Man "dark version" test); user A/B
  showed detail increases monotonically as cfg drops 6 → 4 → 2. Clear, explicit edit prompts help.
- **Steps**: 10 steps + cfg 6 produced an almost blank flat-vector logo (L range 245–255). Keep ≥20.
- **Current upscaling in the app = none**: everything renders at working size; `pipeline::postprocess`
  Lanczos-resizes to output. Edits at Balanced work at ~1 MP and are stretched back (5120x2880 source
  → worked at 1376x768 → 3.7x Lanczos stretch = soft).

### 1.3 sd-server capabilities found in the binary (strings, not yet exercised over HTTP)

- `POST /sdcpp/v1/upscale` — ESRGAN upscale endpoint; needs RGB ESRGAN models in `--hires-upscalers-dir`
  ("no RGB ESRGAN upscaler models are available; start the server with --hires-upscalers-dir"),
  max 8192x8192, errors: "image is required", "no compatible image upscaler called …". Capabilities
  JSON has `upscalers`, `max_upscale_width/height`, feature `image_upscale`. Request field names are
  unknown — inspect via `GET /sdcpp/v1/capabilities` and trial requests.
- `img_gen` request has a `hires` object: `enabled, upscaler, model_path, scale, target_width,
  target_height, steps, denoising_strength, upscale_tile_size, custom_sigmas`.
- `img_gen` also has `vae_tiling_params` (already in `sdcpp.rs`) and strings `cache_mode`,
  `cache_option`, `cache_params` (per-request EasyCache plausible, unverified).

## 2. Already implemented (not yet in a release build — run `scripts\build-release.ps1`)

- `AppConfig.sage_attention` (default true) → `--sage-attn`; Settings switch.
- `AppConfig.vae_tile_size` (default 768, 0 = off) → `--vae-tiling --vae-tile-size N --vae-tile-overlap 0.125`; Settings field.
- `RecommendedSettings.cfg_edit` (default 4.0) used for every non-Generate mode; t2i stays at `cfg` 6.0.
- All 92 qls-core tests pass; UI typechecks.

**Known bug to fix first:** `planner.rs` sets `vae_tiling` automatically for areas > 2.7 MP and
`pipeline.rs:120` then sends `VaeTilingParams { tile_size_w: 0, tile_size_h: 0, target_overlap: 0.5 }`,
which probably overrides the server's 768 px default with 256 px tiles. Send the configured tile size
(768, overlap 0.125) or omit the field when the server flag already applies.

## 3. Implementation plan

### Phase 0 — verify over HTTP (short, before coding the client)
1. Start sd-server with the app's args + `--hires-upscalers-dir C:\AI\sdcpp\models\upscalers`; dump
   `GET /sdcpp/v1/capabilities` (look for `upscalers`, `image_upscale`, hires/cache features).
2. Find the `/sdcpp/v1/upscale` body schema (try `{image, upscaler, scale|width/height}`; read error messages).
3. Confirm `hires` works in `img_gen` for t2i **and** for edits with `ref_images` (does the refine pass keep refs?).
4. Check whether `cache_mode`/`cache_option` are accepted per request.
Long runs (model loads, 2K jobs) are executed by the user: give copy-paste commands and wait for output.

### Phase 1 — backend plumbing (qls-core)
- `config.rs`: `upscalers_dir` (default `<sdcpp>\models\upscalers`), `upscaler_model` (default
  `RealESRGAN_x4plus`); pass `--hires-upscalers-dir` in `process.rs` when the dir exists.
- `sdcpp.rs`: `HiresParams` on `ImgGenRequest` (skip when None); `upscale()` client for `/sdcpp/v1/upscale`;
  parse `upscalers` from capabilities; optional per-request cache params if Phase 0 confirms.
- Diagnostics/setup: report whether an ESRGAN model is present, with the download URL
  `https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth` (64 MB).
  The app does not download models itself (Setup.tsx says so) — keep that, show instructions.
- Fix the per-request VAE tiling bug above.

### Phase 2 — two-pass Quality for text-to-image (planner + pipeline)
- When the target area is > ~1.3 MP (Quality / 2K / custom large): base pass at ~1 MP with the same
  aspect (aligned to 32), `hires` to the target size with the ESRGAN upscaler, denoise 0.35.
- Two settings: **Best** = 8 refine steps, no cache (~86 s at 2K) · **Fast** = 6 refine steps + EasyCache
  threshold 0.2 (~55 s). Expose as a preference (default Fast?) and in Advanced.
- Targets above 2K (e.g. 4K): two-pass to 2K, then ESRGAN to the final size (Phase 3 path), never a
  single diffusion pass > ~2.4 MP.
- Update `estimate_seconds` to model base + refine + ESRGAN + decode; record `working_*` and hires
  info in the gallery metadata and the Details panel ("rendered 1376x768 → refined 2048x1152 → upscaled …").

### Phase 3 — ESRGAN instead of Lanczos when enlarging
- `pipeline::postprocess`: when output is larger than the working content by > ~1.1x (edits with
  preserve_dimensions, outpaint final size, any custom large size), upscale with `/sdcpp/v1/upscale`
  (x4) and then Lanczos *down* to the exact size. Lanczos only for shrinking or tiny factors.
- Inpaint composites keep their current path (patch is generated near native size).

### Phase 4 — Upscale action in the UI
- New toolbar button on any image: **Quick** (ESRGAN only, seconds) and **Detailed** (img2img refine at
  ~2K with denoise ~0.3, cfg_edit, then ESRGAN to the target) with target presets 2x / 4K / original
  source size / custom; result saved as a child of the source (parent_id).
- For the 3.7x edit case: edit at ~1 MP → (Detailed) refine at 2048x1152 → ESRGAN x4 → downscale to
  5120x2880. Expected ~70–80 s total (unmeasured).
- MCP server (`mcp/`): expose the upscale action too.

### Phase 5 — validation
- Unit tests: planner two-pass sizes/aspect, hires request JSON, postprocess choosing ESRGAN vs Lanczos.
- User-run benchmarks via `bench-2k.ps1` / the app: 2K Fast vs Best timings, 4K t2i, the Spider-Man
  5120x2880 edit, a flat logo (thin lines) to check ESRGAN doesn't smear text.
- Update `README.md` (upscaler model requirement, new settings).

## 4. Relevant code
- `crates/qls-core/src/config.rs` — AppConfig, RecommendedSettings (cfg, cfg_edit, steps, areas)
- `crates/qls-core/src/process.rs` — `build_server_args`, log progress parser (add hires stage parsing:
  "hires fix: upscaling to", "hires sampling 1/1 completed")
- `crates/qls-core/src/planner.rs` — mode detection, working/output sizes (~L510-628), auto VAE tiling (~L633), estimates (~L395)
- `crates/qls-core/src/pipeline.rs` — request building (~L100-125), `postprocess` resize (~L207)
- `crates/qls-core/src/sdcpp.rs` — native API client and request structs
- `crates/qls-core/src/imaging.rs` — Lanczos resize
- `src/components/Settings.tsx`, `Advanced.tsx`, `Stage.tsx` (toolbar), `src/lib/types.ts`

## 5. Implementation status (2026-10-06)

All phases are implemented; real-hardware validation is pending (scripts below).

### Phase 0 — answered from the server source at commit 3f8527a (`examples/server`, `src/pipeline/image.cpp`)
- Requests start from the server's launch defaults and are merged **field by field**
  (`SDGenerationParams::from_json_str`). Consequence: the old per-request
  `vae_tiling_params {tile_size_w: 0}` really did override the 768 px server tiles with 256 px.
- `POST /sdcpp/v1/upscale` body: `{image, upscaler?, repeats? (1-4), tile_size?, output_format?}` →
  `{images[{b64_json}], upscaler, scale, width, height}`; synchronous, holds the GPU lock, loads
  the ESRGAN model per call, output ≤ 8192 px per axis (so ≤ 2048 px input for x4), RGB only.
- Capabilities: `upscale` (bool) and `upscalers[] {name, model, image_upscale}`; `features_by_mode.img_gen.hires/cache`.
- `hires` in `img_gen`: `{enabled, upscaler, target_width, target_height, steps, denoising_strength, ...}`;
  the refine pass reuses `ref_latents`, `ref_image_params` and an interpolated inpaint mask, so
  edits keep their references. `hires.steps` = effective steps (schedule scaled by 1/denoise).
- `cache_mode` / `cache_option` are honored per request (`initialize_cache_params` runs on each
  request) and apply to both passes.
- Model-backed upscaler names resolve as `<dir>/<name>.{gguf,safetensors,pt,pth}`.

### What was built
- VAE tiling fix: nothing is sent when the server already tiles; only an explicit "off" is sent
  (`pipeline::vae_tiling_params`). Auto tiling threshold lowered to 1.6 MP.
- Config: `refineMode` (fast default | best | off), `upscalersDir`, `upscalerModel`, `upscaleTileSize`;
  `--hires-upscalers-dir` passed when the folder exists; setup check (optional) with download URL.
- Planner: two-pass when the working area would exceed 1.3 MP — t2i and edit/multi-ref (refs at
  the base size); refine canvas ≤ 2048×1152; larger outputs delivered at the exact size via ESRGAN.
  `Mode::Upscale` (Quick / Detailed). Estimates calibrated on the bench (Best ≈ 87 s, Fast ≈ 57 s).
- Engine: ESRGAN enlargement whenever the result is enlarged > 1.1× (client-side tiling for
  the 8192 px limit, alpha preserved, Lanczos fallback with a note); runtime check of the server's
  upscaler list; Upscaling/Refining progress stages; OOM retry drops two-pass to one 1 MP pass.
- Gallery `pipeline` metadata (+ A1111 "Hires upscale/steps/upscaler" parameters), Details panel row.
- UI: Upscale button (2×, 4K, source size, custom; Quick/Detailed), refine selector in Advanced
  and Settings. MCP `upscale_image`; API `POST /upscale`.

### Validation (run by the user; close the app first)
```powershell
# Raw sd-server checks + ESRGAN tile size (128/256/512) timing, ~6 min:
powershell -ExecutionPolicy Bypass -File C:\AI\sdcpp\bench\validate-sdserver.ps1
# Through the app engine (2K Fast/Best, 4K, 5120x2880 edit, Quick/Detailed upscale, logo), ~15 min:
cd C:\AI\qwen-local-studio; cargo build -p qls-service
powershell -ExecutionPolicy Bypass -File scripts\dev\validate-upscale.ps1
```

## 6. Upscaling quality review (2026-10-06, after first real use)

### Findings
- **ESRGAN smooths real texture.** Spider-Man "dark version" edit, Balanced (rendered at 1365×768, restored to
  5120×2880): RealESRGAN_x4plus gives crisper edges (+46 % Laplacian energy) but erases the suit's fine
  mesh and invents painterly "crinkle" strokes; the old Lanczos stretch keeps the mesh (soft). It also cost
  +39 s (58.7 s → 98.0 s). Cause: x4plus is trained to remove blur/noise/JPEG from degraded photos, and a
  1–2 px pattern on a clean AI render looks like noise to it. The Best/Fast refine was not involved
  (Balanced is below the 1.3 MP two-pass threshold).
- **sd.cpp's ESRGAN loader** (3f8527a) accepts any RRDBNet with 64 features / 32 growth channels, any block
  count, x1/x2/x4, in both RealESRGAN and old-ESRGAN key formats. Community models trained for clean
  inputs (4x NMKD Siax — WTFPL; 4x Remacri, 4x UltraSharp — non-commercial) should load. Not tested.
- **Guided detail transfer from the source** (prototype, fast guided filter: local affine color model fitted at
  low res, applied to the full-res original): recovers the original's mesh/grain, 5 s on CPU, 40 dB
  consistency with the edit; ghost outlines where the model moved content (chest emblem). **User judged
  plain Lanczos better.**
- **SeedVR2** (ByteDance, ICLR 2026, one-step diffusion SR): best-regarded diffusion upscaler, but users report
  over-generated detail/oversharpening on clean AI-generated inputs (our case), plastic skin, halos, tile
  seams; needs a PyTorch sidecar and can't share 16 GB with Qwen. SUPIR: SDXL-based, slow, strays further.

### Decisions
| Option | Decision |
| --- | --- |
| ESRGAN for automatic enlargement (edits restored to source size, outpaint, > 2K) | **Drop** — back to Lanczos by default |
| Guided transfer from source | **Drop** (visually worse to the user) |
| SeedVR2 / SUPIR | **Not now** (risk of the same over-detail, new runtime, VRAM) |
| Two-pass hires with ESRGAN *between* passes (> 1.3 MP) | **Keep** (bench: best 2K look; Qwen redraws detail after it) |
| **Lanczos upscale + Qwen low-denoise refine** (whole image at 2K, or ~1 MP tiles at full res) | **Pursue** — measure first |
| Better ESRGAN model (Siax/Remacri) for the Quick upscale | Optional, after `compare-upscalers.ps1` |

### To do
1. Add an "enlarge results with" setting (Lanczos | ESRGAN), default **Lanczos**; ESRGAN stays available in
   the Upscale button's Quick mode.
2. Run `python C:\AI\sdcpp\bench\tiled_refine_test.py` (~10–15 min; user runs it). It compares
   `refine2k`, `tiles_cfg4`, `tiles_cfg1`, `tiles_cfg1_ec`, `tiles_s020` (24 tiles of 960×832 for 5120×2880,
   denoise 0.3 = 6 of 20 steps, neutral prompt) and logs time per stage. Speed levers to evaluate:
   CFG 1 (one model pass per step, ~2×), EasyCache, lower denoise, tile size, keeping 3 jobs queued
   (the server caches the prompt encoding across tiles).
3. Pick the quality/time trade-off; implement it as the Upscale **Detailed** mode (replacing the current
   2K img2img + ESRGAN) and, if fast enough, as an option for large edits.
4. Optional: `compare-upscalers.ps1` with Siax/Remacri; consider also Lanczos as the hires upscaler between passes.
5. Still pending from section 5: `validate-sdserver.ps1` and `scripts\dev\validate-upscale.ps1`.

### Update 2026-10-06 (evening): GAN model choice
- A/B of 9 ESRGAN-family models on the Spider-Man edit (`C:\AI\sdcpp\bench\gan_compare.py`, sheets in 
  `bench\out\gan\`): **4x-UltraSharp** and **4x_NMKD-Superscale-SP** keep the fine texture; x4plus/esrnet
  smooth it, Valar crushes shadows. BSRGAN and RealisticRescaler don't load in sd.cpp.
- Texture transfer from the original (A, `gan_methods.py`) and Qwen tiled refine over the GAN (B,
  `gan-qwen-refine.ps1`) were **rejected** by the user.
- **Implemented:** Settings → Models offers UltraSharp (default) and Superscale-SP for every ESRGAN use
  (hires pass, enlargement, Upscale button); configs on the old x4plus default migrate (schema 2).


## 7. Settings corrected to the official Qwen-Image 2.1 release (2026-10-07)

Every measurement and A/B test above ran at **CFG 6 (edits 4) with 20 steps**, taken from the
example command in sd.cpp's `docs/qwen_image_2.1.md`. The official release does not use
guidance: diffusers `QwenImage21Pipeline` defaults to `true_cfg_scale=1.0` ("Qwen-Image 2.1 is
meant to be sampled without guidance") with 40 steps, and the README's sizes are native 2K
(2048², 2752×1536…). sd-server runs a second, unconditional model pass whenever CFG ≠ 1, so the
old settings doubled the cost of each step and over-guided the model. That may explain the
over-sharpened 2K look, the edits that collapsed to near-black, and the canvas texture. sd-server's
default Qwen 2.1 schedule already matches the official one (exponential shift, mu 0.5→0.9 over
256→8192 tokens, terminal 0.02), so no schedule change is needed.

New defaults (config schema 4 migrates configs still on the old values):

| Preset | Before | Now |
| --- | --- | --- |
| Fast | ~1 MP, 20 steps, CFG 6 | ~1 MP, 20 steps, CFG 1 (shortcut, ~2× faster than before) |
| Balanced | ~1 MP, 20 steps, CFG 6 | ~1 MP, 40 steps, CFG 1 (official) |
| Quality | ~2 MP, 20 steps, CFG 6 | native 2K bucket, 40 steps, CFG 1 (official) |
| Edits / inpaint / outpaint / Detailed upscale | CFG 4 | CFG 1 |

Time estimates halve the CFG-6 step times for CFG 1. They need re-measuring. The quality
conclusions above (ESRGAN choice, single pass vs refine, edit CFG) were drawn at CFG 6 and
should be re-checked. Bench scripts now default to the official settings: `turbo_test.py`
writes to `out/turbo_official`, `nude_turbo_bench.py` to `out/nude_official`.
`validate-sdserver.ps1`, `bench-2k.ps1` and `tiled_refine_test.py` still test the old CFG-6
two-pass flow and are kept only as a historical record.
