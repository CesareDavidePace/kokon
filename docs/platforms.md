# Platforms: engines, requirements and what is verified

Kokon runs Qwen-Image 2.1 through stable-diffusion.cpp. Each platform uses a prebuilt engine
from the same pinned release ([master-929-3f8527a](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-929-3f8527a)),
so model behaviour is identical everywhere; only speed and memory differ.

| Platform | Engine | Status |
| --- | --- | --- |
| Windows + NVIDIA | CUDA 12 build | ✅ verified (RTX 5070 Ti 16 GB, RTX 5070 12 GB) |
| Windows + AMD / Intel | Vulkan build (30 MB, uses the driver's Vulkan runtime) | 🟡 beta: works on NVIDIA (RTX 5070 Ti, 2K/20 steps, slower than CUDA); AMD / Intel to verify |
| macOS + Apple Silicon | Metal build (universal binary, shaders embedded) | 🟡 built, first run on an M2 Pro 16 GB pending (`scripts/dev/mac-dev.sh`) |
| Windows + AMD via ROCm | ROCm build | ✗ not planned: open crashes (#2008), DLL conflicts (#2012), black images with k-quants (#1385) |
| Linux | Vulkan / CUDA builds exist | later |

## How the app sizes itself (all platforms)

One number drives everything: the memory the GPU can use for the diffusion model.

- **Discrete GPU** (NVIDIA, AMD, Intel): its VRAM. NVIDIA through `nvidia-smi`; any other card
  through DXGI (dedicated video memory). The text encoder lives in system RAM.
- **Apple Silicon**: unified memory is shared, so the planner uses
  `planner::unified_memory_planning_mib`: Metal's GPU share of RAM (≈ 2/3, ≈ 3/4 from 64 GB)
  minus the text encoder, which sits in the same pool.

From that number the existing logic picks the preset (which weights to download) and the token
budget (largest working size per quality level). Nothing is tuned per platform.

## Windows: AMD and Intel (Vulkan, beta)

- Engine: `sd-master-3f8527a-bin-win-vulkan-x64.zip`, installed into `<sdcpp>\engine-vulkan`,
  next to the CUDA build (switching engines needs no re-download).
- Launch: `--backend vulkan0`, flash attention, q8_0 prefix cache. No SageAttention (CUDA only).
- Requirements: a discrete GPU with **12 GB+ VRAM** (RX 6700 XT 12 GB / 7700 XT and up,
  Arc A770 16 GB / B580 12 GB), current driver, 16 GB RAM.
- Known upstream issues:
  - integrated GPUs report 0 MB and fail (#2022), so they are not supported;
  - an edit with 2 reference images can crash (#1991, fix in ggml not yet synced);
  - no published Qwen 2.1 speeds on AMD/Intel yet.

## macOS: Apple Silicon (in testing)

Evidence: [qwen-image-studio](https://github.com/sanketshinde3001/qwen-image-studio) runs Qwen 2.1
t2i and edits on a 16 GB M5 with **our exact engine build**. Its numbers: 10–12 GB peak
(`--mmap --params-backend disk`), about 75 s/step at 1024² with CFG 6, so about 37 s per model
pass. Kokon uses CFG 1, which is one pass per step.

### Draft system requirements

| | Minimum | Recommended |
| --- | --- | --- |
| **Chip** | Apple M1 or newer (Intel Macs: no) | M-series Pro / Max / Ultra |
| **macOS** | 26 Tahoe (the engine binary's minimum) | latest |
| **Unified memory** | 24 GB (Fast/Balanced, up to ~1.5K) | 32 GB+ (Quality 2K); 48 GB+ gets the 8-bit model |
| **Disk** | 15 GB free | 25 GB on the internal SSD |

16 GB Macs can run it only in the lightest setup: weights streamed from disk, 1 MP, much
slower. It should be offered as "not supported, try at your own risk" rather than blocked.

### Memory math

Weights: diffusion Q4_K_M 4.2 GB + text encoder Q4_K_M 5.0 GB + vision projector 1.2 GB +
VAE 0.25 GB ≈ 10.7 GB.

| Unified memory | GPU share | − text encoder | Plans like a | Preset |
| --- | --- | --- | --- | --- |
| 16 GB | ~10.7 GB | ~4.5 GB | 4–6 GB card | lightest, 1 MP |
| 24 GB | ~16 GB | ~9.8 GB | 10 GB card | 12 GB preset |
| 32 GB | ~21 GB | ~15 GB | 16 GB card | 16 GB preset (2K) |
| 48 GB | ~32 GB | ~26 GB | 24 GB+ card | 24 GB preset (Q8 model) |
| 64 GB+ | ~48 GB | ~42 GB | 32 GB+ card | 32 GB+ preset |

### Expected speed

These are **estimates, not measured**. They scale the M5 result by GPU size and are for
1 MP, 20 steps, CFG 1.

| Chip class | GPU cores | Bandwidth | Per image (rough) |
| --- | --- | --- | --- |
| M1–M5 (base) | 7–10 | 68–153 GB/s | 10–15 min |
| Pro | 14–20 | 150–273 GB/s | 4–6 min |
| Max | 30–40 | 300–546 GB/s | 2–3 min |
| Ultra | 48–80 | ~800 GB/s | 1–2 min |

The setup screen should show the chip, its memory and this expected time before downloading,
so nobody with a base chip is surprised.

### Work needed (✅ = done 2026-10-09, runs from source with `scripts/dev/mac-dev.sh`)

1. ✅ Detection: `sysctl hw.memsize`, `machdep.cpu.brand_string`, GPU cores from
   `system_profiler SPDisplaysDataType -json`; feed `unified_memory_planning_mib` into the
   planner and presets.
2. ✅ Engine: the macOS zip (`sd-server`, `libstable-diffusion.dylib`) in `<sdcpp>/engine-metal`;
   launch with `--backend metal` (sd-server maps "metal" to ggml's MTL device), `--fa`, `--mmap`,
   no SageAttention; RAM placement becomes "everything on the GPU" (same memory), and Macs
   that plan below 8 GiB (16 GB) get `--params-backend disk`.
3. ✅ Portability fixes (all Rust type-checks for aarch64-apple-darwin):
   - catalog paths use `\` (switch to `/`);
   - HEIC decoding uses Windows WIC (macOS: `sips`);
   - Tauri sidecar names and the `.exe` suffix;
   - data folders (`dirs` already maps them to `~/Library/...`).
4. Packaging (todo; `tauri.macos.conf.json` prepared): a GitHub Actions `macos-14` (arm64) runner builds the `.dmg`. Without an Apple
   Developer ID ($99/year) the app is unsigned, so users must right-click → Open the first
   time. The engine itself, downloaded by the app over HTTP, carries no quarantine flag and
   runs, because it is ad-hoc signed.
5. Known Metal issue: transparent (RGBA) output decodes as white (#2024, fix not merged).
   Kokon outputs RGB, so only transparency-preserving edits are affected.
6. Testing: needs a real Apple Silicon Mac with 24 GB+ (GitHub's runners have 7 GB: build
   only). Without one, ship as "beta, untested".
