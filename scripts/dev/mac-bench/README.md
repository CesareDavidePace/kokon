# Mac engine bench

Qwen-Image 2.1 Turbo on Apple Silicon: Kokon's engine against other engines and settings, to find
what makes a Mac render faster. The goal is not to switch engines but to see which optimisations
pay off, so they can be brought into Kokon's build of stable-diffusion.cpp.

```bash
bash scripts/dev/mac-bench/setup.sh            # once: upstream sd.cpp, mflux 0.23, MLX 4-bit (~11 GB)
python3 -I scripts/dev/mac-bench/bench.py --list
python3 -I scripts/dev/mac-bench/bench.py      # quit Kokon first; ~1 h on an M2 Pro 16 GB
```

Downloads, images and reports go to `~/kokon-bench` (`KOKON_BENCH_DIR`), not into
the repo. Kokon's own engine and GGUF files are read from its install (`config.json`).

## What runs

- **`kokon`**: the reference, the flags Kokon gives sd-server on a 16 GB Mac
  (`process.rs`: Disk placement, mmap, flash attention, VAE tiles of 768, prefix cache q8_0).
- **One change each** against it: weights kept resident, flash attention off, prefix cache off or
  f16, VAE in one piece, sd.cpp's step caches (EasyCache, DBCache, Spectrum), Q8_0 weights
  (`setup.sh --with-q8`), upstream sd.cpp master.
- **mflux 0.23** (MLX) with the 4-bit Turbo bundle: default bf16, `--compute-precision float16`,
  `--low-ram`, step reuse.
- **`siliconed`** (not in the default set): install [Siliconed](https://github.com/gwenn-ha-dev/Siliconed)
  in `/Applications`, install Qwen-Image-2.1 Turbo from its window (the licence is accepted there; `silicontrol
  install qwen-image-2.1 --light` opens the sheet), and choose Faster Rendering in its Settings. bench.py
  queues the render with `silicontrol add --wait` and reports the app's own time; its first render after a
  launch also measures the Mac (~30–50 s, not in that time).
- Cases: text-to-image 1024², and one edit at 1376×768 with the cabin example as reference.

Every run is a new process, so loading counts, as on a first generation. PSNR is against the
first variant of the same engine (same noise): it shows what an approximate setting costs.

## What the search found (2026-10-25)

| Source | What it does | Kokon's engine |
| --- | --- | --- |
| Qwen-Image 2.1 prefix K/V cache (Qwen blog, vLLM-Omni, mflux 0.21) | text tokens are conditioned at t = 0, so their K/V are computed once and reused: exact | has it (`qwen_image_2_1_prefix_cache_type=q8_0`) |
| mflux 0.21–0.23 | fused Q/K RMSNorm + RoPE Metal kernel, packed QKV/FFN projections, Turbo's 8-sigma grid unshifted | to measure |
| mflux 0.22 `--compute-precision float16` | attention and FFN in f16: 8–35 % per step on an M1 Max (no fast bf16 before M3) | ggml computes in f16/f32 already |
| mflux step reuse, mlx-fast first-block cache | skip steps or blocks that change little: approximate, mflux disables it under 10 steps for edits | sd.cpp has EasyCache, DBCache, Spectrum |
| tillknuesting/qwen-image-2.1-mlx-fast (M4 16 GB) | the above on MLX 4-bit; Turbo LoRA 6 steps 1024² in 139–145 s | — |
| Draw Things (Metal FlashAttention 2.x) | own attention/GEMM kernels; closed app, gains mostly on M5 neural accelerators | — |
| Siliconed 0.12 (Swift + Metal, Apache-2.0), M1 Pro 16 GB 1024² 8 steps: 143 s, 104 s with half-size first steps | matmuls split across GPU + CPU AMX + Neural Engine by a per-chip measured map; attention split by heads GPU/CPU; weights streamed from disk (peak footprint ~2 GB); first 3 of 8 steps at half size (approximate); official 8-sigma grid unshifted | none of these; sd.cpp samples with its Flux scheduler |
| Rapid-MLX | server around mflux (4-bit, 8.9 GiB) | same engine as mflux |
| LightX2V | PyTorch MPS; INT8 Metal GEMM only on M5/M6; M5 Max ~3.5 min for an edit | — |
| Core ML, diffusers MPS, ComfyUI | no Core ML conversion exists; diffusers needs bf16 (32 GB); ComfyUI GGUF on MPS is slower than sd.cpp (M4 16 GB: 736 s for 25 steps) | — |
| mflux small text encoder (Qwen3.5-0.8B + zen-image-edit adapter) | 2 GB instead of 5 GB for the text encoder: frees memory on 16 GB Macs, changes the output | idea, not measured here |

M2 Pro arithmetic: 7.1 B parameters × 1024 tokens ≈ 14.6 TFLOP per step at 512², and the GPU
peaks near 6.8 TFLOPS, so 6.3 s per step is ~35 % of peak. At 1 MP (4096 tokens) the floor is
~9 s per step; Kokon takes 27.3 s (measured below).

## First results (2026-10-25, M2 Pro 16 GB, macOS 27.0.1)

Text-to-image 1024², 8 steps, seed 42, one run each, every process fresh:

| Engine | Time | Per step | Peak memory | Notes |
| --- | --- | --- | --- | --- |
| `kokon` (Disk placement, today) | 274.9 s | 27.3 s | 5.1 GB | text in the image right ("Earl Grey") |
| `kokon-resident` | 252.7 s | 27.3 s | (mmap) | only the ~22 s load is saved: compute bound |
| `upstream` sd.cpp master-1004 | = kokon | = kokon | | 512² only: the same image to the bit |
| mflux 0.23, 4-bit | 157.0 s | 18.0 s | 15.0 GB | 4.3 GB swapped out (MLX peak 18.45 GB); label text garbled |
| Siliconed 0.14 Light, every step at full size | 133.7 s | | 1.4 GB | 0 swap; label right; the like-for-like number |
| Siliconed 0.14 Light, Faster Rendering (default) | 100.8 s | | 1.4 GB | 3 of 8 steps at half size (approximate); label garbled |

After a reboot (swap 0, only VS Code open), same case, system memory sampled every second
("rise" = Memory Used at peak minus just before the run: app memory + wired + compressor;
the file cache is apart since macOS drops it when needed):

| Engine | Time | Memory rise | Wired peak | Compressor peak | Swap | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `kokon` | 276.2 s (27.4 s/step) | +3.5 GB | 6.9 GB | 3.4 GB | 0 | free memory down to 0.02 GB |
| Siliconed Light, full size, Neural Engine on | **94.5 s** | **+2.3 GB** | 3.2 GB | 2.5 GB | 0 | the map turned the ANE on after the relaunch; image as the full-size one |
| mflux 4-bit | 157.7 s (18.0 s/step) | +7.7 GB | 10.9 GB | 5.9 GB | +5.0 GB | |

Siliconed's times are the app's own (queue to PNG); its first launch adds ~47 s of Mac calibration.
Same 4-bit DiT class as Kokon's, so the 2.1× at full size is the engine, not the weights.
