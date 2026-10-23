# Status

**Updated 2026-10-23.** What is in progress, what comes next, and where help is welcome. Kept current at the end
of every task (see [AGENTS.md](../AGENTS.md#keeping-docsstatusmd-current)); GitHub is the source of truth.

## Current version: [0.10.2](https://github.com/GabrieleLozupone/kokon/releases/tag/v0.10.2) (2026-10-23)

Windows, macOS (Apple Silicon, 15+) and Linux (`.deb`, `.rpm`). Installed copies update from the app.

## In progress: 0.11, Kokon as a server

Tracked in [#15](https://github.com/GabrieleLozupone/kokon/issues/15); design in [linux.md](linux.md).
- The browser UI, served by `kokon-service` itself.
- Access from the local network: off by default, behind a password with a limit on attempts. For access from
  outside, a guide to Tailscale or another VPN, never direct exposure.
- A Docker image for NVIDIA (CUDA) first; an AMD/Intel (Vulkan) image once it can be tested.
- Later: the desktop app as a client of a Kokon server elsewhere.

## Next

- **0.12: reliability foundations** (details soon).
- On the list, no dates yet: a custom upscaler picker
  ([#34](https://github.com/GabrieleLozupone/kokon/issues/34)), everything in one folder / a portable build
  ([#26](https://github.com/GabrieleLozupone/kokon/issues/26)), translations (only with contributor help).

## Known issues

- **AMD RX 6000 (RDNA2) cards are slow on Vulkan** (an RX 6700 XT takes 21-41 s per step at 1 MP). Under
  investigation: flash attention and memory placement are ruled out (the model stays in VRAM, the GPU is at 99 %).
  RX 7900 XTX and RX 9060 XT work well.
- **LoRA is beta** and tested on Windows only; reports from macOS and Linux are welcome
  ([#33](https://github.com/GabrieleLozupone/kokon/issues/33) has a product-shot test).
- **Macs are slow** compared with NVIDIA cards (about 4 minutes for 1 MP on an M2 Pro 16 GB). Engine speed-ups are
  being measured.
- The Windows installer is not code-signed: SmartScreen or Smart App Control may block it (README FAQ: Unblock).
- HEIC photos can't be opened on Linux yet.

## Under review

- [#29](https://github.com/GabrieleLozupone/kokon/pull/29) EasyCache step cache (giveen): to be A/B-tested on the
  test set (text-to-image, edit, multi-image, LoRA) before it becomes a default.

## Help wanted

- Good first issues: [#2](https://github.com/GabrieleLozupone/kokon/issues/2) gallery keyboard,
  [#3](https://github.com/GabrieleLozupone/kokon/issues/3) site slider keyboard,
  [#4](https://github.com/GabrieleLozupone/kokon/issues/4) reduced-motion video.
- AMD RDNA3/RDNA4 owners who know HIP: compare a ROCm engine with Vulkan
  ([#38](https://github.com/GabrieleLozupone/kokon/issues/38)).
- Hardware reports: AMD and Intel on Windows and Linux, Linux on NVIDIA outside WSL, Macs other than an M2 Pro.
- Model proposals that meet the switch rule: [Discussion #35](https://github.com/GabrieleLozupone/kokon/discussions/35).

## Recently shipped

- **0.10.2** (2026-10-23): new image from reference images (also in MCP and the HTTP API), a Prompt+ menu to
  read and change the prompt first, longer Prompt+ prompts, edits never reuse their source's seed, Vulkan decodes
  in 512 px tiles (AMD: seconds instead of 25-70 s), engine
  [kokon-1013-1814814](https://github.com/GabrieleLozupone/stable-diffusion.cpp/releases/tag/kokon-1013-1814814),
  `kokon-mcp --help` ([#36](https://github.com/GabrieleLozupone/kokon/pull/36)) and the full HTTP API reference
  ([#37](https://github.com/GabrieleLozupone/kokon/pull/37)) by shrinivas-sn.
- **0.10.1** (2026-10-22): own model files never take a text-only Qwen3 as the text encoder; Linux NVIDIA blank
  window fixed; the progress card fits small windows ([#32](https://github.com/GabrieleLozupone/kokon/issues/32));
  open the images folder from Settings and the gallery ([#34](https://github.com/GabrieleLozupone/kokon/issues/34));
  `kokon-service --version` ([#31](https://github.com/GabrieleLozupone/kokon/pull/31)); CI on pull requests
  ([#25](https://github.com/GabrieleLozupone/kokon/pull/25)).
- **0.10.0** (2026-10-20): Prompt+ (the text encoder writes the prompt, no second model), your own LoRA files
  (beta), Kokon's own engine on every platform, a new look.
- **0.9.4** (2026-10-16): Linux, macOS 15, colour LUTs. **0.9.3** (2026-10-15): your own model files, GPU picker,
  texture-fix VAE, 8 GB beta.

Full notes: [releases](https://github.com/GabrieleLozupone/kokon/releases) and [marketing/](../marketing/).
