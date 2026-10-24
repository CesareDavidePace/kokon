# Status

**Updated 2026-10-24.** What is in progress, what comes next, and where help is welcome. Kept current at the end
of every task (see [AGENTS.md](../AGENTS.md#keeping-docsstatusmd-current)); GitHub is the source of truth.

## Current version: [0.10.2](https://github.com/GabrieleLozupone/kokon/releases/tag/v0.10.2) (2026-10-23)

Windows, macOS (Apple Silicon, 15+) and Linux (`.deb`, `.rpm`). Installed copies update from the app.

## In progress: 0.11, Kokon as a server

Tracked in [#15](https://github.com/GabrieleLozupone/kokon/issues/15); design in [linux.md](linux.md). Built on the
branch `server-0.11`, being tested before the release:
- The browser UI, served by `kokon-service` itself (Settings → Other devices, or `kokon-service --remote`).
- Access from the local network: off by default, behind a password (Argon2id hash) with a limit on attempts.
  For access from outside, a guide to Tailscale or another VPN: [remote-access.md](remote-access.md).
- A Docker image for NVIDIA (CUDA): [docker.md](docker.md). An AMD/Intel (Vulkan) image once it can be tested.
- Projects shared by every device: the list is kept by `kokon-service` (`projects.json`), not in each app;
  versions made on a phone show up in the same project on the computer. Each app's list from before
  0.11 is merged in once, at the first start.
- A phone layout of its own for the browser UI: a library of projects as the home screen, a project's versions
  swiped sideways with the timeline under them, the image's actions in one row, a screen of its own for writing the
  prompt above the keyboard, menus as sheets from the bottom, the status bar in the theme's colour. Images
  develop into view (a blurred thumbnail sharpening into the picture) instead of showing a checkerboard while they
  load, also in the desktop app.
- Browsers stay signed in when `kokon-service` restarts (an app update, a reboot): `remote-sessions.json` in the
  data folder keeps only a SHA-256 of each session. A new password ends them all, also one set while Kokon was off;
  a signed-out page goes to the sign-in at once.
- A browser page reloads itself once when the computer running Kokon has updated to another version.
- Fixed on iPhone: the What's new buttons fit a phone screen (stacked).
- Fixed on iPhone: full screen could crash the page (Safari ran out of memory on large results and edits): in a
  browser it now draws the image at screen size, shows the preview until you zoom in past it, and loads the
  original only when a comparison shows it. Take Photo uploads the picture (only the photo library worked).
- On a phone, a photo just taken can be flipped before it is uploaded (selfies from Safari's camera are not
  mirrored the way the Camera app saves them), and menus open lower, near the bottom edge.
- Touch screens: swipe between images in full screen and in All images.
- iPhone: the status bar follows the theme (also when the system switches) and turns black in full screen.
- Phones: the paint editor uses much less memory on large images (drawn at screen size, as full screen).
- Own model files: a warning (in setup and at startup) when the image model is a known base Qwen-Image 2.1
  file instead of Turbo (Kokon runs 8 steps, made for Turbo), and a hint in setup that FP8 files run much
  slower than the GGUF on AMD/Intel with Vulkan ([#38](https://github.com/GabrieleLozupone/kokon/issues/38)).
- MCP: an output folder listed in `QLS_MCP_ALLOWED_DIRS` now works (the service refused it), and the error for a
  folder outside the allowed ones names the variable ([#39](https://github.com/GabrieleLozupone/kokon/issues/39)).
- Opening a project with many versions is fast on a phone: the browser loads all its records in one request
  (`/projects/{id}/records`), the stage shows a screen-sized preview (`/gallery/{id}/preview`, at most 1280 px,
  JPEG) when it is as sharp there, and the original of an edit loads only when a comparison view shows it. Zoom,
  full screen, download, copy, the comparison views and the editors keep the full image; the desktop app is
  unchanged. Measured on a 30-version project of 2048 px PNGs (Playwright, phone size, cache off): from 63
  requests and 15.1 MB to 32 requests and 0.84 MB; first image from 6.9 s to 0.37 s (20 Mbit/s) and from 15.0 s
  to 0.42 s (9 Mbit/s, 150 ms). Stepping through 10 versions: 137 MB to 2.4 MB.
- Later: the desktop app as a client of a Kokon server elsewhere.

## Next

- **0.12: reliability foundations** (details soon).
- On the list, no dates yet: a custom upscaler picker
  ([#34](https://github.com/GabrieleLozupone/kokon/issues/34)), everything in one folder / a portable build
  ([#26](https://github.com/GabrieleLozupone/kokon/issues/26)), translations (only with contributor help).

## Known issues

- **AMD RX 6000 (RDNA2) cards are slow on Vulkan** (an RX 6700 XT takes about 41 s per step at 1 MP).
  Most of each step (over 80 % at 1 MP) goes into the attention kernel, which is slow on GPUs without matrix cores.
  RX 7000 and RX 9000 cards have them and work well (RX 9070 XT: 1.75 s per step at 1 MP).
- **LoRA is beta** and tested on Windows only; reports from macOS and Linux are welcome
  ([#33](https://github.com/GabrieleLozupone/kokon/issues/33) has a product-shot test).
- **Macs are slow** compared with NVIDIA cards (about 4 minutes for 1 MP on an M2 Pro 16 GB). Engine speed-ups are
  being measured.
- The Windows installer is not code-signed: SmartScreen or Smart App Control may block it (README FAQ: Unblock).
- HEIC photos can't be opened on Linux yet.

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
