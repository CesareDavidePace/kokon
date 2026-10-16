# Linux: plan, status and engine upkeep

Kokon supports **Linux from 0.9.3** (x86_64, `.deb` + `.rpm`, NVIDIA CUDA and Vulkan for AMD/Intel),
next to Windows and macOS. Every release from now on ships all three.

## The plan (decided 2026-10-16)

The app is already client/server: the UI talks to `kokon-service` only over HTTP (base URL + token,
`src/lib/api.ts`); the desktop shell just starts the service and hands the UI `127.0.0.1` + token
(`src-tauri/src/lib.rs` `connect`). So the three steps are one product started three ways:

1. **Native Linux app** (this document, 0.9.3). Same app as Windows/Mac: local service on 127.0.0.1.
2. **Headless server + browser UI**, then **Docker**. `kokon-service` serves the built UI itself;
   network access is opt-in, off by default, with a real password (rate-limited) before anything
   listens beyond localhost; recommend Tailscale/VPN over exposing it. Docker = the same Linux build
   without a window, model folders mounted as volumes.
3. **Kokon as a client**: the desktop app (Windows/Mac/Linux) connects to a Kokon server elsewhere.
   "Connect to a server" in Setup/Settings (address + password); `connect` returns the remote URL.
   Needs: copy-image by bytes instead of a local path (`Stage.tsx` → `copy_image`), hide "Show in
   folder"/"Open logs", import/export by upload/download instead of paths (`api.ts` `importPath`,
   `exportGeneration`), setup's "Use models I already have" picks a folder **on the server**, and a
   client/server version check.

Rough effort for one person: 1 = days, 2 = 3–5 days, 3 = about a week. Order: 1 → 2 (+Docker) → 3.
Public promise: issue #15 comment (2026-10-16). #22 closed as its duplicate.

## Why Kokon builds its own Linux engine

Upstream stable-diffusion.cpp publishes Linux zips for Vulkan, ROCm and CPU only, **no CUDA**, built on
Ubuntu 24.04: they need **glibc 2.38**, so they don't run on Ubuntu 22.04 or Debian 12. Kokon builds the
exact upstream commit it ships on **Ubuntu 22.04** (glibc 2.35; the Vulkan build needs ≤ 2.34) with
upstream's own flags and CUDA architectures (`61;70;75;80;86;89;90;100;120`, CUDA 12.8).

- Fork: `GabrieleLozupone/stable-diffusion.cpp`, branch **`kokon-linux`** = the upstream commit + one
  commit adding `.github/workflows/kokon-linux.yml`.
- A push to `kokon-linux` builds Vulkan + CUDA, smoke-tests that the engine loads on Ubuntu 22.04/24.04,
  Debian 12, Fedora 40/42, and publishes prerelease **`kokon-linux-<commit>`** with
  `sd-master-<commit>-bin-Linux-Ubuntu-22.04-x86_64-{vulkan,cuda12}.zip`,
  `cudart-sd-bin-linux-cu12-x64.zip` (libcudart/libcublas/libcublasLt, next to the engine; RPATH
  `$ORIGIN`) and `SHA256SUMS.txt`.
- The app pins these in `crates/qls-core/src/catalog.rs` (`#[cfg(target_os = "linux")]` versions of
  `ENGINE_CUDA`, `CUDA_RUNTIME`, `ENGINE_VULKAN`, plus `LINUX_*_SIZE/SHA256`).
- `engine.json` (engine updates without an app release) lists upstream's Windows/Mac builds only, so
  **Linux ignores it** (`apply_engine_release` returns early). Linux engines change with app releases
  until the manifest gets Linux entries (to do, below).
- First build: run 38885751650 (2026-10-16): Vulkan 12 min, CUDA 1 h 32 min. Its CUDA smoke test was
  wrong (sd-server links libcuda.so.1, which only the NVIDIA driver provides, so it can't start in CI);
  the release was published by hand from that run's artifacts after a real check in WSL (RTX 5070 Ti
  found, CUDA backend loads). The workflow now only checks libraries for CUDA and never replaces a
  published release (rebuilds aren't byte-identical and the app pins SHA-256).

## Engine upkeep: check at every release (and when engine.json moves)

Upstream moves fast (2026-10-16: master-968-8cd731e, 4 commits after our 1d0a58a, including a Vulkan
direct-convolution fix and FLUX.3 Klein). When Windows/Mac move to a newer engine, Linux must follow:

1. `gh api repos/leejet/stable-diffusion.cpp/compare/<our commit>...master` and the upstream releases.
2. In the fork clone: `git fetch upstream`, `git switch kokon-linux`, `git rebase --onto <new commit>
   <old commit>` (keeps only the workflow commit), set `ENGINE_SHORT` in the workflow, push to `fork`.
3. Wait for the run; it publishes `kokon-linux-<new commit>`. Check the glibc report in the build log.
4. In Kokon: update the Linux URLs, sizes and SHA-256 in `catalog.rs` with the `ENGINE_TAG` bump,
   test once in WSL (engine download + one image), release.
5. Same pipeline later for #24: a Mac engine built for macOS 15 (planned 0.9.4).

## Building and testing the Linux app

- Build machine: **WSL Ubuntu-22.04** on the dev PC (installed 2026-10-16; sees the RTX 5070 Ti through
  the Windows driver, so CUDA can be tested; windows open through WSLg). Clone on the Linux disk:
  `git clone -b <branch> /mnt/c/AI/qwen-local-studio ~/kokon` (never build from /mnt/c: npm would replace
  the Windows node_modules). Ubuntu's `cargo` package is too old; use rustup (`. ~/.cargo/env`).
- Release: `bash scripts/build-release-linux.sh --setup` once, then `bash scripts/build-release-linux.sh`
  → `dist-release/Kokon_<v>_amd64.deb`, `Kokon-<v>-1.x86_64.rpm`, `updater-linux-x86_64-{deb,rpm}.json`.
  Signed with the same updater key as Windows/Mac (`~/.tauri/kokon-updater.key`, also found under
  /mnt/c/Users/*/.tauri).
- Quick run without packaging: `npm run build && cargo build -p kokon --release --features
  tauri/custom-protocol && ./target/release/kokon` (without the feature the window loads the dev server
  and shows "Could not connect to localhost").
- In-app updates work on Linux: tauri-plugin-updater 2.13 installs `.deb`/`.rpm` through pkexec;
  `latest.json` needs `linux-x86_64-deb` and `linux-x86_64-rpm` entries.
- Package dependencies (tauri.linux.conf.json): libssl3 / openssl-libs (reqwest native-tls), libgomp,
  libvulkan1 / vulkan-loader. HEIC photos can't be opened on Linux yet (message shown).

## Status (2026-10-16)

Done
- ✅ Fork workflow + first engine build; prerelease `kokon-linux-1d0a58a` published (Vulkan loads on
  Ubuntu 22.04/24.04, Debian 12, Fedora 40/42; CUDA finds the RTX 5070 Ti in WSL).
- ✅ Kokon branch `linux` (commit 9d8ed7f, local): Linux engines in the catalog, manifest ignored on
  Linux, AMD GPU detection from /sys/class/drm + pci.ids names, deb/rpm config, bundled resources in
  /usr/lib/Kokon, logs-folder path, kokon-mcp Linux build, updater manifest Linux entries,
  build-release-linux.sh. Windows tests still pass (core 185, UI 128).
- ✅ First Linux compile in WSL OK; app opens, detects RTX 5070 Ti 16 GB, picks CUDA + 16 GB preset,
  install folder ~/.local/share/KokonData/sdcpp.

To do for "0.9.3 for Linux"
- ✅ Engine sizes and SHA-256 pinned in catalog.rs (from GitHub's asset digests).
- [ ] Full test in WSL: Install → CUDA engine download → one image + one edit on the 5070 Ti.
- [ ] `build-release-linux.sh` end to end (tests, kokon-mcp, deb + rpm, signatures). Install the .deb in
      a clean WSL distro; ideally also Fedora (`wsl --install -d FedoraLinux-42`) and Debian.
- [ ] Merge `linux` into master (no app change for Windows/Mac), add the .deb/.rpm to the v0.9.3 release,
      merge the Linux entries into v0.9.3's `latest.json` (download it, add the two platforms, re-upload;
      Windows/Mac entries unchanged).
- [ ] README (platform table, badges, install commands `sudo apt install ./Kokon_…deb` /
      `sudo dnf install ./Kokon-…rpm`), site download button + FAQ, release notes section, bug-report
      template OS list, docs/platforms.md.
- [ ] Announce: #15 comment (ping giveen, swingtempo), Reddit "Update" lines, X; later r/selfhosted with
      the server mode.

Later
- [ ] engine.json Linux entries (fork URLs) so Linux gets engine updates without an app release.
- [ ] Real-hardware reports: AMD on Linux (swingtempo R9700, alean200), a non-WSL NVIDIA machine.
- [ ] AppImage? Not planned (large, and deb/rpm cover the main distros).
- [ ] Steps 2 and 3 of the plan above.
