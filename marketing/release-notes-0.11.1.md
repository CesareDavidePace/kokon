## Kokon 0.11.1

Kokon's own engine for Apple Silicon, a new copy of the image model, and fixes you reported.

- **Kokon's Mac engine (experimental)**: Kokon now has an engine of its own for Apple Silicon, written for Qwen-Image 2.1 Turbo. The GPU, the CPU's matrix unit and the Neural Engine work together: a new 1024 × 1024 image takes about a minute on an M2 Pro (16 GB) instead of over four, and Prompt+ is about twice as fast. New images, edits with reference images, brushed edits, extending and Prompt+ all run on it. It comes inside Kokon.app, is offered once at start on Macs with 16 GB or more, and stays off until you try it. Switch back any time in Settings → Performance → Kokon's Mac engine; LoRAs and the upscaler model still use the standard engine.
- **Benchmark this Mac**: Settings → Performance measures the Mac engine on your Mac in about 10 minutes and gives a report you can copy. Reports from M1, M3, M4 and M5 Macs help most: please share yours in [Discussion #43](https://github.com/GabrieleLozupone/kokon/discussions/43).
- **New copy of the image model**: new installs download [Unsloth's](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) Q4_K_M of Qwen-Image 2.1 Turbo (4.2 GB) instead of the Q4_K: the same model and the same 4-bit size, better supported and slightly better. Existing installs keep the Q4_K until you accept the update (see below).
- **Expand with a ratio lock** ([#42](https://github.com/GabrieleLozupone/kokon/issues/42)): Expand can keep the aspect ratio while you change the width or height, and offers 1080p, 1440p and 4K sizes. The size menu has the same sizes, and an exact size left in the fields can be picked again without retyping it.
- **Copy from the info panel**: the prompt, seed and settings of an image can be selected (on macOS and Linux too, and in the browser on a phone), and each has a copy button.
- **Reuse keeps a fixed seed**: reusing an image's settings with a fixed seed no longer picks a new one.
- **Visible selection on Linux** ([#41](https://github.com/GabrieleLozupone/kokon/issues/41)): selected text could not be seen in the dark theme.

**Updating from 0.11.0**: the app updates, then What's new offers the new image model file (4.2 GB). Choose **Update now and remove the old file**, or **Later** (Settings → Models has the same switch). The old Q4_K is deleted only after the new file is downloaded and checked, only from Kokon's own folder, and only if no model setup uses it. If you use **your own model files**, Kokon never switches them: if your image model is the Q4_K that Kokon used before, it suggests downloading the Q4_K_M into its own folder (your file stays where it is), or you keep your file and it won't ask again. Your images and settings are kept.

On macOS, if the first launch only says **The application "Kokon" can't be opened**, with just an OK button and no *Open Anyway* in System Settings, run `xattr -dr com.apple.quarantine /Applications/Kokon.app` in Terminal, then open Kokon. Details in the [README](https://github.com/GabrieleLozupone/kokon#macos-says-kokon-cannot-be-opened).

### Installing on Linux

```bash
sudo apt install ./Kokon_0.11.1_amd64.deb      # Ubuntu 22.04+, Debian 12+, Mint 21+
sudo dnf install ./Kokon-0.11.1-1.x86_64.rpm   # Fedora 40+
```

NVIDIA cards need the proprietary driver (the CUDA runtime comes with Kokon). AMD and Intel cards use Mesa's Vulkan driver.

### Thanks

- @Tiauma, for the dark-theme selection bug on Linux (#41) and the Expand ratio lock (#42).
- No-Tell786 on Reddit, for the info panel and the Reuse seed reports, and for comparing Mac speeds.
- Unsloth, for the Qwen-Image 2.1 Turbo GGUF files.

### Downloads

| File | SHA-256 |
|---|---|
| `Kokon_0.11.1_x64-setup.exe` (Windows) | `e9cfd72bf53fe095bca3ed78d59d4f2f3f0d9c657709255f5fc141b10f9b9c2f` |
| `Kokon_0.11.1_aarch64.dmg` (Mac, macOS 15+) | `a0f09488119ff97a27bcdad262747ef6804c2502793b64b9b3075b1e4a1880fa` |
| `Kokon_0.11.1_amd64.deb` (Ubuntu, Debian, Mint) | `211306d8b31d115e3c312053331067cea8eb7a193b1f7881ebf9fe1188894f14` |
| `Kokon-0.11.1-1.x86_64.rpm` (Fedora) | `042ca5a1dd8647fb881b3e8ab958a01cb53249ba62d8a0659030711fdbe25b77` |
