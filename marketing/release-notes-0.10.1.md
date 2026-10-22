## Kokon 0.10.1

A small release with the fixes you reported after 0.10.0.

- **Your own model files**: setup no longer picks a text-only Qwen3 8B (for example a prompt-enhancer model in a ComfyUI folder) as the Qwen3-VL text encoder. That choice made the engine fail to start; now only the real Qwen3-VL file is used.
- **Linux with an NVIDIA card** ([#15](https://github.com/GabrieleLozupone/kokon/issues/15)): Kokon turns off WebKitGTK's DMA-BUF renderer when it finds the NVIDIA driver, the usual cause of a blank window. If your window still stays blank, please tell us in #15.
- **Small windows** ([#32](https://github.com/GabrieleLozupone/kokon/issues/32)): the progress card fits: the title stays on one line clear of Stop, and details hide when there is no room. A long Prompt+ prompt now scrolls inside its frame, and Stop stays visible.
- **A softer finish**: the Kokon logo a result appears from is smaller, and in the dark theme it is grey instead of near-white, so it no longer flashes on a big screen.
- **Image details fit the window**: the details of an image with a long prompt no longer run off the top of a small window; the panel scrolls instead (the same for every menu).
- **On the Mac**: in full screen the top bar no longer leaves an empty gap where the window buttons were, and the dark theme is a step lighter, closer to macOS's own dark apps.
- **Support Kokon**: Kokon stays free and open source. If it's useful to you, you can now leave a tip on [Ko-fi](https://ko-fi.com/gabrielelozupone), from Settings → General or [kokon.app/support](https://kokon.app/support/). Tips are gifts and don't unlock anything.
- **Open your images folder** from Settings and from the gallery, and a link to the GitHub page in Settings ([#34](https://github.com/GabrieleLozupone/kokon/issues/34)).
- `kokon-service --version` and `--help` ([#31](https://github.com/GabrieleLozupone/kokon/pull/31)).
- For contributors: every pull request now runs the UI and MCP tests ([#25](https://github.com/GabrieleLozupone/kokon/pull/25)).

**Updating from 0.10.0**: only the app updates; the engine and the models stay as they are. Your images and settings are kept.

On macOS, if the first launch only says **The application "Kokon" can't be opened**, with just an OK button and no *Open Anyway* in System Settings, run `xattr -dr com.apple.quarantine /Applications/Kokon.app` in Terminal, then open Kokon. Details in the [README](https://github.com/GabrieleLozupone/kokon#macos-says-kokon-cannot-be-opened).

### Installing on Linux

```bash
sudo apt install ./Kokon_0.10.1_amd64.deb      # Ubuntu 22.04+, Debian 12+, Mint 21+
sudo dnf install ./Kokon-0.10.1-1.x86_64.rpm   # Fedora 40+
```

NVIDIA cards need the proprietary driver (the CUDA runtime comes with Kokon). AMD and Intel cards use Mesa's Vulkan driver.

### Thanks

- @giveen, for `--version` and `--help` in the service (#31).
- @petrovicistefan, for running the tests on every pull request (#25).
- @badraymen, for the list of suggestions behind the folder and GitHub links (#34).
- @electric-dragon-80000v, for the report on small windows (#32).
- Sdesser on Reddit, for saying the white logo hurt the eyes on a large screen.
- chillahc on Reddit, for finding the text-encoder mix-up with files from ComfyUI.

### Downloads

| File | SHA-256 |
|---|---|
| `Kokon_0.10.1_x64-setup.exe` (Windows) | `3a47de6632db895839040a19a9a39389ace87c91c0721a57ce26fa1db721be10` |
| `Kokon_0.10.1_aarch64.dmg` (Mac, macOS 15+) | `e944f3d78c9ffbc1471a4e837d108b967a89eda4ea7bdca50cf57420c53a57b2` |
| `Kokon_0.10.1_amd64.deb` (Ubuntu, Debian, Mint) | `c6d1496a5b8aa051da97cfff990834848d417d6dad5f7fce3bcd8b704b278ea3` |
| `Kokon-0.10.1-1.x86_64.rpm` (Fedora) | `ae08d602ecd8c26926a780cf37af17b1ee4cfb352d30563cd3fa7bf66290400b` |
