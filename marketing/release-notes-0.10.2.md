## Kokon 0.10.2

A new engine, new images from reference images, and cleaner edits.

- **New image from reference images**: with New image, the images you add are now references (a person, a product, a style) for a new picture. Its size comes from your quality and aspect ratio, not from the first image, and Prompt+ writes the prompt with that in mind. Also in MCP (`generate_image` with `reference_images`) and the [HTTP API](https://github.com/GabrieleLozupone/kokon/blob/master/docs/http-api.md).
- **Read the prompt before it runs**: the Prompt+ menu has three choices: off, write and make, or write the prompt into the box so you can read and change it first.
- **Longer Prompt+ prompts** for crowded scenes: they were cut at 512 tokens.
- **Cleaner edits**: an edit made with the same seed as the image it starts from could come out over-sharpened and crunchy, and often ignored the instruction. Kokon now picks another seed in that case and tells you why. MCP tools tell agents not to reuse the source's seed.
- **Faster decoding on AMD and Intel cards (Vulkan)**: the last step of every image, the VAE decode, could take 25-70 s on AMD cards, because it asked for more memory in one piece than the driver allows (2 GiB). With Vulkan, Kokon now decodes in tiles of 512 px, which takes a few seconds. Settings → Diagnostics shows the tile size the engine uses.
- **New engine** ([kokon-1013-1814814](https://github.com/GabrieleLozupone/stable-diffusion.cpp/releases/tag/kokon-1013-1814814)): Kokon's engine now follows stable-diffusion.cpp up to fd21ecf. It includes a Vulkan fix for AMD and Intel cards without Resizable BAR (SAM on AMD): the engine could put the model in system memory instead of on the card, which makes every step much slower. If your AMD or Intel card was slow on Vulkan, please try again and tell us your seconds per step.
- `kokon-mcp --help` ([#36](https://github.com/GabrieleLozupone/kokon/pull/36)).
- For developers: every route of the local HTTP API is documented in [http-api.md](https://github.com/GabrieleLozupone/kokon/blob/master/docs/http-api.md), with a macOS `curl` example ([#37](https://github.com/GabrieleLozupone/kokon/pull/37)).

**Updating from 0.10.x**: the app updates and, if your copy hasn't fetched it already, offers the new engine once (about 345 MB with an NVIDIA card, 30 MB with Vulkan or on a Mac). The models stay as they are. Your images and settings are kept.

On macOS, if the first launch only says **The application "Kokon" can't be opened**, with just an OK button and no *Open Anyway* in System Settings, run `xattr -dr com.apple.quarantine /Applications/Kokon.app` in Terminal, then open Kokon. Details in the [README](https://github.com/GabrieleLozupone/kokon#macos-says-kokon-cannot-be-opened).

### Installing on Linux

```bash
sudo apt install ./Kokon_0.10.2_amd64.deb      # Ubuntu 22.04+, Debian 12+, Mint 21+
sudo dnf install ./Kokon-0.10.2-1.x86_64.rpm   # Fedora 40+
```

NVIDIA cards need the proprietary driver (the CUDA runtime comes with Kokon). AMD and Intel cards use Mesa's Vulkan driver.

### Thanks

- @shrinivas-sn, for `kokon-mcp --help` (#36) and the HTTP API reference (#37).
- Solembumm3 on Reddit, for the RX 6700 XT logs and tests with Resizable BAR and VAE tile sizes.
- hakanavgin on Reddit, for tracking the slow decode on an RX 9070 XT down to AMD's 2 GiB buffer limit.
- The stable-diffusion.cpp and ggml maintainers, for the Vulkan fix for cards without Resizable BAR and everything else in the new engine.

### Downloads

| File | SHA-256 |
|---|---|
| `Kokon_0.10.2_x64-setup.exe` (Windows) | `492c338c54b69ff01f0d361fc42eb618bc15edaff60e0f2f47ad416de94dca7f` |
| `Kokon_0.10.2_aarch64.dmg` (Mac, macOS 15+) | `912dcdf21f6dd4c90e58e7a25621cf84e0cecf4a863a7660629669db5989f6d3` |
| `Kokon_0.10.2_amd64.deb` (Ubuntu, Debian, Mint) | `fa26536119e4186c132a6fd39912752efdcea2b2f80e6ae74748f889b14afa05` |
| `Kokon-0.10.2-1.x86_64.rpm` (Fedora) | `e11638647a69abf2083882d72a0b6273a98fb8535465a24cc6f0ae5704dd3268` |
