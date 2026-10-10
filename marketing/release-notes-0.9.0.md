First public beta of Kokon: edit and create photos by describing the change, on your own GPU, with Qwen-Image 2.1 Turbo. One installer, no Python, no cloud.

Kokon is a native desktop app built on [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp). It picks the mode, the working size and the memory settings for you, and includes an MCP server and a local HTTP API so AI agents can create and edit images on your machine.

## Download

| | File |
| --- | --- |
| **Windows 10/11** | `Kokon_0.9.0_x64-setup.exe` (installs for your user only, no admin rights) |
| **Mac (Apple Silicon, beta)** | coming to this release shortly |

On first launch Kokon detects your graphics card (or your Mac's chip and memory), recommends a model preset and downloads about 12 GB of models once. Downloads resume if interrupted and every file is checked against its SHA-256. After that Kokon works offline.

## Highlights

- **Create, edit, combine:** a sentence makes a new image; a photo plus a sentence edits it, keeping its size and aspect ratio; several photos can be combined ("image 1", "image 2").
- **Brush edits:** paint an area and choose Change, Remove, Replace or Add. Kokon marks the area on a second reference image so the model knows exactly where to work.
- **Expand** (outpainting) by dragging the edges, and **upscale** to 2×, 4K or a custom width with an ESRGAN upscaler.
- **Qwen's official 8-step Qwen-Image 2.1 Turbo** model for every preset, with three quality presets: Fast (~1 MP, default), Balanced (~1.5K) and Quality (native 2K).
- **Before/after** compare (Slider, Flicker, Changes), undo/redo between versions, and a gallery that keeps every image's full recipe (also embedded in the PNG).
- **iPhone photos:** HEIC opens directly; HDR gain-map photos can be saved as Ultra HDR JPEG.
- **Private by design:** no account, no uploads, no telemetry. Private mode keeps nothing.
- **Time estimates** that learn your machine's speed after the first image.
- **MCP server** (`kokon-mcp`) for Claude Code, Cursor and other MCP hosts, plus a token-protected HTTP API on `127.0.0.1`. Tools: `generate_image`, `edit_image`, `edit_images`, `inpaint_image`, `outpaint_image`, `upscale_image`, `get_image_info`, `get_generator_status`.
- **One-click updates:** Kokon checks once a day for a new version (you can turn this off) and installs it, after verifying its signature, when you click **Update**.

## Requirements

- **Windows + NVIDIA:** 12 GB VRAM or more (RTX 3060 12 GB, 4070 and up; 16 GB+ recommended), CUDA engine. 16 GB RAM, 15 GB free disk.
- **Windows + AMD or Intel:** 12 GB VRAM or more, Vulkan engine (beta). Integrated graphics are not supported.
- **Mac:** Apple Silicon (M1 or newer), macOS 26, 16 GB unified memory minimum, 32 GB or more recommended (beta).

## Known limitations

- **AMD and Intel are unverified.** The Vulkan engine works on NVIDIA cards, but nobody has tried it on a real AMD or Intel card yet. It is slower than CUDA, and its first image takes longer. Please report how it goes.
- **Macs are slow.** On a 16 GB M2 Pro a 1-megapixel image takes about 4 minutes, in the lightest mode. Chips with more GPU cores and 32 GB or more should be faster, and get native 2K. So far the Mac build has been tested only on a 16 GB M2 Pro.
- **8 GB cards** are not supported: 12 GB is the smallest size Kokon is designed for.
- **No Linux build** yet.

## Model licenses

Kokon's code is open source under the GNU AGPL-3.0. The models are downloaded from their authors and keep their own licenses: Qwen-Image 2.1 Turbo is under the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) (research and evaluation only, no commercial use without a license from the Qwen team), the Qwen3-VL text encoder is Apache-2.0, and the optional 4x-UltraSharp upscaler is CC BY-NC-SA 4.0 (non-commercial only). Details in [NOTICE](https://github.com/GabrieleLozupone/kokon/blob/master/NOTICE).

## First launch

- **Windows:** SmartScreen may say "Windows protected your PC" because the app is new and has no download history yet. Click **More info → Run anyway**. You can check the installer against the SHA-256 below.
- **Mac:** Kokon is not notarized by Apple, so macOS blocks it the first time. Try to open it once, then go to **System Settings → Privacy & Security** and click **Open Anyway**. In-app updates don't trigger this again.

## Checksum

```
SHA-256  Kokon_0.9.0_x64-setup.exe
e27b4a8b779644232443e6b3bedfe4b009c30d511b682822cea51aeccd42a608
```

## Feedback

This is a beta, built by one person with the help of AI. Please report problems, and how Kokon runs on your GPU or Mac, in [Issues](https://github.com/GabrieleLozupone/kokon/issues). For questions and ideas, use [Discussions](https://github.com/GabrieleLozupone/kokon/discussions).
