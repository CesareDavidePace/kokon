Better enlargements and a gallery you can browse. Kokon 0.9.1 lets you choose how results are enlarged to their final size, and compare the options side by side.

## Download

| | File |
| --- | --- |
| **Windows 10/11** | `Kokon_0.9.1_x64-setup.exe` (installs for your user only, no admin rights) |
| **Mac (Apple Silicon, beta)** | `Kokon_0.9.1_aarch64.dmg` (macOS 26 or newer, 32 GB of memory recommended) |

Already on 0.9.0? Kokon offers the update itself (**Update to 0.9.1** in the top bar), or install over it with the file above. Your gallery, settings and models are kept.

## What's new

- **Superscale-SP is the new default upscaler** on Windows. It's sharper and more natural than UltraSharp, which was great on some images but smoothed skin and textures into an artificial look on others. With the new engine it is also about twice as fast as 0.9.0's upscaler: 13 s instead of 25 s to enlarge a 1472×1472 image 4× on an RTX 5070 Ti.
- **Choose your upscaler** in the Advanced menu next to the prompt, or in Settings → Models: Superscale-SP, **MoSR** (experimental, new), UltraSharp (optional download) or a plain resize (Lanczos). Missing models download from the same menu.
- **Upscaler strength:** a slider that mixes the upscaler's result with a plain resize, to tone down texture the model invents.
- **Compare upscalers:** the same crops (a face, a JPEG photo, text, a night street, a landscape, a 4× upscale) through each upscaler, with what each one is good and bad at.
- **On a Mac, enlargements use a plain resize by default.** AI upscalers take minutes on Apple GPUs (Superscale-SP: 5 min 45 s to enlarge 1024×1024 to 2048×2048 on an M2 Pro, against about a second). You can still pick one in the menu.
- **Gallery preview:** clicking an image in the gallery opens a large preview. Browse with the arrow keys, F or double-click for full screen, Enter to open it in its project.
- **Full screen follows the project's timeline:** the arrow keys step through versions without leaving full screen.
- **Renaming** starts from the current name instead of replacing it.

## First start after updating

Kokon shows what's new and offers a one-time download: a newer build of stable-diffusion.cpp, needed for the new upscaler support, and, on Windows, the Superscale-SP model. That's about 405 MB with the CUDA engine, 97 MB with Vulkan and 35 MB on a Mac. You can postpone it; until then, enlargements use a plain resize. Your models are not downloaded again.

The engine is [Kokon's build of stable-diffusion.cpp](https://github.com/GabrieleLozupone/stable-diffusion.cpp/releases/tag/kokon-mosr-1): the same commit as 0.9.0 with MoSR support added ([pull request](https://github.com/leejet/stable-diffusion.cpp/pull/2128) to the original project). Qwen-Image results are unchanged.

## Known limitations

- **MoSR is experimental.** It is the fastest upscaler and good for 4× enlargements, but on clean images enlarged less than about 3× it draws a fine grid pattern. Compare upscalers shows it.
- **Superscale-SP** can draw a faint grid on JPEG photos and in dark areas. Lower the upscaler strength if you see one.
- **AMD and Intel are unverified.** The Vulkan engine works on NVIDIA cards, but nobody has tried it on a real AMD or Intel card yet. Please report how it goes.
- **Macs are slow.** On a 16 GB M2 Pro a 1-megapixel image takes about 4 minutes. So far the Mac build has been tested only on a 16 GB M2 Pro.
- **8 GB cards** are not supported, and there is **no Linux build** yet.

## Model licenses

Kokon's code is open source under the GNU AGPL-3.0. The models keep their own licenses: Qwen-Image 2.1 Turbo is under the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) (research and evaluation only, no commercial use without a license from the Qwen team), and the Qwen3-VL text encoder is Apache-2.0. Upscalers: Superscale-SP by Nmkd is WTFPL, MoSR (4xNomos2_hq_mosr by Philip Hofmann, included in the app) is CC BY 4.0, and the optional UltraSharp by Kim2091 is CC BY-NC-SA 4.0 (non-commercial only). Details in [NOTICE](https://github.com/GabrieleLozupone/kokon/blob/master/NOTICE).

## First launch

- **Windows:** SmartScreen may say "Windows protected your PC" because the app has little download history yet. Click **More info → Run anyway**. You can check the installer against the SHA-256 below.
- **Mac:** Kokon is not notarized by Apple, so macOS blocks a fresh install the first time. Try to open it once, then go to **System Settings → Privacy & Security** and click **Open Anyway**. In-app updates don't trigger this again.

## Checksums

```
SHA-256  Kokon_0.9.1_x64-setup.exe
TODO

SHA-256  Kokon_0.9.1_aarch64.dmg
TODO
```

## Feedback

Kokon is a beta, built by one person with the help of AI. Please report problems, and how Kokon runs on your GPU or Mac, in [Issues](https://github.com/GabrieleLozupone/kokon/issues). For questions and ideas, use [Discussions](https://github.com/GabrieleLozupone/kokon/discussions).
