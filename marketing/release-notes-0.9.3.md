## Kokon 0.9.3

**Cleaner images.** The fine grid pattern that showed on skin, hair and smooth areas came from Qwen-Image 2.1's image decoder (VAE): a faint 2-pixel checkerboard. Kokon now uses [madebyollin's texture-fix VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1), which removes it. The image, its colours and the speed stay the same. Thanks to @badraymen for reporting it in [#21](https://github.com/GabrieleLozupone/kokon/issues/21).

- **Use the model files you already have** ([#14](https://github.com/GabrieleLozupone/kokon/issues/14)): already downloaded Qwen-Image 2.1 for ComfyUI or another app? Point Kokon at those files instead of downloading them again. Kokon checks what each file is, and finds them again if they move or a drive was disconnected.
- **Pick your graphics card** ([#16](https://github.com/GabrieleLozupone/kokon/issues/16)): on computers with more than one GPU, setup lets you choose which one runs the model and sizes the preset for it.
- **Zoom and pan** in full screen and in the inpaint editor ([#19](https://github.com/GabrieleLozupone/kokon/issues/19)). The painted area now shows on the image while it is being edited.
- **Light and dark**: Settings → General → Appearance. Press `?` for the keyboard shortcuts.
- Fixes: the port-in-use hint points to the right setting ([#12](https://github.com/GabrieleLozupone/kokon/issues/12)), and durations round correctly ([#13](https://github.com/GabrieleLozupone/kokon/issues/13)).

Updating from 0.9.2: the app offers the new decoder (675 MB). If you accept, it replaces the old one in Kokon's folder. If you keep the old one, you can switch any time in Settings → Models. If you use your own model files, your VAE file is never deleted. Your images, models and settings stay as they are.

### Coming in 0.9.4: LoRA

- Add up to three LoRAs to an edit or a new image, each with its own weight, from the Advanced menu.
- A curated list of LoRAs tested with Kokon, one click to download, plus your own files.
- The list is a public `loras.json` in this repository: suggest a LoRA with an issue or a pull request. We test it with Qwen-Image 2.1 Turbo, then add it with its license and recommended weight.

### Downloads

| File | SHA-256 |
|---|---|
| `Kokon_0.9.3_x64-setup.exe` (Windows) | `TODO` |
| `Kokon_0.9.3_aarch64.dmg` (Mac) | `TODO` |
