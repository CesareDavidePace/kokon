## Kokon 0.9.2

**Native 2K edits, faster.** On graphics cards with 16 GB or more, 2K edits are now drawn at full size instead of enlarged afterwards: sharper, and about a third faster (41 s instead of 66 s for a 2752×1536 edit on an RTX 5070 Ti).

- **Reference detail** (Advanced menu): your source images go to the model at 1 MP by default, leaving more room for the result. Choose Full for finer source detail. Every new chat starts at 1 MP.
- **Faster decoding** on 16 GB+ cards: tiles shaped to the image (7.8 s → 4.75 s at 2752×1536).
- **Official stable-diffusion.cpp.** The MoSR upscaler Kokon contributed was merged upstream ([#2128](https://github.com/leejet/stable-diffusion.cpp/pull/2128)), so Kokon now runs the official release `master-964-1d0a58a` instead of its own fork. The app checks every few days for a newer engine tested with Kokon and offers it in one click.
- **Prompt tips**: a hint under the prompt and a Prompt tips panel. Describing the scene, not only the subject, keeps the action you asked for.

Updating from 0.9.1: the app asks to download the new engine (~340 MB on NVIDIA, ~30 MB with Vulkan or on a Mac). Your images, models and settings stay as they are.

### Downloads

| File | SHA-256 |
|---|---|
| `Kokon_0.9.2_x64-setup.exe` (Windows) | `TBD` |
| `Kokon_0.9.2_aarch64.dmg` (Mac) | `TBD` |
