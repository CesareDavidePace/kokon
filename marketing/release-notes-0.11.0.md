## Kokon 0.11.0

Use Kokon from your phone, tablet or another computer, in a browser. The images are still made on the computer with the graphics card and saved in its gallery.

- **Other devices** ([#15](https://github.com/GabrieleLozupone/kokon/issues/15)): open **Settings → Other devices**, choose a password and open the address it shows on any device on the same network. Each browser asks for the password once. It is off until you turn it on, and nothing listens on the network before a password exists.
- **Safe by default**: the password is stored only as a hash, 5 wrong tries in a row make that device wait, and a browser can't touch files on the computer by path or change its folders, model files or network settings. Kokon's own token (desktop app, MCP server) still works only on the computer itself.
- **From outside your home**: a short guide to [Tailscale](https://github.com/GabrieleLozupone/kokon/blob/master/docs/remote-access.md#from-outside-your-home-tailscale) (or another VPN). Never open Kokon's port on your router.
- **In the browser**: add photos with the + button, by dropping files or by pasting; the download button saves results on the device you are using.
- **Docker for NVIDIA cards**: Kokon without a window, as the image `ghcr.io/gabrielelozupone/kokon` (published with each release, no models inside: the setup downloads them once into a volume). Download `docker/compose.yaml`, run `docker compose up -d`, set the password. Guide: [docs/docker.md](https://github.com/GabrieleLozupone/kokon/blob/master/docs/docker.md). An image for AMD and Intel cards (Vulkan) will follow.
- **Headless Linux**: `kokon-service --set-password`, then `kokon-service --persistent --remote`.
- The MCP server finds the service on Linux too.

**On a phone** Kokon has a layout of its own in the browser:

- Your projects as the home screen, a project's versions swiped sideways with the timeline under them, the image's actions in one row, a screen of its own for writing the prompt, and menus as sheets from the bottom. Add it to the phone's home screen (on iPhone, Share → **Add to Home Screen**) to open it like an app.
- **The same projects on every device**: `kokon-service` keeps the list of projects, so a version made on the phone shows up in the same project on the computer. Each app's own list from before 0.11 is merged in once, at the first start.
- Fast over Wi-Fi and mobile data: a project opens with one request, and the screen shows a screen-sized copy of each image (at most 1280 px). Zoom, download and copy still use the full image.
- Full screen and the paint editor draw large images at screen size, so a 4K upscale no longer runs Safari out of memory on an iPhone; pinch or double-tap still shows the real pixels.
- Swipe between images in full screen and in All images.
- **Take Photo** uploads the picture, with a Flip button first (Safari's camera doesn't mirror selfies the way the Camera app does).
- Images develop into view (a blurred thumbnail sharpening into the picture) instead of popping in over a checkerboard, also in the desktop app.
- The status bar follows the theme, and turns black in full screen.
- Browsers stay signed in when the computer running Kokon restarts or updates, and a page reloads itself once when that computer has updated to another version. Choosing a new password signs every browser out.

**Also in this release**

- **Your own model files**: a warning, in setup and at startup, when the image model is a known base Qwen-Image 2.1 file instead of Turbo (Kokon runs Turbo's 8 steps; the base model needs many more). Setup also says that FP8 files run much slower than the GGUF on AMD and Intel cards with Vulkan ([#38](https://github.com/GabrieleLozupone/kokon/issues/38)).
- **MCP**: an output folder listed in `QLS_MCP_ALLOWED_DIRS` now works (the service refused it), and the error for a folder outside the allowed ones names the variable ([#39](https://github.com/GabrieleLozupone/kokon/issues/39)).
- Closing the desktop app ends the service even when a browser tab is open; only a server started with `--persistent` or `--remote` keeps running. Ctrl+C and `docker stop` shut it down cleanly.

**Updating from 0.10.2**: only the app updates; the engine and the models stay as they are. Your images and settings are kept, and access from other devices stays off until you turn it on.

Full guide: [docs/remote-access.md](https://github.com/GabrieleLozupone/kokon/blob/master/docs/remote-access.md).

On macOS, if the first launch only says **The application "Kokon" can't be opened**, with just an OK button and no *Open Anyway* in System Settings, run `xattr -dr com.apple.quarantine /Applications/Kokon.app` in Terminal, then open Kokon. Details in the [README](https://github.com/GabrieleLozupone/kokon#macos-says-kokon-cannot-be-opened).

### Installing on Linux

```bash
sudo apt install ./Kokon_0.11.0_amd64.deb      # Ubuntu 22.04+, Debian 12+, Mint 21+
sudo dnf install ./Kokon-0.11.0-1.x86_64.rpm   # Fedora 40+
```

NVIDIA cards need the proprietary driver (the CUDA runtime comes with Kokon). AMD and Intel cards use Mesa's Vulkan driver.

### Thanks

- Everyone who asked for a browser interface and Docker in [#15](https://github.com/GabrieleLozupone/kokon/issues/15): @frostworx, @giveen, @Markus-, @openSourcerer9000, @pastukhvladyslav, @strid3r21 and @swingtempo.
- @XeonG, for reporting the MCP output folders (#39).

### Downloads

| File | SHA-256 |
|---|---|
| `Kokon_0.11.0_x64-setup.exe` (Windows) | `bdce51ae8a6035fb6324ef9ffc36cfe2bb52b1bf9637a90471b6332cd6fe6385` |
| `Kokon_0.11.0_aarch64.dmg` (Mac, macOS 15+) | `5e401c43795de798b20c47bf816d6672617ac75599c5b0acb1e40f1875c69188` |
| `Kokon_0.11.0_amd64.deb` (Ubuntu, Debian, Mint) | `3b3ae88639a12bd17518025ead114e3d52696895f75fbc922f6a042fc3fd33db` |
| `Kokon-0.11.0-1.x86_64.rpm` (Fedora) | `9f37114b3804dc4e1544fba06cd1aa9e1db470ee284c62e99aa6c395e196e69b` |
