## Kokon 0.11.0

Use Kokon from your phone, tablet or another computer, in a browser. The images are still made on the computer with the graphics card and saved in its gallery.

- **Other devices** ([#15](https://github.com/GabrieleLozupone/kokon/issues/15)): open **Settings → Other devices**, choose a password and open the address it shows on any device on the same network. Each browser asks for the password once. It is off until you turn it on, and nothing listens on the network before a password exists.
- **Safe by default**: the password is stored only as a hash, 5 wrong tries in a row make that device wait, and a browser can't touch files on the computer by path or change its folders, model files or network settings. Kokon's own token (desktop app, MCP server) still works only on the computer itself.
- **From outside your home**: a short guide to [Tailscale](https://github.com/GabrieleLozupone/kokon/blob/master/docs/remote-access.md#from-outside-your-home-tailscale) (or another VPN). Never open Kokon's port on your router.
- **In the browser**: add photos with the + button, by dropping files or by pasting; the download button saves results on the device you are using.
- **Docker for NVIDIA cards**: Kokon without a window, with the models, settings and gallery in volumes and an example `docker compose` file. Guide: [docs/docker.md](https://github.com/GabrieleLozupone/kokon/blob/master/docs/docker.md). An image for AMD and Intel cards (Vulkan) will follow.
- **Headless Linux**: `kokon-service --set-password`, then `kokon-service --persistent --remote`.
- The MCP server finds the service on Linux too.

**Updating from 0.10.2**: only the app updates; the engine and the models stay as they are. Your images and settings are kept, and access from other devices stays off until you turn it on.

Full guide: [docs/remote-access.md](https://github.com/GabrieleLozupone/kokon/blob/master/docs/remote-access.md).

### Installing on Linux

```bash
sudo apt install ./Kokon_0.11.0_amd64.deb      # Ubuntu 22.04+, Debian 12+, Mint 21+
sudo dnf install ./Kokon-0.11.0-1.x86_64.rpm   # Fedora 40+
```

NVIDIA cards need the proprietary driver (the CUDA runtime comes with Kokon). AMD and Intel cards use Mesa's Vulkan driver.

### Thanks

- Everyone who asked for a browser interface and Docker in [#15](https://github.com/GabrieleLozupone/kokon/issues/15): @frostworx, @giveen, @Markus-, @openSourcerer9000, @pastukhvladyslav, @strid3r21 and @swingtempo.

### Downloads

| File | SHA-256 |
|---|---|
| `Kokon_0.11.0_x64-setup.exe` (Windows) | `TODO` |
| `Kokon_0.11.0_aarch64.dmg` (Mac, macOS 15+) | `TODO` |
| `Kokon_0.11.0_amd64.deb` (Ubuntu, Debian, Mint) | `TODO` |
| `Kokon-0.11.0-1.x86_64.rpm` (Fedora) | `TODO` |
