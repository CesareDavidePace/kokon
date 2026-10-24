# Use Kokon from other devices

Kokon can open its interface in a browser on your phone, tablet or another computer. The images are
still made on the computer with the graphics card and saved in its gallery; the other device only
shows the interface, uploads the photos you pick and downloads the results you save.

It is off until you turn it on, and it always asks for a password.

## On your home network

1. On the computer running Kokon, open **Settings → Other devices**.
2. Choose a password (at least 8 characters) and click **Set and turn on**.
3. Settings shows a QR code and an address such as `http://192.168.1.20:39152`. Scan the code with
   the phone's camera (or type the address in a browser on the same network) and enter the password.
   That browser stays signed in for 90 days.
4. To keep Kokon on the phone's home screen like an app: on iPhone, Share → **Add to Home Screen**;
   on Android, the browser's menu → **Add to Home screen**. The icon opens Kokon full screen (on iPhone
   it asks for the password once more the first time).

The router may give the computer a different address later (after a restart, or when it switches
between cable and Wi-Fi), and the saved icon stops working. To keep it the same, reserve the address
in your router's settings (usually "DHCP reservation" or "static IP" for this computer); or scan the
QR code again.

The first time, your system may ask whether to let Kokon on the network (Windows Defender Firewall,
macOS). Allow it on **private** networks only.

It works while Kokon is open on that computer. To keep it running without the window, use a headless
server or Docker (below).

A browser stays signed in when Kokon restarts or updates; after an update the page reloads itself
with the new version. **Changing the password** signs every browser out. **Remove** turns access off and deletes the
password. Forgot it? Set a new one in the same place.

### On a server without a screen (Linux)

`kokon-service` comes with the Linux packages (`/usr/bin/kokon-service`). Set the password once, then
start it with access on:

```bash
kokon-service --set-password         # asks twice, nothing is shown while you type
kokon-service --persistent --remote  # keeps running; the browser UI is on port 39152
```

The first time, open the address in a browser and follow the setup to download the model. A password
set while the service runs takes effect within a few seconds. For a container, see
[docker.md](docker.md).

## What is protected, and what is not

- The password is stored only as a hash (Argon2id) in Kokon's data folder; it never appears in the
  settings or the logs.
- After 5 wrong passwords in a row, that device waits 1 minute, then longer after each new mistake
  (up to 15 minutes). Other devices are not affected.
- Nothing listens on the network until a password exists and access is on. Kokon's own token, used by
  the desktop app and the MCP server, works only on the computer itself.
- A browser can do what the interface does: create and edit, the gallery, Settings. It cannot read or
  write files on the computer by path, change folders or model files, or turn network access on or
  off. Those stay in the desktop app.
- Web pages from other sites cannot use your signed-in session.

**The connection is plain HTTP, not encrypted.** On your home Wi-Fi that is fine. On a shared network
(office, university, café) someone on the same network could read the password and your images on the
way. Use it only on networks you trust, or through Tailscale (below), which encrypts everything.

## From outside your home: Tailscale

Never open Kokon's port on your router (port forwarding): anyone on the internet could then try the
password. Use a private network instead. [Tailscale](https://tailscale.com) is free for personal use
and takes a few minutes:

1. Install Tailscale on the computer running Kokon and on your phone or laptop, and sign in with the
   same account on both.
2. Turn on access in Kokon (above).
3. On the other device, with Tailscale on, open `http://<computer name>:39152`. The computer name is
   in the Tailscale app (for example `http://gaming-pc:39152`), or use its `100.x.y.z` address.

Traffic between your devices is encrypted by Tailscale, wherever you are.

**Optional, HTTPS:** with [MagicDNS and HTTPS](https://tailscale.com/kb/1153/enabling-https) turned
on for your tailnet, run on the Kokon computer:

```bash
tailscale serve --bg 39152
```

Then open `https://<computer name>.<your tailnet>.ts.net`. With HTTPS the browser also lets Kokon copy
images to the clipboard.

Other private networks work the same way: WireGuard, ZeroTier, or your router's own VPN. Connect to
your home network, then open the same address as at home.

## What is different in a browser

- **The same projects everywhere:** every device sees the same list of projects, kept by the computer
  running Kokon. A version made on the phone shows up in that project on the computer.
- **On a phone** the interface has a layout of its own: your projects as the home screen, a project's
  versions swiped sideways, the image's actions in one row, and a screen of its own for writing the prompt.
- **Adding photos:** the + button, dropping files on the page or pasting; on a phone, pick from the photo
  library or take a photo (a photo just taken is shown first, with a **Flip** button for selfies). They
  are uploaded to Kokon.
- **Saving:** the download button saves the image on the device you are using. On an iPhone over plain
  HTTP it shows the image full screen: press and hold it, then **Save to Photos** (with HTTPS, the share
  sheet opens instead).
- **Copying an image** needs HTTPS (or the computer itself at `localhost`); on plain HTTP the button
  is hidden.
- Not available: show in folder, open the logs, LoRA and `.cube` files from disk, choosing model files
  you already have, updates. Do those on the computer running Kokon; settings that name a folder show
  its value there.

## It does not connect

- Check that both devices are on the same network (or both on Tailscale) and that **Settings → Other
  devices** shows the address.
- A firewall may block the port. Windows: set your home network as **Private** (Settings → Network &
  internet → your connection), then allow Kokon on private networks only (Windows Security →
  Firewall → Allow an app). If Windows calls your home network Public and you allowed Kokon there,
  it would also be reachable on any public Wi-Fi you join. Linux with ufw: `sudo ufw allow from 192.168.0.0/16 to any port 39152`.
- Another program may use port 39152: Settings shows the error; choose another port there.
