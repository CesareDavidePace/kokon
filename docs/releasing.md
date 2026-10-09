# Releasing Kokon

How to publish a version that installed copies update to, on Windows and macOS.

## How updates work

- Once a day (unless turned off), the app reads
  `https://github.com/GabrieleLozupone/kokon/releases/latest/download/latest.json`.
- If its `version` is newer than the app's, a **Update to x.y.z** pill appears in the top bar.
- One click downloads the installer (Windows) or the app archive (Mac) and checks its
  signature against the public key in `src-tauri/tauri.conf.json`. Then it stops the local
  service, installs, and restarts Kokon. Settings, gallery and models are kept.
- **While the repository is private, the check finds nothing**, because GitHub serves release
  files of a private repo only to signed-in users. Updates start working when the repo is
  public.

## The signing key (do this once)

- **The private key** is `C:\Users\gabri\.tauri\kokon-updater.key`. It was generated on
  2026-10-09 with no password, and its public half is in `tauri.conf.json`.
- **Back it up** (password manager, encrypted drive). If it's lost, already installed copies
  can never be updated again: you'd need a new key and every user would have to reinstall
  by hand.
- **Never commit it.** It lives outside the repo on purpose.
- **Mac:** copy it to `~/.tauri/kokon-updater.key` (for example over AirDrop, then
  `chmod 600 ~/.tauri/kokon-updater.key`). Both build scripts read it from there, or from the
  `TAURI_SIGNING_PRIVATE_KEY` environment variable.

## Publishing a version

1. **Bump the version** in `src-tauri/tauri.conf.json`, `Cargo.toml` (workspace) and
   `package.json`, all to the same value. The updater only offers versions *higher* than the
   installed one.
2. **Windows:** `powershell -ExecutionPolicy Bypass -File scripts\build-release.ps1`
3. **Mac:** `bash scripts/build-release-mac.sh`. Then copy `dist-release/` from the Mac to the
   PC (or the other way around) so that one folder holds both platforms' files.
4. **Write the manifest:**
   `node scripts/release/updater.mjs manifest --notes "What changed, in one or two lines"`
5. **Publish** (the tag must be `v` + the version):
   ```bash
   gh release create v0.9.1 --title "Kokon 0.9.1" --notes "…" \
     "dist-release/Kokon Setup.exe" dist-release/Kokon_0.9.1_aarch64.dmg \
     dist-release/Kokon_0.9.1_x64-setup.exe dist-release/Kokon_0.9.1_aarch64.app.tar.gz \
     dist-release/latest.json
   ```
   `Kokon Setup.exe` and the `.dmg` are for people downloading by hand. The `_x64-setup.exe`
   and `.app.tar.gz` are what the updater downloads; `latest.json` points to them. A release
   marked *pre-release* is not "latest", so the updater ignores it.

## Testing an update

1. Install version A (for example 0.9.0) from its installer.
2. Publish version B (0.9.1) as above.
3. Open A. The check runs at startup, at most once a day; to force it, use Settings → General →
   **Check now**. Then click **Update to 0.9.1**. Kokon restarts as 0.9.1, and Settings →
   General shows the new version.
