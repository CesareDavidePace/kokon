#!/usr/bin/env bash
# Builds the macOS release on an Apple Silicon Mac: kokon-service, kokon-mcp, Kokon.app and
# the disk image ("Kokon_<version>_aarch64.dmg"), copied to dist-release/.
#
#   bash scripts/build-release-mac.sh
#
# Same prerequisites as scripts/dev/mac-dev.sh (run it with --check). Takes ~5-15 minutes on
# the first run (optimized Rust build). Signed ad hoc (no Apple Developer ID): users open it
# the first time through System Settings → Privacy & Security → Open Anyway.
set -euo pipefail
cd "$(dirname "$0")/.."
TRIPLE=aarch64-apple-darwin

bash scripts/dev/mac-dev.sh --check

echo "1/5 Packages"
npm ci --no-audit --no-fund
(cd mcp && npm ci --no-audit --no-fund)

echo "2/5 Tests (Rust core + frontend + MCP)"
cargo test -p qls-core --release --quiet
npx vitest run
(cd mcp && npx vitest run)

echo "3/5 kokon-service (release)"
cargo build -p qls-service --release
mkdir -p src-tauri/binaries
cp target/release/kokon-service "src-tauri/binaries/kokon-service-$TRIPLE"

echo "4/5 kokon-mcp"
(cd mcp && npm run build:exe)

echo "5/5 Kokon.app + disk image (signed for the in-app updater)"
# Updates are verified against the public key in tauri.conf.json. Copy the private key from
# the PC that made it (~/.tauri/kokon-updater.key) once; see docs/releasing.md.
if [ -z "${TAURI_SIGNING_PRIVATE_KEY:-}" ]; then
  KEY="$HOME/.tauri/kokon-updater.key"
  [ -f "$KEY" ] || { echo "✗ Updater signing key not found: $KEY (see docs/releasing.md)"; exit 1; }
  export TAURI_SIGNING_PRIVATE_KEY="$KEY"
fi
npx tauri build --target "$TRIPLE"

DMG="$(ls -t target/$TRIPLE/release/bundle/dmg/*.dmg | head -1)"
mkdir -p dist-release
cp "$DMG" dist-release/
node scripts/release/updater.mjs fragment darwin-aarch64 "target/$TRIPLE/release/bundle/macos/Kokon.app.tar.gz"
codesign --verify --deep --strict "target/$TRIPLE/release/bundle/macos/Kokon.app" && echo "✓ signature (ad hoc) valid"
ls -lh dist-release/*.dmg
echo "Disk image: $(pwd)/dist-release/$(basename "$DMG")"
