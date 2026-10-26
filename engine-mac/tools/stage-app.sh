#!/bin/bash
# Builds kokon-engine-mac and puts it where Tauri packs it into Kokon.app (scripts/dev/mac-dev.sh and
# scripts/build-release-mac.sh call it): the executable as a sidecar next to kokon-service
# (src-tauri/binaries, Contents/MacOS in the app), its kernels as a resource (Contents/Resources/engine-mac).
#
#   bash engine-mac/tools/stage-app.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
(cd "$root/engine-mac" && swift build -c release 2>&1 | tail -1)
mkdir -p "$root/src-tauri/binaries"
cp "$root/engine-mac/.build/release/kokon-engine-mac" "$root/src-tauri/binaries/kokon-engine-mac-aarch64-apple-darwin"
rm -rf "$root/src-tauri/resources/engine-mac" && mkdir -p "$root/src-tauri/resources/engine-mac"
cp -R "$root/engine-mac/.build/release/KokonEngineMac_KokonEngine.bundle" "$root/src-tauri/resources/engine-mac/"
