#!/bin/bash
# Builds kokon-engine-mac and puts it where Kokon looks for it when "Kokon's Mac engine" is on (Settings →
# Performance) and the app has no copy of its own (Kokon.app and `tauri dev` have one, see stage-app.sh):
# <stable-diffusion.cpp folder>/engine-mac/, beside sd.cpp's engine-metal. The kernels and resources are the
# KokonEngineMac_KokonEngine.bundle next to the executable.
#
#   bash engine-mac/tools/install-dev.sh [stable-diffusion.cpp folder]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
target="${1:-$HOME/Library/Application Support/KokonData/sdcpp}/engine-mac"
(cd "$here" && swift build -c release)
mkdir -p "$target"
rm -rf "$target/KokonEngineMac_KokonEngine.bundle"
cp "$here/.build/release/kokon-engine-mac" "$target/"
cp -R "$here/.build/release/KokonEngineMac_KokonEngine.bundle" "$target/"
echo "installed in $target"
