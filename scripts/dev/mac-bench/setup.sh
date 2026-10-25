#!/usr/bin/env bash
# Downloads the engines compared by bench.py into a folder outside the repo (default
# ~/kokon-bench). Kokon's own engine and GGUF files are not downloaded: bench.py
# reads them from Kokon's install (config.json).
#
#   bash scripts/dev/mac-bench/setup.sh              # upstream sd.cpp + mflux + MLX 4-bit Turbo
#   bash scripts/dev/mac-bench/setup.sh --with-q8    # also the Q8_0 GGUF (7.6 GB), to see if Q4_K
#                                                    # dequantisation costs time on Metal
#
# Sizes: upstream sd.cpp zip ~90 MB, mflux venv ~1 GB, MLX 4-bit bundle 9.9 GB.
set -euo pipefail

ROOT="${KOKON_BENCH_DIR:-$HOME/kokon-bench}"
WITH_Q8=0
for a in "$@"; do
  case "$a" in
    --with-q8) WITH_Q8=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

# Pinned versions, so a rerun compares the same code.
SDCPP_TAG="master-1004-a85db96"
SDCPP_ZIP="sd-master-a85db96-bin-Darwin-macOS-26.6.2-arm64.zip"
MFLUX_VERSION="0.23.0"
MLX_REPO="OsaurusAI/Qwen-Image-2.1-Turbo-mflux-4bit"

mkdir -p "$ROOT/engines" "$ROOT/models" "$ROOT/results"

echo "== upstream stable-diffusion.cpp $SDCPP_TAG"
UP="$ROOT/engines/sdcpp-$SDCPP_TAG"
if [ ! -x "$UP/sd-cli" ]; then
  mkdir -p "$UP.download"
  gh release download "$SDCPP_TAG" -R leejet/stable-diffusion.cpp -p "$SDCPP_ZIP" -D "$UP.download" --clobber
  mkdir -p "$UP"
  unzip -q -o "$UP.download/$SDCPP_ZIP" -d "$UP"
  # The zip may hold a top folder; flatten it so sd-cli sits in $UP.
  if [ ! -e "$UP/sd-cli" ]; then
    found="$(find "$UP" -name sd-cli -type f | head -1)"
    [ -n "$found" ] && mv "$(dirname "$found")"/* "$UP"/
  fi
  xattr -dr com.apple.quarantine "$UP" 2>/dev/null || true
  chmod +x "$UP"/sd-* 2>/dev/null || true
  rm -rf "$UP.download"
fi
"$UP/sd-cli" --help 2>&1 | sed -n 1p || true

echo "== mflux $MFLUX_VERSION (own venv)"
MF="$ROOT/engines/mflux-$MFLUX_VERSION"
if [ ! -x "$MF/bin/mflux-generate-qwen-2.1" ]; then
  uv venv --python 3.12 "$MF"
  # Pillow 13 does not import on Python 3.12 (numbers._IntegralLike is missing).
  VIRTUAL_ENV="$MF" uv pip install --python "$MF/bin/python" "mflux==$MFLUX_VERSION" "pillow<13"
fi
"$MF/bin/python" -I -c "import mflux, mlx.core as mx; print('mflux', mflux.__version__ if hasattr(mflux, '__version__') else '?', 'mlx', mx.__version__)" || true

echo "== $MLX_REPO"
MD="$ROOT/models/qwen-image-2.1-turbo-mflux-4bit"
"$MF/bin/hf" download "$MLX_REPO" --local-dir "$MD" || "$MF/bin/huggingface-cli" download "$MLX_REPO" --local-dir "$MD"
du -sh "$MD"

if [ "$WITH_Q8" = 1 ]; then
  echo "== Q8_0 GGUF"
  "$MF/bin/hf" download unsloth/Qwen-Image-2.1-Turbo-GGUF qwen-image-2.1-turbo-Q8_0.gguf --local-dir "$ROOT/models/gguf"
fi

echo "done: $ROOT"
