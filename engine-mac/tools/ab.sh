#!/bin/bash
# Alternated A/B runs of kokon-engine-mac: each arm is a snapshot of a build (the executable and its resource bundle,
# which must sit beside it) plus environment variables; the arms run in turn, `rounds` times, and each run prints its
# per-step times, total, peak footprint and the page-ins it caused (vm_stat).
#
#   bash tools/ab.sh snapshot <name>                 # copy the current release build to $AB_DIR/<name>/
#   bash tools/ab.sh run <rounds> "<label>|<snapshot>|<ENV=x ...>" ... -- <kokon-engine-mac arguments>
#
# Example, an evaluation alone (fast) and then whole edits:
#   bash tools/ab.sh snapshot base   # before the change;  swift build -c release;  bash tools/ab.sh snapshot new
#   bash tools/ab.sh run 2 "base|base|X=1" "new|new|X=1" -- time-dit <dit.gguf> 64 32
#   bash tools/ab.sh run 2 "base|base|X=1" "new|new|X=1" -- generate <enc> <dit> <vae> "prompt" --ref in.png --mmproj <mmproj>
# Close other heavy apps (asitop too) first; do not build while it runs.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
dir="${AB_DIR:-$HOME/kokon-bench/ab}"
case "${1:-}" in
snapshot)
    name="${2:?a snapshot name}"
    rm -rf "$dir/$name" && mkdir -p "$dir/$name"
    cp "$here/.build/release/kokon-engine-mac" "$dir/$name/"
    cp -R "$here/.build/release/KokonEngineMac_KokonEngine.bundle" "$dir/$name/"
    echo "snapshot $dir/$name ($(git -C "$here" rev-parse --short HEAD)$(git -C "$here" diff --quiet || echo ' + changes'))"
    ;;
run)
    rounds="${2:?rounds}"; shift 2
    arms=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do arms+=("$1"); shift; done
    shift || true
    pageins() { vm_stat | awk '/Pageins/ {gsub("\\.", "", $2); print $2}'; }
    for r in $(seq 1 "$rounds"); do
        for arm in "${arms[@]}"; do
            IFS='|' read -r label snapshot envs <<< "$arm"
            p0=$(pageins)
            out=$(env $envs "$dir/$snapshot/kokon-engine-mac" "$@" 2>&1) || echo "$label r$r FAILED: $(echo "$out" | tail -2)"
            p1=$(pageins)
            steps=$(echo "$out" | grep -E "^step [0-9]+/" | sed -E 's/.*: ([0-9.]+) s/\1/' | tr '\n' ' ' || true)
            evals=$(echo "$out" | grep -E "^evaluation [0-9]+: " | sed -E 's/^evaluation [0-9]+: ([0-9.]+) s.*/\1/' | tr '\n' ' ' || true)
            total=$(echo "$out" | grep -oE "total [0-9.]+ s" | tail -1 || true)
            echo "$label r$r | page-ins $(( (p1 - p0) * 16384 / 1000000 )) MB | ${steps:+steps $steps| }${evals:+evaluations $evals| }${total:+$total | }$(echo "$out" | grep -o 'peak footprint [0-9.]* GB' || true)"
        done
    done
    ;;
*) sed -n 2,14p "$0"; exit 1 ;;
esac
