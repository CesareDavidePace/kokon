"""Qwen-Image 2.1 Turbo on Apple Silicon: Kokon's engine against other engines and settings.

Run `bash scripts/dev/mac-bench/setup.sh` once (upstream sd.cpp, mflux, MLX 4-bit weights), close
Kokon (its sd-server holds Metal memory), then:

    python3 -I scripts/dev/mac-bench/bench.py --list                 # variants and cases
    python3 -I scripts/dev/mac-bench/bench.py                        # default set, t2i + edit
    python3 -I scripts/dev/mac-bench/bench.py --only kokon,mflux-4bit --cases t2i --repeat 2
    python3 -I scripts/dev/mac-bench/bench.py --size 512             # quick pass (not Turbo's size)

Every variant changes one thing against `kokon`, which uses the flags Kokon gives sd-server on
a 16 GB Mac (crates/qls-core/src/process.rs: Disk placement, mmap, flash attention, VAE tiles
of 768, prefix cache q8_0). Each run is a fresh process, so model loading is part of the time,
as on a first generation in Kokon (Disk placement reloads the weights on every run anyway).

Writes <bench>/results/<time>/: results.json, results.csv, report.html (timing table + images),
and per run the log and the image. Phases come from sd.cpp's log (load, text encoding,
sampling, decode) and from mflux's progress bar (per-step time; the rest is "fixed").
PSNR is against the first variant of the same `family` (same engine and noise), so it shows
what an approximate setting (step caching, f16 compute) costs in fidelity; across engines the
noise differs and only the images can be compared.
"""

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
BENCH = Path(os.environ.get("KOKON_BENCH_DIR", HOME / "kokon-bench"))
KOKON_CONFIG = HOME / "Library" / "Application Support" / "Kokon" / "config.json"
KOKON_DATA = HOME / "Library" / "Application Support" / "KokonData"
KOKON_ENGINE = KOKON_DATA / "sdcpp" / "engine-metal"
UPSTREAM_ENGINE = BENCH / "engines" / "sdcpp-master-1004-a85db96"
MFLUX = BENCH / "engines" / "mflux-0.23.0"
MLX_4BIT = BENCH / "models" / "qwen-image-2.1-turbo-mflux-4bit"
Q8_GGUF = BENCH / "models" / "gguf" / "qwen-image-2.1-turbo-Q8_0.gguf"

SEED = 42
STEPS = 8  # Turbo, CFG 1, Euler: what planner.rs uses

CASES = {
    # 1 MP square: Fast on a Mac renders this area directly.
    "t2i": {
        "prompt": "A ceramic teapot on a wooden kitchen table by a window, soft morning light, "
        "steam rising, a small handwritten label that reads 'Earl Grey', photorealistic",
        "size": (1024, 1024),
        "ref": None,
    },
    # The site's cabin example: one reference at Fast's 16:9 working size.
    "edit": {
        "prompt": "Make it winter: snow on the roof and the ground, frost on the windows, keep everything else",
        "size": (1376, 768),
        "ref": KOKON_DATA / "examples" / "cabin.jpg",
    },
}


def kokon_profile():
    cfg = json.loads(KOKON_CONFIG.read_text())
    prof = next(p for p in cfg["profiles"] if p["id"] == cfg["activeProfile"])
    return cfg, prof


def sdcpp_base(engine: Path, diffusion: Path | None = None):
    """sd-cli flags equal to Kokon's sd-server launch on a 16 GB Mac (process.rs)."""
    cfg, prof = kokon_profile()
    a = [
        str(engine / "sd-cli"),
        "--diffusion-model", str(diffusion or prof["diffusionModel"]),
        "--vae", prof["vae"],
        "--llm", prof["textEncoder"],
        "--backend", "mtl0", "--params-backend", "disk",
        "--mmap", "--flash-attn", "on",
        "--model-args", f"qwen_image_2_1_prefix_cache_type={cfg.get('prefixCacheType', 'q8_0')}",
    ]
    tile = cfg.get("vaeTileSize", 768)
    if tile:
        a += ["--vae-tiling", "--vae-tile-size", str(tile), "--vae-tile-overlap", "0.125"]
    return a


def replace(args, flag, value=None, drop=False):
    """`args` with `flag` set to `value` (or removed); values follow their flag."""
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            has_val = i + 1 < len(args) and not args[i + 1].startswith("--")
            i += 2 if has_val else 1
            continue
        out.append(args[i])
        i += 1
    if not drop:
        out += [flag] + ([value] if value is not None else [])
    return out


def sd_variant(name, family, desc, fn=None, engine=KOKON_ENGINE, diffusion=None, needs=None):
    def build():
        a = sdcpp_base(engine, diffusion)
        return fn(a) if fn else a
    return {"name": name, "kind": "sdcpp", "family": family, "desc": desc, "build": build, "needs": needs or [engine / "sd-cli"]}


def mf_variant(name, family, desc, extra=()):
    return {"name": name, "kind": "mflux", "family": family, "desc": desc, "extra": list(extra), "needs": [MFLUX / "bin" / "python", MLX_4BIT / "model_index.json"]}


VARIANTS = [
    sd_variant("kokon", "sd", "Kokon today: engine kokon-1013, Q4_K, Disk placement (the reference)"),
    sd_variant("kokon-resident", "sd", "weights stay on the GPU (24 GB+ Mac flags): the cost of reloading from disk",
               lambda a: replace(replace(a, "--params-backend", drop=True), "--auto-fit", "off")),
    sd_variant("kokon-no-fa", "sd", "flash attention off", lambda a: replace(a, "--flash-attn", "off")),
    sd_variant("kokon-prefix-off", "sd", "text prefix K/V recomputed every step",
               lambda a: replace(a, "--model-args", "qwen_image_2_1_prefix_cache=false")),
    sd_variant("kokon-prefix-f16", "sd", "prefix cache in f16 instead of q8_0",
               lambda a: replace(a, "--model-args", "qwen_image_2_1_prefix_cache_type=f16")),
    sd_variant("kokon-no-vae-tiles", "sd", "VAE decoded in one piece", lambda a: [x for x in a if x != "--vae-tiling"]),
    sd_variant("kokon-easycache", "sd", "EasyCache step reuse (approximate), default options",
               lambda a: a + ["--cache-mode", "easycache"]),
    sd_variant("kokon-easycache-02", "sd", "EasyCache threshold 0.2 (approximate)",
               lambda a: a + ["--cache-mode", "easycache", "--cache-option", "threshold=0.2"]),
    sd_variant("kokon-dbcache", "sd", "DBCache block-level cache (approximate), default options",
               lambda a: a + ["--cache-mode", "dbcache"]),
    sd_variant("kokon-spectrum", "sd", "Spectrum forecasting (approximate), default options",
               lambda a: a + ["--cache-mode", "spectrum"]),
    sd_variant("kokon-q8", "sd-q8", "Q8_0 weights: is Q4_K dequantisation the bottleneck?", diffusion=Q8_GGUF,
               needs=[KOKON_ENGINE / "sd-cli", Q8_GGUF]),
    sd_variant("upstream", "sd", "upstream sd.cpp master-1004 (2026-10-24), same flags", engine=UPSTREAM_ENGINE),
    mf_variant("mflux-4bit", "mlx", "mflux 0.23 (MLX), Osaurus 4-bit bundle, bf16 compute"),
    mf_variant("mflux-4bit-fp16", "mlx", "mflux with --compute-precision float16 (M1/M2 have no fast bf16)",
               ["--compute-precision", "float16"]),
    mf_variant("mflux-4bit-lowram", "mlx", "mflux --low-ram (frees each part after use, like Disk placement)", ["--low-ram"]),
    mf_variant("mflux-4bit-stepcache", "mlx", "mflux step reuse 0.25 (approximate; text-to-image only)",
               ["--step-cache-ratio", "0.25"]),
]
SILICONTROL = Path("/Applications/Siliconed.app/Contents/Helpers/silicontrol")
SILICONED_MODELS = HOME / "Library" / "Application Support" / "Siliconed" / "store"
VARIANTS += [
    # The app renders; silicontrol queues and waits. The install (Light: GGUF Q4_K_M DiT, 8-bit
    # text encoder) and Faster Rendering (Settings, on by default) are chosen in the app.
    {"name": "siliconed", "kind": "siliconed", "family": "siliconed",
     "desc": "Siliconed 0.14 (Swift + Metal + AMX/ANE), the install and Faster Rendering as set in the app",
     "needs": [SILICONTROL, SILICONED_MODELS]},
]
DEFAULT = ["kokon", "kokon-resident", "kokon-no-fa", "kokon-prefix-off", "kokon-easycache", "kokon-dbcache",
           "upstream", "mflux-4bit", "mflux-4bit-fp16", "mflux-4bit-lowram"]


# ----------------------------------------------------------------------------------------- run

def vm_gb():
    """System memory now (GB), from vm_stat. "used" follows Activity Monitor's Memory Used: app
    (anonymous) memory + wired + the compressor's own pages; the file cache is apart, since macOS
    drops it when memory is needed (weights read with mmap or streamed from disk land there)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    g = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1)) * page / 2**30
    anon, wired, comp = g("Anonymous pages"), g("Pages wired down"), g("Pages occupied by compressor")
    return {"used": anon + wired + comp, "anon": anon, "wired": wired, "compressor": comp,
            "file_cache": g("File-backed pages"), "free": g("Pages free")}


class SystemSampler:
    """Samples vm_stat every second while a run lasts: peaks, and the rise over the baseline."""

    def __init__(self):
        import threading
        self.base, self.samples, self.done = vm_gb(), [], threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        while not self.done.wait(1):
            self.samples.append(vm_gb())

    def stop(self):
        self.done.set()
        self.th.join()
        s = self.samples or [self.base]
        peak = lambda k: max(x[k] for x in s)
        r = lambda x: round(x, 2)
        return {"sys_used_base_gb": r(self.base["used"]), "sys_used_peak_gb": r(peak("used")),
                "sys_used_rise_gb": r(peak("used") - self.base["used"]),
                "wired_peak_gb": r(peak("wired")), "compressor_peak_gb": r(peak("compressor")),
                "file_cache_peak_gb": r(peak("file_cache")), "free_min_gb": r(min(x["free"] for x in s))}


def swap_used_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def prepare_ref(src: Path, size, out_dir: Path):
    """The reference at the exact working size (Kokon sends refs pre-sized, mode=none)."""
    dst = out_dir / f"ref-{size[0]}x{size[1]}.png"
    if not dst.exists():
        tmp = out_dir / "ref-tmp.png"
        subprocess.run(["sips", "-s", "format", "png", str(src), "--out", str(tmp)], check=True, capture_output=True)
        # Cover-crop to the target aspect, then resize.
        w, h = (int(re.search(rf"{k}: (\d+)", subprocess.run(["sips", "-g", k, str(tmp)], capture_output=True, text=True).stdout).group(1))
                for k in ("pixelWidth", "pixelHeight"))
        tw, th = size
        if w * th > h * tw:
            cw, ch = round(h * tw / th), h
        else:
            cw, ch = w, round(w * th / tw)
        subprocess.run(["sips", "-c", str(ch), str(cw), str(tmp)], check=True, capture_output=True)
        subprocess.run(["sips", "-z", str(th), str(tw), str(tmp), "--out", str(dst)], check=True, capture_output=True)
        tmp.unlink()
    return dst


def command(v, case, size, ref, out_png):
    w, h = size
    if v["kind"] == "sdcpp":
        a = v["build"]() + [
            "--cfg-scale", "1", "--sampling-method", "euler", "--steps", str(STEPS), "-s", str(SEED),
            "-W", str(w), "-H", str(h), "-p", case["prompt"], "-o", str(out_png),
        ]
        if ref:
            a += ["-r", str(ref), "--image-preprocess", "target=ref,mode=none", "--ref-image-args", "resize_before_vae=false"]
        return a
    if v["kind"] == "siliconed":
        a = [str(SILICONTROL), "add", "--wait", "--out", str(out_png.parent / (out_png.stem + ".siliconed")),
             "--model", "qwen-image-2.1", "--steps", str(STEPS), case["prompt"], str(SEED)]
        # An edit follows image 1's size, so the format is left out (help add).
        return a + (["--ref", str(ref)] if ref else [f"{w}x{h}"])
    exe = "mflux-generate-qwen-2.1-edit" if ref else "mflux-generate-qwen-2.1"
    a = [str(MFLUX / "bin" / exe), "--model", str(MLX_4BIT), "--base-model", "qwen-image-2.1-turbo",
         "--steps", str(STEPS), "--seed", str(SEED), "--width", str(w), "--height", str(h),
         "--prompt", case["prompt"], "--output", str(out_png)] + v["extra"]
    if ref:
        a += ["--image-paths", str(ref)]
    return a


def parse_sdcpp(log: str):
    num = lambda pat: [float(x) for x in re.findall(pat, log)]
    its = num(r"(\d+\.\d+)s/it")
    r = {
        "load_s": sum(num(r"loading tensors completed, taking ([\d.]+)s")),
        "text_s": sum(num(r"get_learned_condition completed, taking ([\d.]+)s")),
        "ref_encode_s": sum(num(r"encode_first_stage completed, taking ([\d.]+)s")) or None,
        "sampling_s": sum(num(r"sampling completed, taking ([\d.]+)s")),
        "decode_s": sum(num(r"decode_first_stage completed, taking ([\d.]+)s")),
        "total_s": (num(r"generate_image completed in ([\d.]+)s") or [None])[-1],
        # The bar's first s/it is the first step alone (graph build + prefix pass). Its later
        # values read low against the sampling total (5.6 vs 6.3 s at 512², M2 Pro), so the
        # steady step is derived from the total instead.
        "first_step_s": its[0] if its else None,
    }
    if r["sampling_s"] and STEPS > 1 and r["first_step_s"]:
        r["steady_step_s"] = r["step_s"] = round((r["sampling_s"] - r["first_step_s"]) / (STEPS - 1), 2)
    return r


def parse_mflux(log: str):
    # tqdm prints the running mean: the first value is step 1 alone, the last the mean of all.
    its = [float(x) for x in re.findall(r"([\d.]+)s/it", log)]
    if len(its) < 2:
        return {"step_s": its[-1] if its else None}
    first, mean = its[0], its[-1]
    steady = round((STEPS * mean - first) / (STEPS - 1), 2)
    return {"first_step_s": first, "steady_step_s": steady, "step_s": steady, "sampling_s": round(STEPS * mean, 1)}


def parse_time_l(err: str):
    m = re.search(r"(\d+)\s+peak memory footprint", err)
    r = re.search(r"(\d+)\s+maximum resident set size", err)
    return {"peak_footprint_gb": round(int(m.group(1)) / 2**30, 2) if m else None,
            "max_rss_gb": round(int(r.group(1)) / 2**30, 2) if r else None}


def parse_siliconed(log: str):
    """Step times from the arrival times bench adds to silicontrol's "step" lines."""
    t = [float(m) for m in re.findall(r"^\[(\d+\.\d+)\] .*\"type\":\s*\"step\"", log, re.M)]
    t0 = re.search(r"^\[(\d+\.\d+)\] .*\"type\":\s*\"start\"", log, re.M)
    r = {}
    # The app's own render time (queue to PNG), without its once-per-launch Mac calibration.
    s = re.findall(r"\"seconds\":\s*([\d.]+)", log)
    if s:
        r["total_s"] = float(s[-1])
    if len(t) >= 2:
        steps = [b - a for a, b in zip(t, t[1:])]
        r["steady_step_s"] = r["step_s"] = round(statistics.median(steps), 2)
        if t0:
            r["first_step_s"] = round(t[0] - float(t0.group(1)), 2)
    return r


def run_siliconed(cmd, out_dir: Path, png: Path):
    """silicontrol talks to the running app: time the app's footprint, not the client's."""
    import threading
    pid = subprocess.run(["pgrep", "-x", "Siliconed"], capture_output=True, text=True).stdout.split()
    vals, done = [], threading.Event()

    def sample():  # `footprint --sample` prints no per-sample total; poll single reads instead
        while pid and not done.wait(2):
            out = subprocess.run(["footprint", "-p", pid[0], "-f", "bytes"], capture_output=True, text=True).stdout
            vals.extend(int(x) for x in re.findall(r"Footprint: (\d+) B", out))
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t0 = time.monotonic()
    lines = []
    # silicontrol buffers its JSON lines when stdout is a pipe; `script` gives it a terminal so
    # each step line arrives when it happens.
    p = subprocess.Popen(["script", "-q", "/dev/null"] + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, text=True, cwd=out_dir)
    for line in p.stdout:
        lines.append(f"[{time.monotonic() - t0:.2f}] {line.rstrip()}")
    p.wait()
    wall = time.monotonic() - t0
    done.set()
    th.join()
    peak = round(max(vals) / 2**30, 2) if vals else None
    got = sorted((png.parent / (png.stem + ".siliconed")).glob("*.png"))
    if got:
        shutil.move(str(got[-1]), png)
    return p.returncode, "\n".join(lines), wall, peak


def run_one(v, case_name, size, ref, out_dir: Path, rep: int):
    case = CASES[case_name]
    tag = f"{v['name']}__{case_name}__{rep}"
    png = out_dir / f"{tag}.png"
    cmd = command(v, case, size, ref, png)
    swap0 = swap_used_mb()
    sampler = SystemSampler()
    if v["kind"] == "siliconed":
        code, log, wall, peak = run_siliconed(cmd, out_dir, png)
        p = subprocess.CompletedProcess(cmd, code, log, "")
    else:
        t0 = time.monotonic()
        p = subprocess.run(["/usr/bin/time", "-l"] + cmd, capture_output=True, text=True, cwd=out_dir,
                           env={**os.environ, "PYTHONNOUSERSITE": "1"})
        wall = time.monotonic() - t0
    mem = sampler.stop()
    log = p.stdout + "\n" + p.stderr
    (out_dir / f"{tag}.log").write_text(" ".join(cmd) + "\n\n" + log)
    r = {"variant": v["name"], "family": v["family"], "case": case_name, "rep": rep, "size": f"{size[0]}x{size[1]}",
         "ok": p.returncode == 0 and png.exists(), "wall_s": round(wall, 1),
         "swap_delta_mb": round(swap_used_mb() - swap0), "image": png.name if png.exists() else None, **mem}
    if v["kind"] == "siliconed":
        r.update(parse_siliconed(log), peak_footprint_gb=peak)
    else:
        r.update(parse_time_l(p.stderr))
        r.update(parse_sdcpp(log) if v["kind"] == "sdcpp" else parse_mflux(log))
    if r.get("sampling_s") and v["kind"] == "mflux":
        r["fixed_s"] = round(wall - r["sampling_s"], 1)  # load, text encoding, decode
    if not r["ok"]:
        r["error"] = log.strip().splitlines()[-5:]
    return r


# ----------------------------------------------------------------------------------------- report

PSNR_SCRIPT = r"""
import json, sys, numpy as np
from PIL import Image
pairs = json.loads(sys.argv[1]); out = {}
for a, b in pairs:
    x = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    y = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    if x.shape != y.shape: continue
    mse = ((x - y) ** 2).mean()
    out[b] = 99.0 if mse == 0 else round(10 * np.log10(255 ** 2 / mse), 2)
print(json.dumps(out))
"""


def add_psnr(rows, out_dir: Path):
    """PSNR of each image against the first ok variant of its family, same case and rep."""
    py = MFLUX / "bin" / "python"
    if not py.exists():
        return
    first, pairs = {}, []
    for r in rows:
        if not r["ok"]:
            continue
        key = (r["family"], r["case"], r["rep"])
        if key not in first:
            first[key] = r
        elif first[key]["image"] != r["image"]:
            pairs.append((str(out_dir / first[key]["image"]), str(out_dir / r["image"])))
    if not pairs:
        return
    p = subprocess.run([str(py), "-I", "-c", PSNR_SCRIPT, json.dumps(pairs)], capture_output=True, text=True)
    if p.returncode != 0:
        print("PSNR skipped:", p.stderr.strip()[-300:])
        return
    scores = json.loads(p.stdout)
    for r in rows:
        if r.get("image") and str(out_dir / r["image"]) in scores:
            r["psnr_vs_family_ref"] = scores[str(out_dir / r["image"])]


COLS = ["variant", "case", "size", "wall_s", "total_s", "load_s", "text_s", "ref_encode_s", "sampling_s",
        "first_step_s", "steady_step_s", "step_s", "fixed_s", "decode_s", "peak_footprint_gb", "swap_delta_mb",
        "sys_used_base_gb", "sys_used_peak_gb", "sys_used_rise_gb", "wired_peak_gb", "compressor_peak_gb",
        "file_cache_peak_gb", "free_min_gb", "psnr_vs_family_ref"]


def summarize(rows):
    """Median per (variant, case) over repeats."""
    groups = {}
    for r in rows:
        if r["ok"]:
            groups.setdefault((r["variant"], r["case"]), []).append(r)
    out = []
    for (v, c), rs in groups.items():
        s = {"variant": v, "case": c, "size": rs[0]["size"], "n": len(rs)}
        for k in COLS[3:]:
            vals = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            s[k] = round(statistics.median(vals), 2) if vals else None
        out.append(s)
    return out


def write_report(rows, summary, variants, out_dir: Path, meta):
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n"] + COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "runs": rows, "summary": summary}, indent=2))

    desc = {v["name"]: v["desc"] for v in variants}
    base = {s["case"]: s["wall_s"] for s in summary if s["variant"] == "kokon"}
    fmt = lambda x: "" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
    parts = [f"<!doctype html><meta charset=utf-8><title>Mac engine bench</title><style>"
             "body{font:14px system-ui;margin:24px;background:#fff;color:#111}table{border-collapse:collapse}"
             "td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}td:first-child,th:first-child{text-align:left}"
             ".g{display:flex;flex-wrap:wrap;gap:12px}.g figure{margin:0;width:320px}.g img{width:320px}"
             "figcaption{font-size:12px}</style>",
             f"<h1>Qwen-Image 2.1 Turbo on {html.escape(meta['chip'])}, {meta['ram_gb']} GB</h1>",
             f"<p>{html.escape(meta['started'])} · seed {SEED} · {STEPS} steps · medians of {meta['repeat']} run(s)</p>"]
    for case in CASES:
        rs = [s for s in summary if s["case"] == case]
        if not rs:
            continue
        parts.append(f"<h2>{case}</h2><table><tr><th>variant</th><th>speed-up</th>" +
                     "".join(f"<th>{c}</th>" for c in COLS[3:]) + "<th>what</th></tr>")
        for s in sorted(rs, key=lambda s: s["wall_s"] or 1e9):
            sp = f"{base[case] / s['wall_s']:.2f}×" if base.get(case) and s["wall_s"] else ""
            parts.append(f"<tr><td>{s['variant']}</td><td>{sp}</td>" + "".join(f"<td>{fmt(s.get(c))}</td>" for c in COLS[3:]) +
                         f"<td style='text-align:left'>{html.escape(desc.get(s['variant'], ''))}</td></tr>")
        parts.append("</table><div class=g>")
        for r in rows:
            if r["case"] == case and r["ok"] and r["rep"] == 0:
                parts.append(f"<figure><img src='{r['image']}' loading=lazy><figcaption>{r['variant']} · {r['wall_s']} s"
                             f"{' · PSNR ' + str(r['psnr_vs_family_ref']) if r.get('psnr_vs_family_ref') else ''}</figcaption></figure>")
        parts.append("</div>")
    failed = [r for r in rows if not r["ok"]]
    if failed:
        parts.append("<h2>Failed</h2><ul>" + "".join(
            f"<li>{r['variant']} / {r['case']}: <code>{html.escape(' | '.join(r.get('error', [])))}</code></li>" for r in failed) + "</ul>")
    (out_dir / "report.html").write_text("\n".join(parts))


# ----------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated variants (default: %s)" % ",".join(DEFAULT))
    ap.add_argument("--all", action="store_true", help="every variant whose files exist")
    ap.add_argument("--cases", default="t2i,edit", help="t2i, edit")
    ap.add_argument("--size", type=int, help="override: a square of this side for t2i, same area for edit")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=20, help="seconds between runs (thermals, Metal release)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args()

    if args.list:
        for v in VARIANTS:
            have = all(Path(p).exists() for p in v["needs"])
            print(f"{'*' if v['name'] in DEFAULT else ' '} {v['name']:<22} {'' if have else '(missing files) '}{v['desc']}")
        print("\ncases:", ", ".join(f"{k} {c['size'][0]}x{c['size'][1]}" for k, c in CASES.items()))
        return

    names = [v["name"] for v in VARIANTS] if args.all else (args.only.split(",") if args.only else DEFAULT)
    unknown = set(names) - {v["name"] for v in VARIANTS}
    if unknown:
        sys.exit(f"unknown variants: {', '.join(sorted(unknown))} (see --list)")
    # In the order given: put the engines that swap (mflux on 16 GB) last, so the pages they
    # push out do not weigh on the next runs.
    by_name = {v["name"]: v for v in VARIANTS}
    variants = []
    for v in (by_name[n] for n in names):
        missing = [str(p) for p in v["needs"] if not Path(p).exists()]
        if missing:
            print(f"skip {v['name']}: missing {missing[0]} (run setup.sh)")
            continue
        variants.append(v)
    cases = args.cases.split(",")

    if not args.dry_run and subprocess.run(["pgrep", "-x", "sd-server"], capture_output=True).returncode == 0:
        sys.exit("sd-server is running: quit Kokon first (it holds Metal memory and the GPU).")

    started = dt.datetime.now()
    out_dir = BENCH / "results" / started.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    ram = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30)
    meta = {"started": started.isoformat(timespec="seconds"), "chip": chip, "ram_gb": ram, "repeat": args.repeat,
            "macos": subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True).stdout.strip(),
            "seed": SEED, "steps": STEPS}

    rows = []
    for case_name in cases:
        case = CASES[case_name]
        size = case["size"]
        if args.size:
            area = args.size * args.size
            ar = size[0] / size[1]
            size = (round((area * ar) ** 0.5 / 32) * 32, round((area / ar) ** 0.5 / 32) * 32)
        ref = prepare_ref(case["ref"], size, out_dir) if case["ref"] else None
        for rep in range(args.repeat):
            for v in variants:
                if ref and "--step-cache-ratio" in v.get("extra", []):
                    continue  # the edit command has no --step-cache-ratio
                if args.dry_run:
                    print(" ".join(f"'{x}'" if " " in x else x for x in command(v, case, size, ref, out_dir / "x.png")), "\n")
                    continue
                print(f"[{case_name} {size[0]}x{size[1]} #{rep}] {v['name']} ...", end=" ", flush=True)
                r = run_one(v, case_name, size, ref, out_dir, rep)
                rows.append(r)
                print(f"{r['wall_s']} s" + (f", {r['step_s']} s/step" if r.get("step_s") else "") +
                      (f", peak {r['peak_footprint_gb']} GB" if r.get("peak_footprint_gb") else "") +
                      ("" if r["ok"] else "  FAILED (see log)"))
                (out_dir / "results.json").write_text(json.dumps({"meta": meta, "runs": rows}, indent=2))
                time.sleep(args.cooldown)
    if args.dry_run:
        shutil.rmtree(out_dir)
        return
    add_psnr(rows, out_dir)
    summary = summarize(rows)
    write_report(rows, summary, variants, out_dir, meta)
    print(f"\n{out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
