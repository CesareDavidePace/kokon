"""Same-seed A/B of Kokon's quality levels and few-step LoRAs (text-to-image and edits).

Close Kokon first: this starts its own sd-server with the exact launch flags of Kokon's last
run (from the service log) plus `--lora-model-dir`, so every variant runs on the same engine
settings. Each case uses one fixed seed for every variant. Variants at the same size start
from the same noise, so their compositions are directly comparable (Fast = the full model at
1 MP, the reference for the 1 MP few-step LoRAs).

    python -I scripts/dev/compare-fast-modes.py                # all variants, 10 t2i + 6 edits
    python -I scripts/dev/compare-fast-modes.py --only fast,viggle6,official8 --cases 3

Writes <out>/report.html (side-by-side grid + timing table), results.json and the images.
Missing LoRAs and checkpoints (about 7 GB for all) are downloaded into Kokon's model folders
and checked against their SHA-256. official8 is another diffusion model, so it runs on a
second server after the first one.
"""

import argparse
import base64
import hashlib
import html
import io
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "KokonData"
LOG = LOCAL / "logs" / "service.log"
LORA_DIR = LOCAL / "sdcpp" / "models" / "loras"
PORT = 39160

# ---------------------------------------------------------------------------------------
# Few-step schedules
# ---------------------------------------------------------------------------------------


def dynamic_shift(nodes, w, h, base_shift=0.5, max_shift=0.9, base_seq=256, max_seq=8192):
    """Timesteps under diffusers' dynamic exponential shift (mu linear in the token count,
    extrapolated past max_seq), then 0. Same as `planner::turbo_sigmas` in the app."""
    tokens = (w // 16) * (h // 16)
    e = math.exp(base_shift + (max_shift - base_shift) * (tokens - base_seq) / (max_seq - base_seq))
    return [e / (e + 1 / t - 1) for t in nodes] + [0.0]


def tez_sigmas(w, h):
    # Sarvam Tez: {1, s, 0} with s = index 20 of the base 40-step schedule (stretch 0.02);
    # the model card's value at 1024², the only size the LoRA variants run at here.
    assert (w, h) == (1024, 1024), "tez sigmas are only known for 1024x1024"
    return [1.0, 0.65667, 0.0]


HF = "https://huggingface.co"
MODELS_DIR = LOCAL / "sdcpp" / "models" / "qwen21"

# name -> settings. `area` = working pixels; `lora` = file in LORA_DIR, `model` = another
# diffusion model in MODELS_DIR (run on a separate server), each with url/size/sha256 to
# download it; `sigmas(w, h)` = exact schedule (None: the server's default for `steps`);
# `t2i_only` = skipped for edits.
VARIANTS = {
    "balanced": {"label": "Balanced (1.5K, 20 steps)", "area": 1536 * 1536, "steps": 20},
    "fast": {"label": "Fast (1 MP, 20 steps)", "area": 1024 * 1024, "steps": 20},
    "viggle6": {
        "label": "Viggle turbo v0.3 (6 steps)",
        "area": 1024 * 1024,
        "lora": "viggle-turbo-v0.3-r128.safetensors",
        "url": "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors",
        "size": 679_604_800,
        "sha256": "0c98591700346f9777051d4e6fa29aa94519abec0d85b1f1f672a2a3db8c94b3",
        "sigmas": lambda w, h: dynamic_shift([1.0, 0.9375, 0.875, 0.75, 0.5, 0.25], w, h),
    },
    "turbo8": {
        "label": "Turbo8 LoRA (8 steps)",
        "area": 1024 * 1024,
        "lora": "turbo8_lora_step2500.safetensors",
        "url": f"{HF}/chriswritescode/Turbo8-LoRA-Qwen-Image-2.1/resolve/main/turbo8_lora_step2500.safetensors",
        "size": 1_359_148_744,
        "sha256": "a0b7a7634ff2f82ccbe209e16b0ba19b25e2d9e18624d89edc490503b6a75d39",
        "sigmas": lambda w, h: dynamic_shift([1 - i / 8 for i in range(8)], w, h),
    },
    "pruna8": {
        "label": "Pruna 8-step LoRA v0.1",
        "area": 1024 * 1024,
        "lora": "p_qwen_image_2.1_8step_v0.1.safetensors",
        "url": f"{HF}/PrunaAI/Pruna-Qwen-Image-2.1/resolve/main/p_qwen_image_2.1_8step_v0.1.safetensors",
        "size": 335_606_104,
        "sha256": "f0865d68b02511a3a0ed232d9d1aa99cac3a94166f574b38bcbab1a7a297bb15",
        # r64 with alpha 128 only in the file metadata, which sd.cpp ignores (scale 1); 2.0
        # matches diffusers' strength 1.
        "multiplier": 2.0,
        "sigmas": lambda w, h: [1, 14 / 15, 6 / 7, 10 / 13, 2 / 3, 6 / 11, 0.4, 2 / 9, 0.0],
    },
    "tez2": {
        "label": "Sarvam Tez LoRA (2 steps, t2i only)",
        "area": 1024 * 1024,
        "t2i_only": True,
        "lora": "Qwen-Image-2.1-sarvam-tez-2step-lora-r256.safetensors",
        "url": f"{HF}/sarvamaigc/Qwen-Image-2.1-Sarvam-Tez/resolve/main/Qwen-Image-2.1-sarvam-tez-2step-lora-r256.safetensors",
        "size": 1_342_238_184,
        "sha256": "1f964e6e9621c973a13868d277f3747a7c8cb0682bc432835679b4063392682a",
        "sigmas": tez_sigmas,
    },
    "official8": {
        "label": "Official Qwen-Image-2.1-Turbo Q4_K (8 steps)",
        "area": 1024 * 1024,
        "model": "qwen-image-2.1-turbo-Q4_K.gguf",
        "url": f"{HF}/DogukanUrker/Qwen-Image-2.1-Turbo-GGUF/resolve/main/qwen-image-2.1-turbo-Q4_K.gguf",
        "size": 4_050_916_384,
        "sha256": "f8d5611b495848b632f8ae14dc7da6525fe2796eb80998f8dee5e04d0a2f69eb",
        # Fixed schedule from the checkpoint's model_index.json (no dynamic shift).
        "sigmas": lambda w, h: [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0],
    },
}
# The same official Turbo at higher precision (quantization A/B; each is its own server run).
for _q, _size, _sha in [
    ("Q6_K", 5_876_562_976, "a61e8bfe4318a2df2a2321bfaa23796eb157d184fa53200539507cea5555658d"),
    ("Q8_0", 7_591_558_176, "228d7931adade6cdd9ce2c8fcba0fc2a54552679efaef2dd34b83413750c8c76"),
]:
    VARIANTS[f"official8_{_q.lower()}"] = {
        **VARIANTS["official8"],
        "label": f"Official Turbo {_q} (8 steps)",
        "model": f"qwen-image-2.1-turbo-{_q}.gguf",
        "url": f"{HF}/DogukanUrker/Qwen-Image-2.1-Turbo-GGUF/resolve/main/qwen-image-2.1-turbo-{_q}.gguf",
        "size": _size,
        "sha256": _sha,
    }

# ---------------------------------------------------------------------------------------
# Test cases: 10 text-to-image prompts, then edits of the Fast results (so every variant edits
# the same input) and one two-image combine.
# ---------------------------------------------------------------------------------------

T2I = [
    ("portrait", "Close-up portrait of an elderly fisherman with a weathered face and a grey beard, soft window light, shallow depth of field, photorealistic"),
    ("poster", 'A vintage travel poster for Lisbon with the title "LISBOA" in large bold letters and the subtitle "City of Seven Hills" below, yellow tram, sunset colours'),
    ("landscape", "Misty pine forest in the mountains at sunrise, a lake reflecting the peaks, golden light, ultra-detailed landscape photograph"),
    ("product", "Studio product photo of a matte black wireless headphone on a marble pedestal, soft shadows, minimal beige background"),
    ("animal", "A red fox sitting in fresh snow, looking at the camera, fine fur detail, telephoto wildlife photograph"),
    ("architecture", "Interior of a modern library with curved wooden shelves, skylights, people reading, wide-angle architectural photo"),
    ("illustration", "Cozy watercolor illustration of a small bakery on a rainy street at night, warm light in the windows, a cat on the doorstep"),
    ("food", "Overhead photo of a Neapolitan pizza margherita on a wooden table, basil leaves, flour dust, natural light"),
    ("signage", 'A small café chalkboard menu on the sidewalk that reads "TODAY: Espresso 1.20, Cappuccino 1.80, Cornetto 1.50" in neat chalk handwriting'),
    ("hands", "Two hands of a potter shaping a clay bowl on a spinning wheel, workshop background, detailed fingers, documentary photo"),
]

EDITS = [
    ("portrait", "Change the background to a busy harbour with boats, keep the man identical"),
    ("poster", 'Change the title text to "PORTO" and make the tram blue'),
    ("landscape", "Make it a winter scene with snow on the trees and a frozen lake"),
    ("product", "Change the headphone colour to bright red and add a small plant next to the pedestal"),
    ("animal", "Turn the fox into an oil painting in the style of Van Gogh"),
    ("food", "Remove the basil leaves and add slices of salami"),
]

COMBINE = ("Put the fox from image 1 in front of the bakery from image 2, matching the watercolor style", ["animal", "illustration"])


def dims(aspect, area, align=32):
    w = math.sqrt(area * aspect)
    h = w / aspect
    return max(align, round(w / align) * align), max(align, round(h / align) * align)


# ---------------------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------------------


def kokon_server_args(model=None):
    """sd-server command of Kokon's last launch, from the service log."""
    if not LOG.is_file():
        sys.exit(f"No Kokon log at {LOG}: start Kokon once and generate an image first.")
    lines = [l for l in LOG.read_text(encoding="utf-8", errors="replace").splitlines() if "qls_core::process: starting " in l and "sd-server" in l]
    if not lines:
        sys.exit("The Kokon log has no sd-server launch line; generate one image in Kokon first.")
    cmd = lines[-1].split("qls_core::process: starting ", 1)[1].strip()
    exe, _, rest = cmd.partition(" --")
    args = []
    # Flags start with " --"; a value may contain spaces (paths).
    for part in ("--" + rest).split(" --"):
        part = part.strip()
        if not part:
            continue
        part = part if part.startswith("--") else "--" + part
        name, _, value = part.partition(" ")
        args.append(name)
        if value:
            args.append(value)
    # Our own port, and the LoRA folder.
    i = args.index("--listen-port")
    args[i + 1] = str(PORT)
    if "--lora-model-dir" not in args:
        args += ["--lora-model-dir", str(LORA_DIR)]
    if model:
        args[args.index("--diffusion-model") + 1] = str(MODELS_DIR / model)
    return exe.strip(), args


def http(method, path, body=None, timeout=60):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method=method, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def start_server(out, model=None):
    exe, args = kokon_server_args(model)
    print("starting", exe, " ".join(args))
    log = open(out / f"sd-server-{model or 'kokon'}.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([exe, *args], cwd=str(Path(exe).parent), stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if proc.poll() is not None:
            sys.exit(f"sd-server exited (code {proc.returncode}); see {log.name}. Is Kokon still running?")
        try:
            caps = http("GET", "/sdcpp/v1/capabilities", timeout=5)
            return proc, caps
        except Exception:
            time.sleep(1)
    proc.kill()
    sys.exit("sd-server did not come up in 10 minutes.")


def generate(body):
    """Submit, wait, return (PIL image, seconds from submit to result)."""
    t0 = time.perf_counter()
    job = http("POST", "/sdcpp/v1/img_gen", body)
    while True:
        j = http("GET", f"/sdcpp/v1/jobs/{job['id']}")
        if j["status"] == "completed":
            img = Image.open(io.BytesIO(base64.b64decode(j["result"]["images"][0]["b64_json"]))).convert("RGB")
            return img, time.perf_counter() - t0
        if j["status"] in ("failed", "cancelled"):
            raise RuntimeError(json.dumps(j.get("error")))
        time.sleep(0.5)


def b64png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def request(v, prompt, seed, w, h, refs=()):
    sp = {"sample_method": "euler", "sample_steps": v.get("steps", 20), "guidance": {"txt_cfg": 1.0}}
    if v.get("sigmas"):
        sp["custom_sigmas"] = v["sigmas"](w, h)
        sp["sample_steps"] = len(sp["custom_sigmas"]) - 1
    body = {
        "prompt": prompt, "negative_prompt": "", "width": w, "height": h, "seed": seed, "batch_count": 1,
        "sample_params": sp, "output_format": "png", "embed_image_metadata": False,
    }
    if refs:
        # References at the working size, as Kokon sends them (no server-side transforms).
        body["ref_images"] = [b64png(r.resize(dims(r.width / r.height, w * h), Image.LANCZOS)) for r in refs]
        body["image_preprocess"] = ["target=ref,mode=none"]
    if v.get("lora"):
        body["lora"] = [{"path": v["lora"], "multiplier": v.get("multiplier", 1.0)}]
    return body


# ---------------------------------------------------------------------------------------
# LoRA files
# ---------------------------------------------------------------------------------------


def ensure_files(v):
    if v.get("lora"):
        dest = LORA_DIR / v["lora"]
    elif v.get("model"):
        dest = MODELS_DIR / v["model"]
    else:
        return
    if dest.is_file() and dest.stat().st_size == v["size"]:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The copy from earlier sd.cpp tests, if it is there.
    old = Path(r"C:\AI\sdcpp\models\loras") / dest.name
    if old.is_file() and old.stat().st_size == v["size"]:
        print(f"copying {old}")
        shutil.copyfile(old, dest)
        return
    print(f"downloading {v['url']} ({v['size'] / 1e6:.0f} MB)")
    tmp = dest.with_suffix(".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(v["url"]) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != v["sha256"]:
        tmp.unlink()
        sys.exit(f"checksum mismatch for {dest.name}")
    tmp.replace(dest)


# ---------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------


def report(out, names, results):
    by_case = {}
    for r in results:
        by_case.setdefault((r["kind"], r["case"]), {})[r["variant"]] = r
    rows = []
    for (kind, case), cells in by_case.items():
        prompt = next(iter(cells.values()))["prompt"]
        tds = []
        for n in names:
            r = cells.get(n)
            if not r:
                tds.append("<td>–</td>")
            elif r.get("error"):
                tds.append(f"<td class=err>{html.escape(r['error'][:200])}</td>")
            else:
                tds.append(f'<td><a href="{r["file"]}"><img src="{r["file"]}" loading=lazy></a><div>{r["seconds"]:.1f} s · {r["width"]}×{r["height"]}</div></td>')
        rows.append(f"<tr><th><b>{kind}: {case}</b><p>{html.escape(prompt)}</p></th>{''.join(tds)}</tr>")
    summary = []
    for n in names:
        for kind in ("t2i", "edit"):
            t = [r["seconds"] for r in results if r["variant"] == n and r["kind"] == kind and not r.get("error")]
            if t:
                summary.append(f"<tr><td>{html.escape(VARIANTS[n]['label'])}</td><td>{kind}</td><td>{len(t)}</td><td>{statistics.median(t):.1f} s</td><td>{statistics.mean(t):.1f} s</td></tr>")
    head = "".join(f"<th>{html.escape(VARIANTS[n]['label'])}</th>" for n in names)
    page = f"""<!doctype html><meta charset=utf-8><title>Kokon fast modes</title>
<style>body{{font:14px system-ui;margin:16px;background:#111;color:#ddd}}table{{border-collapse:collapse}}
td,th{{border:1px solid #333;padding:6px;vertical-align:top;text-align:left}}img{{width:300px;display:block}}
th p{{font-weight:normal;max-width:220px;color:#aaa}}.err{{color:#f77;max-width:300px}}</style>
<h1>Same-seed comparison</h1><p>Time = submit to result (encode + sampling + decode), one job at a time; the first job also loads the models.</p>
<table><tr><th>Variant</th><th>Mode</th><th>n</th><th>Median</th><th>Mean</th></tr>{''.join(summary)}</table><br>
<table><tr><th>Case</th>{head}</tr>{''.join(rows)}</table>"""
    (out / "report.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "Desktop" / "kokon-fast-compare"))
    ap.add_argument("--only", help="comma-separated variants (default: all)")
    ap.add_argument("--cases", type=int, default=len(T2I), help="text-to-image cases to run (edits use the first ones)")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(VARIANTS)
    for n in names:
        if n not in VARIANTS:
            sys.exit(f"unknown variant {n}; known: {', '.join(VARIANTS)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    for n in names:
        ensure_files(VARIANTS[n])
    results = []

    def run(kind, case, prompt, seed, n, aspect=1.0, refs=()):
        v = VARIANTS[n]
        w, h = dims(aspect, v["area"])
        name = f"{kind}-{case}-{n}.png"
        try:
            img, secs = generate(request(v, prompt, seed, w, h, refs))
            img.save(out / name)
            r = {"file": name, "seconds": round(secs, 2), "width": w, "height": h}
            print(f"{kind:4} {case:12} {n:10} {secs:6.1f} s")
        except Exception as e:
            r = {"error": str(e)}
            print(f"{kind:4} {case:12} {n:10} FAILED {e}")
        results.append({"kind": kind, "case": case, "variant": n, "prompt": prompt, "seed": seed, **r})
        (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        report(out, names, results)

    cases = T2I[: a.cases]
    # One server per diffusion model: Kokon's first (it also makes the edit inputs), then each
    # other checkpoint.
    for model in [None] + sorted({VARIANTS[n]["model"] for n in names if VARIANTS[n].get("model")}):
        members = [n for n in names if VARIANTS[n].get("model") == model]
        proc, caps = start_server(out, model)
        try:
            known = {l.get("path") for l in caps.get("loras", [])}
            for n in members:
                if VARIANTS[n].get("lora") and VARIANTS[n]["lora"] not in known:
                    print(f"warning: the server does not list {VARIANTS[n]['lora']} (loras: {sorted(known)})")
            # Warm-up so the first measured job does not pay for loading the models.
            generate(request(VARIANTS[members[0]] if members else VARIANTS["fast"], "warm-up", 1, 1024, 1024))
            for i, (case, prompt) in enumerate(cases):
                for n in members:
                    run("t2i", case, prompt, a.seed + i, n)
            # Edits start from the Fast result (the full model at 1 MP), the same input for all.
            base = {}
            for i, (case, prompt) in enumerate(cases):
                p = out / f"t2i-{case}-fast.png"
                if not p.is_file() and model is None:
                    img, _ = generate(request(VARIANTS["fast"], prompt, a.seed + i, 1024, 1024))
                    img.save(p)
                if p.is_file():
                    base[case] = Image.open(p).convert("RGB")
            editors = [n for n in members if not VARIANTS[n].get("t2i_only")]
            for i, (case, prompt) in enumerate(EDITS):
                if case in base:
                    for n in editors:
                        run("edit", case, prompt, a.seed + 100 + i, n, refs=[base[case]])
            prompt, pair = COMBINE
            if all(c in base for c in pair):
                for n in editors:
                    run("edit", "combine", prompt, a.seed + 200, n, refs=[base[c] for c in pair])
        finally:
            proc.terminate()
            proc.wait(timeout=60)
    print(f"\nReport: {out / 'report.html'}")


if __name__ == "__main__":
    main()
