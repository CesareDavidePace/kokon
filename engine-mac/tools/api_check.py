#!/usr/bin/env python3
"""Text to image through a running `kokon-engine-mac serve` (or sd-server) as qls-core sends it: capabilities,
img_gen, polling jobs/{id}, optionally a cancel. Prints the time of each image and writes the PNGs.

    python3 -I tools/api_check.py --port 1234 --images 2 --out /tmp/api [--size 1024x1024] [--cancel-after 10]
"""
import argparse, base64, json, os, time, urllib.request, urllib.error

TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]


def call(port, method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1234)
    p.add_argument("--images", type=int, default=2)
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--prompt", default="a red fox in the snow, morning light, a wooden sign that says KOKON")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=".")
    p.add_argument("--cancel-after", type=float, default=0, help="cancel the last image after this many seconds")
    a = p.parse_args()
    width, height = (int(v) for v in a.size.split("x"))
    status, caps = call(a.port, "GET", "/sdcpp/v1/capabilities", timeout=10)
    print(f"capabilities {status}: engine {caps.get('engine', 'sd-server')}, model {caps.get('model', {}).get('name')}")
    os.makedirs(a.out, exist_ok=True)
    for i in range(a.images):
        body = {"prompt": a.prompt, "negative_prompt": "", "width": width, "height": height, "seed": a.seed + i,
                "batch_count": 1, "sample_params": {"sample_method": "euler", "sample_steps": 8,
                                                    "custom_sigmas": TURBO_SIGMAS, "guidance": {"txt_cfg": 1.0}},
                "output_format": "png", "embed_image_metadata": False}
        started = time.time()
        status, accepted = call(a.port, "POST", "/sdcpp/v1/img_gen", body)
        if status != 202:
            print(f"img_gen {status}: {accepted}")
            return 1
        cancel = a.cancel_after > 0 and i == a.images - 1
        while True:
            time.sleep(0.25)
            if cancel and time.time() - started > a.cancel_after:
                t = time.time()
                status, job = call(a.port, "POST", f"/sdcpp/v1/jobs/{accepted['id']}/cancel", timeout=10)
                print(f"cancel {status} after {t - started:.1f} s: {job.get('status')} (answered in {time.time() - t:.2f} s)")
                cancel = False
            status, job = call(a.port, "GET", f"/sdcpp/v1/jobs/{accepted['id']}", timeout=60)
            if job.get("status") in ("completed", "failed", "cancelled"):
                break
        seconds = time.time() - started
        if job["status"] != "completed":
            print(f"image {i + 1}: {job['status']} after {seconds:.1f} s {job.get('error', '')}")
            continue
        path = os.path.join(a.out, f"api-{i + 1}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(job["result"]["images"][0]["b64_json"]))
        print(f"image {i + 1}: {seconds:.1f} s through the API -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
