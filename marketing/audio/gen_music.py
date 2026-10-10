"""Generate background-music takes for the launch video with a local ACE-Step 1.5 API server.

Start the server first (D:/AI/ACE-Step-1.5):  uv run acestep-api   (listens on :8001)
Then:  python gen_music.py [style ...]    -> takes/<style>-<n>.wav

Every take is instrumental and a little longer than the video (~85 s), so it can fade out.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8001"
OUT = Path(__file__).parent / "takes"
DURATION = 92
TAKES = 4

STYLES = {
    "keynote": (
        "minimal modern electronic, instrumental, warm analog synth pads, soft plucked arpeggio, "
        "gentle punchy kick enters after the intro, builds gradually, clean and optimistic, "
        "tech product launch film, polished mix",
        104,
        "D Major",
    ),
    "piano": (
        "cinematic instrumental, felt piano motif, soft strings, light electronic pulse, airy pads, "
        "slowly building, hopeful and calm, clean modern production",
        96,
        "C Major",
    ),
    "house": (
        "deep melodic house, instrumental, warm sub bass, crisp hi-hats, dreamy chords, "
        "smooth and confident, steady groove, modern clean mix",
        118,
        "A Minor",
    ),
}


def call(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    if resp.get("code") != 200:
        raise RuntimeError(f"{path}: {resp.get('error')}")
    return resp["data"]


def generate(style: str) -> None:
    prompt, bpm, key = STYLES[style]
    task = call("/release_task", {
        "prompt": prompt,
        "lyrics": "[Instrumental]",
        "instrumental": True,
        "thinking": True,
        "bpm": bpm,
        "key_scale": key,
        "time_signature": "4",
        "audio_duration": DURATION,
        "batch_size": TAKES,
        "inference_steps": 8,
        "audio_format": "wav",
    })
    task_id = task["task_id"]
    t0 = time.time()
    while True:
        time.sleep(3)
        item = call("/query_result", {"task_id_list": [task_id]})[0]
        if item["status"] == 1:
            break
        if item["status"] == 2:
            raise RuntimeError(f"{style}: generation failed: {item}")
    results = json.loads(item["result"]) if isinstance(item["result"], str) else item["result"]
    OUT.mkdir(exist_ok=True)
    for n, res in enumerate(results, 1):
        dest = OUT / f"{style}-{n}.wav"
        urllib.request.urlretrieve(API + res["file"], dest)
        print(f"{dest.name}  seed {res.get('seed_value', '?')}  {res.get('metas', {})}")
    print(f"{style}: {len(results)} takes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    for s in sys.argv[1:] or list(STYLES):
        generate(s)
