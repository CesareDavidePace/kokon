"""Generate background-music takes for the launch video with a local ACE-Step 1.5 API server.

Start the server first (D:/AI/ACE-Step-1.5):  uv run acestep-api   (listens on :8001)
Then:  python gen_music.py [style ...]    -> takes/<style>-<n>.wav
       python gen_music.py update         -> takes/update-<stamp>-<n>.wav (the "Kokon 0.10" video)

Every take is instrumental and a little longer than its video (~85 s, or ~44 s for "update"), so
it can fade out. Takes of the launch styles keep their old names; "update" takes get a time stamp,
so no take is ever written over (house-4 is the launch film's music).
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
    # The 0.10 update video (~41 s): a short build, the drop at ~13 s (the image hatches), a
    # bright groove, a lift around 30 s and a clean ending (no fade needed).
    "update": (
        "deep melodic house, instrumental, short filtered intro with soft plucks and rising tension, "
        "big satisfying drop early, warm sub bass, crisp hi-hats, dreamy bright chords, "
        "confident and playful, a lift in the second half, clean definite ending, modern clean mix",
        118,
        "A Minor",
    ),
    # The "use cases" video (~88 s, with voice): one track whose sections follow the video, playful
    # at the start, a quiet piano part in the middle (grandparents, privacy, kids), the groove back
    # for the second half, a clean ending. The sections come from structure tags (STRUCTURE).
    "usecases": (
        "instrumental, no vocals, feel-good upbeat indie pop, warm electric piano, clean funky guitar, "
        "bouncy bass, crisp claps and tight drums, playful and bright; a soft heartfelt solo piano "
        "breakdown in the middle with the drums out; then the full groove returns; clean definite "
        "ending, modern clean mix, sits under a voice-over",
        116,
        "D Major",
    ),
}
# Lyrics as a map of the sections (no words): ACE-Step plans energy and dynamics from the tags.
STRUCTURE = {
    "usecases": "\n\n".join([
        "[Intro - upbeat, drums from the first bar]",
        "[Verse - playful groove, full band]",
        "[Breakdown - soft solo piano and warm pads, no drums, tender]",
        "[Piano Interlude - gentle, emotional]",
        "[Build - drums come back]",
        "[Chorus - full playful groove, bright]",
        "[Chorus - full playful groove, bright]",
        "[Outro - clean definite ending]",
    ]),
}
# The "Kokon 0.11" video (76 s, one track played whole under the voice, no cuts): the 0.10 prompt as
# it was, and the same sound with the same energy from the first bar to the last.
STYLES["v011"] = (STYLES["update"][0], 118, "A Minor")
STYLES["v011-steady"] = (
    "deep melodic house, instrumental, warm sub bass, crisp hi-hats, dreamy bright chords, soft plucks, "
    "confident and playful, steady groove from the first bar to the end, same energy all the way, "
    "no breakdown, no build-ups, clean definite ending, modern clean mix, sits under a voice-over",
    118,
    "A Minor",
)
DURATIONS = {"update": 44, "usecases": 94, "v011": 80, "v011-steady": 80}
# New versions of an existing take (ACE-Step "cover": same tempo, harmony and form, new notes and
# sounds). name: (style whose prompt describes it, source take, cover strength: near 1 stays close).
# The source is the launch film's house-4, which lives in the main clone's takes/ folder.
LAUNCH_TAKES = Path("C:/AI/qwen-local-studio/marketing/audio/takes")
COVERS = {
    "house4-cover": ("house", LAUNCH_TAKES / "house-4.wav", 0.9),
    "house4-cover-loose": ("house", LAUNCH_TAKES / "house-4.wav", 0.75),
}


def call(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body is not None else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    if resp.get("code") != 200:
        raise RuntimeError(f"{path}: {resp.get('error')}")
    return resp["data"]


def call_with_file(path: str, fields: dict, field: str, file: Path) -> dict:
    """A multipart request: the API takes source audio only as an upload, never as a local path."""
    boundary = f"----kokon{int(time.time() * 1000)}"
    parts = []
    for k, v in fields.items():
        v = json.dumps(v) if isinstance(v, bool) else str(v)
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{file.name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode())
    parts += [file.read_bytes(), f"\r\n--{boundary}--\r\n".encode()]
    req = urllib.request.Request(API + path, data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    if resp.get("code") != 200:
        raise RuntimeError(f"{path}: {resp.get('error')}")
    return resp["data"]


def generate(style: str) -> None:
    base, src, strength = COVERS.get(style, (style, None, None))
    prompt, bpm, key = STYLES[base]
    body = {
        "prompt": prompt,
        "lyrics": STRUCTURE.get(style, "[Instrumental]"),
        "instrumental": True,
        "thinking": True,
        "bpm": bpm,
        "key_scale": key,
        "time_signature": "4",
        "audio_duration": DURATIONS.get(style, DURATION),
        "batch_size": TAKES,
        "inference_steps": 8,
        "audio_format": "wav",
    }
    if src:
        body.update(task_type="cover", audio_cover_strength=strength)
        del body["audio_duration"]  # the source's length
        task = call_with_file("/release_task", body, "src_audio", src)
    else:
        task = call("/release_task", body)
    task_id = task["task_id"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
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
        dest = OUT / (f"{style}-{stamp}-{n}.wav" if style in DURATIONS or style in COVERS else f"{style}-{n}.wav")
        urllib.request.urlretrieve(API + res["file"], dest)
        print(f"{dest.name}  seed {res.get('seed_value', '?')}  {res.get('metas', {})}")
    print(f"{style}: {len(results)} takes in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    # A style must be named: running every style would write over the launch takes.
    if not sys.argv[1:]:
        raise SystemExit(f"usage: python gen_music.py STYLE [...]   ({', '.join([*STYLES, *COVERS])})")
    try:
        urllib.request.urlopen(API, timeout=5).close()
    except urllib.error.HTTPError:
        pass  # the server answered
    except OSError:
        raise SystemExit(f"ACE-Step API not reachable at {API}: start it first (cd D:/AI/ACE-Step-1.5; uv run acestep-api) and wait until it listens")
    for s in sys.argv[1:]:
        generate(s)
