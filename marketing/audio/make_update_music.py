"""The music of the "Kokon 0.10" update video, cut from ACE-Step take update-20261020-175836-1.

    D:/AI/tts/.venv/Scripts/python make_update_music.py   -> ../video/public/music-0.10.m4a

The take (118 BPM, measured with librosa: beats every 0.50849 s, drop on the bar line at 16.252 s)
builds quietly, drops, plays 8 bars and stops dead. The video needs the drop on its beat 26 (the
image hatching) and about 12 bars after it, so: start 26 beats before the drop, play bars 1-7,
repeat bars 4-7, then bar 8 with the stop and the take's quiet tail. Every join is on a bar line,
with a 10 ms crossfade. The stop lands 37.63 s into the video (STOP in timeline010.ts).
"""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
TAKE = HERE / "takes" / "update-20261020-175836-1.wav"
OUT = HERE.parent / "video" / "public" / "music-0.10.m4a"
BEAT = 60 / 118
DROP = 16.252
BAR = 4 * BEAT
FADE = 0.010


def main() -> None:
    y, sr = sf.read(TAKE, dtype="float32", always_2d=True)
    at = lambda t: int(round(t * sr))
    bar = lambda i: DROP + i * BAR  # start of groove bar i (0-based)
    start = DROP - 26 * BEAT
    parts = [
        y[at(start):at(bar(7))],  # the build and bars 1-7
        y[at(bar(3)):at(bar(7))],  # bars 4-7 again
        y[at(bar(7)):at(bar(8) + 3.5)],  # bar 8, the stop, the tail
    ]
    n = at(FADE)
    ramp = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    out = parts[0]
    for p in parts[1:]:
        # The tail of what came before overlaps the head of the next part.
        mixed = out[-n:] * (1 - ramp) + p[:n] * ramp
        out = np.concatenate([out[:-n], mixed, p[n:]])
    wav = HERE / "takes" / "music-0.10.wav"
    sf.write(wav, out, sr)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "192k", str(OUT)], check=True)
    stop = (bar(8) - start) - 2 * FADE + 4 * BAR
    print(f"{OUT}: {len(out) / sr:.2f} s, drop at {DROP - start:.3f} s, stop at {stop:.3f} s")


if __name__ == "__main__":
    main()
