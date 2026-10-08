"""Builds docs/media/demo.gif from a raw screen recording of the app.

Usage: python scripts/readme/make_demo_gif.py <recording.mp4> <out.gif>
Interaction plays sped up; each generation wait is squeezed to ~1.2 s with a label
showing how long it really took. Segment times are for the 2026-10-08 recording.
"""
import subprocess, sys

SRC, OUT = sys.argv[1], sys.argv[2]
WIDTH, FPS, WAIT_S, HOLD_S = 1000, 12, 1.2, 2.5
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
# (start, end, speed) for interaction, or (start, end, "wait")
SEGMENTS = [
    (0, 7, 2.5), (7, 27, "wait"), (27, 37, 2.5), (37, 63, "wait"), (63, 71, 2.5),
    (71, 97, "wait"), (97, 102, 2.5), (102, 128, "wait"), (128, 134, 2.5),
    (134, 159, "wait"), (159, 198, 4.0), (198, 238, "wait"), (238, 241.7, 1.0),
]

parts, labels = [], []
for i, (a, b, sp) in enumerate(SEGMENTS):
    f = f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS"
    if sp == "wait":
        f += f",setpts=PTS*{WAIT_S / (b - a):.5f}"
        f += (f",drawtext=fontfile='{FONT}':text='sped up · {round(b - a)} s':fontsize=44:fontcolor=white"
              f":box=1:boxcolor=0x0d0d0dcc:boxborderw=18:x=(w-tw)/2:y=h*0.08")
    else:
        f += f",setpts=PTS/{sp}"
    if i == len(SEGMENTS) - 1:
        f += f",tpad=stop_mode=clone:stop_duration={HOLD_S}"
    parts.append(f + f",fps={FPS}[s{i}]")
    labels.append(f"[s{i}]")
graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(labels)}:v=1:a=0,scale={WIDTH}:-1:flags=lanczos,split[x][y];" \
        "[x]palettegen=max_colors=192:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-filter_complex", graph, "-loop", "0", OUT], check=True)
print("ok", OUT)
