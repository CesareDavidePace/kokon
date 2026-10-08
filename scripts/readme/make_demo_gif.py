"""Builds docs/media/demo.gif from a raw screen recording of the app.

Usage: python scripts/readme/make_demo_gif.py <recording.mp4> <out.gif> [pasta|mug]
Interaction plays sped up; each generation wait is squeezed to ~1.2 s with a label
showing how long it really took. Segment times are per recording (found from the
brightness of the image area: waits are dimmed and flicker with the progress bar).
"""
import subprocess, sys

SRC, OUT = sys.argv[1], sys.argv[2]
WIDTH, FPS, WAIT_S, HOLD_S = 900, 10, 1.0, 3.0
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
# (start, end, speed) for interaction, or (start, end, "wait")
RECORDINGS = {
    # 2026-10-08: mug, create -> colour edits -> kitchen -> text -> expand
    "mug": [
        (0, 7, 2.5), (7, 27, "wait"), (27, 37, 2.5), (37, 63, "wait"), (63, 71, 2.5),
        (71, 97, "wait"), (97, 102, 2.5), (102, 128, "wait"), (128, 134, 2.5),
        (134, 159, "wait"), (159, 198, 4.0), (198, 238, "wait"), (238, 241.7, 1.0),
    ],
    # 2026-10-09: "Fatto in casa." Colosseum -> variation -> pasta -> sauce -> kitchen
    # -> combine -> expand -> title
    "pasta": [
        (0, 4, 3.0), (4, 23, "wait"), (23, 25, 1.0), (25, 45, "wait"), (45, 55, 3.0),
        (55, 79, "wait"), (79, 95, 2.5), (95, 120, "wait"), (120, 137, 3.5),
        (137, 157, "wait"), (157, 172, 3.0), (172, 203, "wait"), (203, 228, 4.0),
        (228, 268, "wait"), (268, 279, 3.0), (279, 320, "wait"), (320, 325.1, 2.0),  # ends on the full title
    ],
}
SEGMENTS = RECORDINGS[sys.argv[3] if len(sys.argv) > 3 else "pasta"]

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
        "[x]palettegen=max_colors=192:stats_mode=diff[p];[y][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, "-filter_complex", graph, "-loop", "0", OUT], check=True)
print("ok", OUT)
