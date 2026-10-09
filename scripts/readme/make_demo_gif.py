"""Builds docs/media/demo.gif (or an .mp4 with the same cut) from a screen recording of the app.

Usage: python scripts/readme/make_demo_gif.py <recording.mp4> <out.gif> [demo|pasta|mug] [width] [fps]
Interaction plays sped up; each generation wait is squeezed to ~1.2 s with a label
showing how long it really took. Segment times are per recording (found from the
brightness of the image area: waits are dimmed and flicker with the progress bar).
"""
import subprocess, sys

SRC, OUT = sys.argv[1], sys.argv[2]
WIDTH, FPS, WAIT_S, HOLD_S = 900, 10, 1.0, 3.0
WIDTH = int(sys.argv[4]) if len(sys.argv) > 4 else WIDTH
FPS = int(sys.argv[5]) if len(sys.argv) > 5 else FPS
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
    # 2026-10-09: "Fatto in casa." TV series. Colosseum -> pasta -> sauce -> kitchen with
    # grandmother -> combine -> expand -> series key art with the title logo
    "series": [
        (0, 100.5, "wait"), (100.5, 113.5, 4.0), (113.5, 197.5, "wait"), (197.5, 216, 3.0),
        (216, 297.5, "wait"), (297.5, 316, 3.0), (316, 328.5, 4.0), (328.5, 417, "wait"),
        (417, 437.5, 4.0), (437.5, 498.5, "wait"), (498.5, 537, 4.0), (537, 626.5, "wait"),
        (626.5, 643, 4.0), (643, 729, "wait"), (729, 734, 1.0),  # ends on the key art
    ],
}
# docs/media/demo.mp4: the "pasta" recording zoomed onto the app window (same timing),
# with black margins around it that are cropped away; then padded in the app's background
# colour so the UI does not touch the edges of GitHub's video player.
RECORDINGS["demo"] = RECORDINGS["pasta"]
CROPS = {"demo": "crop=1304:994:460:78,pad=1464:1134:80:70:color=0x161616,",
         # recorded as the app window only: just add the margin
         "series": "pad=1964:1494:80:70:color=0x161616,"}
NAME = sys.argv[3] if len(sys.argv) > 3 else "demo"
SEGMENTS, CROP = RECORDINGS[NAME], CROPS.get(NAME, "")
# label position as a fraction of the frame: centred on the image, below the app's top bar
LABEL_X, LABEL_Y = {"series": (0.6, 0.2)}.get(NAME, (0.5, 0.08))

parts, labels = [], []
for i, (a, b, sp) in enumerate(SEGMENTS):
    f = f"[0:v]{CROP}trim={a}:{b},setpts=PTS-STARTPTS"
    if sp == "wait":
        f += f",setpts=PTS*{WAIT_S / (b - a):.5f}"
        f += (f",drawtext=fontfile='{FONT}':text='sped up · {round(b - a)} s':fontsize=44:fontcolor=white"
              f":box=1:boxcolor=0x0d0d0dcc:boxborderw=18:x=w*{LABEL_X}-tw/2:y=h*{LABEL_Y}")
    else:
        f += f",setpts=PTS/{sp}"
    if i == len(SEGMENTS) - 1:
        f += f",tpad=stop_mode=clone:stop_duration={HOLD_S}"
    parts.append(f + f",fps={FPS}[s{i}]")
    labels.append(f"[s{i}]")
joined = ";".join(parts) + f";{''.join(labels)}concat=n={len(labels)}:v=1:a=0,scale={WIDTH}:-2:flags=lanczos"
if OUT.endswith(".mp4"):  # the same cut as a small H.264 video, for GitHub's video player
    args = ["-filter_complex", joined, "-an", "-c:v", "libx264", "-crf", "22", "-preset", "slow",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
else:
    args = ["-filter_complex", joined + ",split[x][y];[x]palettegen=max_colors=192:stats_mode=diff[p];"
            "[y][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", "-loop", "0"]
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", SRC, *args, OUT], check=True)
print("ok", OUT)
