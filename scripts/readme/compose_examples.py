"""Builds the README before/after images.

Usage: python scripts/readme/compose_examples.py docs/media/raw docs/media/examples
Reads <name>-before.jpg and <name>-after.png from the first folder and writes
<name>.jpg (1200x600, side by side, labelled) into the second.
"""
import sys, os
from PIL import Image, ImageOps, ImageDraw, ImageFont
R, OUT = sys.argv[1], sys.argv[2]
W, H, GAP = 1200, 600, 12
PAPER = (247,246,242)
f = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 20)
def tag(im, text, x, y):
    d = ImageDraw.Draw(im, "RGBA"); w = d.textlength(text, font=f)
    d.rounded_rectangle([x, y, x+w+20, y+32], 8, fill=(13,13,13,190)); d.text((x+10, y+3), text, font=f, fill="white")
for n in ["remove","replace","relight","restyle","text","expand"]:
    b, a = os.path.join(R, f"{n}-before.jpg"), os.path.join(R, f"{n}-after.png")
    if not os.path.exists(a): print("skip", n); continue
    cw = (W - GAP) // 2
    canvas = Image.new("RGB", (W, H), PAPER)
    for i, p in enumerate([b, a]):
        im = Image.open(p).convert("RGB")
        if n == "expand":   # keep whole frames: square before, 16:9 after, letterboxed on paper
            im.thumbnail((cw, H)); x = i*(cw+GAP) + (cw-im.width)//2; y = (H-im.height)//2
        else:
            im = ImageOps.fit(im, (cw, H), Image.LANCZOS); x, y = i*(cw+GAP), 0
        canvas.paste(im, (x, y)); tag(canvas, ["Before","After"][i], x+12, y+12)
    canvas.save(os.path.join(OUT, f"{n}.jpg"), quality=88, optimize=True); print("ok", n)
