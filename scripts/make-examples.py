"""Builds the example projects bundled with Kokon (assets/examples) from the full-size
showcase originals (made by Kokon, see marketing/showcase/PROMPTS.md).

    python scripts/make-examples.py [originals folder]   # default: ~/Pictures/Kokon/marketing
"""
import os
import sys
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.expanduser("~"), "Pictures", "Kokon", "marketing")
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "examples")
LONG = 1600
# Chains share one scale so masks and expansions keep matching their images.
GROUPS = {
    "street": ["street", "street-removed", "street-cat", "street-mask", "street-add-mask"],
    "cabin": ["cabin", "cabin-winter"],
    "bakery": ["bakery", "bakery-text"],
    "lighthouse": ["lighthouse", "lighthouse-wide"],
    "potter": ["potter", "potter-watercolor"],
}

os.makedirs(OUT, exist_ok=True)
for names in GROUPS.values():
    ims = {n: Image.open(os.path.join(SRC, n + ".png")) for n in names}
    k = min(1.0, LONG / max(max(im.size) for im in ims.values()))
    for n, im in ims.items():
        size = (round(im.width * k), round(im.height * k))
        im = im.resize(size, Image.LANCZOS)
        if im.mode == "L":
            im.point(lambda v: 255 if v >= 128 else 0).save(os.path.join(OUT, n + ".png"), optimize=True)
        else:
            im.convert("RGB").save(os.path.join(OUT, n + ".jpg"), quality=86, optimize=True, progressive=True)
        print(n, size)
