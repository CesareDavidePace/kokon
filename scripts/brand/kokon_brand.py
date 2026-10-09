"""Kokon brand assets: the mark is defined here as geometry (cubic Bézier curves) and
rendered to SVG and PNG, so every size is generated from one source.

    python scripts/brand/kokon_brand.py preview          # variant comparison sheet
    python scripts/brand/kokon_brand.py build <variant>  # full asset set into assets/brand

Needs Pillow + numpy; the wordmark also needs fontTools, brotli and skia-pathops (reads the Inter font the
app already ships in node_modules/@fontsource-variable/inter).
"""

from __future__ import annotations

import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "brand"

INK = "#0D0D0D"        # the app's text colour
PAPER = "#F7F6F2"      # warm off-white of the app icon
INK_RGB = (13, 13, 13)
PAPER_RGB = (247, 246, 242)


# ---------------------------------------------------------------------------------------
# Paths: one builder gives both the SVG path data and a polygon for rasterizing.
# ---------------------------------------------------------------------------------------

class PathB:
    def __init__(self):
        self.d: list[str] = []
        self.pts: list[tuple[float, float]] = []
        self.cur = (0.0, 0.0)

    def move(self, p):
        self.d.append(f"M{p[0]:.2f} {p[1]:.2f}")
        self.pts.append(p)
        self.cur = p
        return self

    def line(self, p):
        self.d.append(f"L{p[0]:.2f} {p[1]:.2f}")
        self.pts.append(p)
        self.cur = p
        return self

    def curve(self, c1, c2, p, n=96):
        self.d.append(f"C{c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p[0]:.2f} {p[1]:.2f}")
        p0 = np.array(self.cur)
        a, b, c = np.array(c1), np.array(c2), np.array(p)
        for t in np.linspace(0, 1, n)[1:]:
            q = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t * t * b + t ** 3 * c
            self.pts.append((float(q[0]), float(q[1])))
        self.cur = p
        return self

    def close(self):
        self.d.append("Z")
        return self

    def transformed(self, s, dx, dy):
        out = PathB()
        out.pts = [(x * s + dx, y * s + dy) for x, y in self.pts]
        # Re-emit SVG data with the transform applied.
        res = []
        for tok in self.d:
            cmd, nums = tok[0], tok[1:].split()
            vals = [float(v) for v in nums]
            vals = [v * s + (dx if i % 2 == 0 else dy) for i, v in enumerate(vals)]
            res.append(cmd + " ".join(f"{v:.2f}" for v in vals) if vals else cmd)
        out.d = res
        return out


# ---------------------------------------------------------------------------------------
# The mark
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    stem: str           # "leaf": two opposite corners rounded; "cocoon": rounded capsule
    wings: bool         # False: equal leaves; True: upper leaf larger, like butterfly wings


VARIANTS = {
    "a": Variant("a", "A · Bozzolo", "cocoon", False),
    "b": Variant("b", "B · Ali", "leaf", True),
    "c": Variant("c", "C · Bozzolo + ali", "cocoon", True),
}

H = 660.0        # mark height (units)
S = 238.0        # stem width
GAP = 26.0       # space between stem and leaves, and between the leaves
W = 430.0        # leaf width
K_STEM = 0.68    # curve fullness fitted to the generated logo (1.0 = square, 0.55 = circle)
K_LEAF = 0.57


def stem_path(v: Variant) -> PathB:
    p = PathB()
    x0, x1, y0, y1 = 0.0, S, 0.0, H
    if v.stem == "leaf":
        ry, k = 245.0, K_STEM
        p.move((x0, y0))
        p.curve((x0 + k * S, y0), (x1, y0 + ry - k * ry), (x1, y0 + ry))
        p.line((x1, y1))
        p.curve((x1 - k * S, y1), (x0, y1 - ry + k * ry), (x0, y1 - ry))
        return p.close()
    # Cocoon: a capsule whose ends are a little taller than a half circle.
    r, ry, k = S / 2, S * 0.62, 0.60
    cx = S / 2
    p.move((x0, y0 + ry))
    p.curve((x0, y0 + ry - k * ry), (cx - k * r, y0), (cx, y0))
    p.curve((cx + k * r, y0), (x1, y0 + ry - k * ry), (x1, y0 + ry))
    p.line((x1, y1 - ry))
    p.curve((x1, y1 - ry + k * ry), (cx + k * r, y1), (cx, y1))
    p.curve((cx - k * r, y1), (x0, y1 - ry + k * ry), (x0, y1 - ry))
    return p.close()


def leaf_up(x0, y0, x1, y1, k=K_LEAF) -> PathB:
    """Corner at bottom-left (x0, y1), tip at top-right (x1, y0)."""
    w, h = x1 - x0, y1 - y0
    p = PathB().move((x0, y1))
    p.curve((x0, y1 - k * h), (x1 - k * w, y0), (x1, y0))
    p.curve((x1, y0 + k * h), (x0 + k * w, y1), (x0, y1))
    return p.close()


def leaf_down(x0, y0, x1, y1, k=K_LEAF) -> PathB:
    """Corner at top-left (x0, y0), tip at bottom-right (x1, y1)."""
    w, h = x1 - x0, y1 - y0
    p = PathB().move((x0, y0))
    p.curve((x0, y0 + k * h), (x1 - k * w, y1), (x1, y1))
    p.curve((x1, y1 - k * h), (x0 + k * w, y0), (x0, y0))
    return p.close()


def mark(v: Variant) -> list[PathB]:
    lx = S + GAP
    room = H - GAP
    if v.wings:
        hu, hl = room * 0.56, room * 0.44
        wu, wl = W, W * 0.84
    else:
        hu = hl = room / 2
        wu = wl = W
    return [stem_path(v), leaf_up(lx, 0, lx + wu, hu), leaf_down(lx, hu + GAP, lx + wl, H)]


def bbox(paths):
    pts = np.array([p for path in paths for p in path.pts])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def placed(paths, box, align="center"):
    """Scales and centres `paths` into box = (x, y, w, h)."""
    x0, y0, x1, y1 = bbox(paths)
    s = min(box[2] / (x1 - x0), box[3] / (y1 - y0))
    dx = box[0] + (box[2] - (x1 - x0) * s) / 2 - x0 * s
    dy = box[1] + (box[3] - (y1 - y0) * s) / 2 - y0 * s
    if align == "left":
        dx = box[0] - x0 * s
    return [p.transformed(s, dx, dy) for p in paths], s


# ---------------------------------------------------------------------------------------
# App icon shape: a superellipse ("squircle"), like iOS and Windows 11 app tiles.
# ---------------------------------------------------------------------------------------

def squircle(cx, cy, r, n=5.0, steps=720) -> PathB:
    p = PathB()
    for i in range(steps):
        t = 2 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        x = cx + r * math.copysign(abs(c) ** (2 / n), c)
        y = cy + r * math.copysign(abs(s) ** (2 / n), s)
        (p.move if i == 0 else p.line)((x, y))
    return p.close()


# ---------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------

SS = 4  # supersampling


def raster(size, layers, bg=None):
    """layers: list of (paths, rgba). Returns an RGBA image of `size` (w, h)."""
    W_, H_ = size[0] * SS, size[1] * SS
    img = Image.new("RGBA", (W_, H_), bg or (0, 0, 0, 0))
    for paths, color in layers:
        m = Image.new("L", (W_, H_), 0)
        d = ImageDraw.Draw(m)
        for p in paths:
            d.polygon([(x * SS, y * SS) for x, y in p.pts], fill=255)
        solid = Image.new("RGBA", (W_, H_), color)
        img = Image.composite(solid, img, m) if color[3] == 255 else Image.alpha_composite(img, Image.composite(solid, Image.new("RGBA", (W_, H_), (0, 0, 0, 0)), m))
    return img.resize(size, Image.LANCZOS)


def svg(size, layers, bg=None) -> str:
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size[0]} {size[1]}" width="{size[0]}" height="{size[1]}">']
    if bg:
        parts.append(f'<rect width="{size[0]}" height="{size[1]}" fill="{bg}"/>')
    for paths, fill, extra in layers:
        parts.append(f'<path fill="{fill}" {extra}d="{" ".join(" ".join(p.d) for p in paths)}"/>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def icon_layers(v: Variant, size: int, inverse=False, border=True):
    """App icon: squircle tile with the mark inside. Returns (raster layers, svg layers)."""
    r = size * 0.46
    tile = squircle(size / 2, size / 2, r)
    m = size * 0.50  # mark height
    paths, _ = placed(mark(v), (size / 2 - m / 2, size / 2 - m / 2, m, m))
    tile_rgb, mark_rgb = (INK_RGB, PAPER_RGB) if inverse else (PAPER_RGB, INK_RGB)
    tile_hex, mark_hex = (INK, PAPER) if inverse else (PAPER, INK)
    rl, sl = [], []
    if border and not inverse:
        # A hairline keeps a light tile visible on light backgrounds (Windows light theme).
        edge = squircle(size / 2, size / 2, r + max(1.0, size / 512))
        rl.append(([edge], (0, 0, 0, 28)))
        sl.append(([edge], "#000000", 'fill-opacity="0.11" '))
    rl += [([tile], tile_rgb + (255,)), (paths, mark_rgb + (255,))]
    sl += [([tile], tile_hex, ""), (paths, mark_hex, "")]
    return rl, sl


def macos_icon(v: Variant, size: int):
    """macOS app icon on Apple's template grid: an 824/1024 squircle centred on the canvas
    (100 px margin), mark at the same share of the tile as the desktop icon, and the soft
    drop shadow every Dock icon has. Without the margin Kokon looks bigger than its
    neighbours in the Dock."""
    k = size / 1024
    r = 412 * k
    cx, cy = size / 2, size / 2
    tile = squircle(cx, cy, r)
    m = 2 * r * (0.50 / 0.92)  # mark/tile ratio of icon_layers
    paths, _ = placed(mark(v), (cx - m / 2, cy - m / 2, m, m))
    # Shadow: the tile shape, 12 px lower, blurred 20 px, 30 % black (at 1024).
    from PIL import ImageFilter
    sh = Image.new("L", (size * SS, size * SS), 0)
    ImageDraw.Draw(sh).polygon([(x * SS, (y + 12 * k) * SS) for x, y in tile.pts], fill=int(255 * 0.30))
    sh = sh.resize((size, size), Image.LANCZOS)
    if size >= 64:
        sh = sh.filter(ImageFilter.GaussianBlur(20 * k))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    img.putalpha(sh)
    body = raster((size, size), [([tile], PAPER_RGB + (255,)), (paths, INK_RGB + (255,))])
    return Image.alpha_composite(img, body)


# ---------------------------------------------------------------------------------------
# Inter (wordmark): instance the variable font the app ships, as TTF for Pillow and paths.
# ---------------------------------------------------------------------------------------

def inter_ttf(weight=600) -> Path:
    out = Path(__file__).with_name(f".inter-{weight}-merged.ttf")
    if out.exists():
        return out
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import OverlapMode, instantiateVariableFont

    src = ROOT / "node_modules" / "@fontsource-variable" / "inter" / "files" / "inter-latin-wght-normal.woff2"
    f = TTFont(src)
    # Variable fonts keep overlapping and self-overlapping contours; merge them (skia-pathops)
    # so the outlines fill correctly with any rule, in SVG and in raster.
    inst = instantiateVariableFont(f, {"wght": weight}, overlap=OverlapMode.REMOVE)
    inst.flavor = None
    inst.save(out)
    return out


def text_paths(text, size_px, x, y_baseline, weight=600, tracking=-0.02) -> list[PathB]:
    """Glyph outlines of `text` as PathB (y down), baseline at y_baseline."""
    from fontTools.pens.basePen import BasePen
    from fontTools.ttLib import TTFont

    f = TTFont(inter_ttf(weight))
    upm = f["head"].unitsPerEm
    s = size_px / upm
    cmap, gs, hmtx = f.getBestCmap(), f.getGlyphSet(), f["hmtx"]
    paths: list[PathB] = []

    class Pen(BasePen):
        def __init__(self, ox):
            super().__init__(gs)
            self.ox = ox
            self.p = None

        def tr(self, pt):
            return (self.ox + pt[0] * s, y_baseline - pt[1] * s)

        def _moveTo(self, pt):
            self.p = PathB().move(self.tr(pt))

        def _lineTo(self, pt):
            self.p.line(self.tr(pt))

        def _curveToOne(self, a, b, c):
            self.p.curve(self.tr(a), self.tr(b), self.tr(c), n=24)

        def _closePath(self):
            paths.append(self.p.close())

    pen_x = x
    for ch in text:
        g = cmap[ord(ch)]
        gs[g].draw(Pen(pen_x))
        pen_x += hmtx[g][0] * s + tracking * size_px
    return paths


def signed_area(p: PathB) -> float:
    pts = p.pts
    return 0.5 * sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))


def raster_text(size, paths, color, bg=None):
    """Non-zero fill like a font renderer: contours turning the same way as the largest one
    are filled (overlapping parts of a glyph merge), the others are holes (the inside of o)."""
    W_, H_ = size[0] * SS, size[1] * SS
    outer_sign = math.copysign(1, max((signed_area(p) for p in paths), key=abs))
    m = Image.new("L", (W_, H_), 0)
    d = ImageDraw.Draw(m)
    for hole in (False, True):
        for p in paths:
            if (math.copysign(1, signed_area(p)) != outer_sign) == hole:
                d.polygon([(x * SS, y * SS) for x, y in p.pts], fill=0 if hole else 255)
    return m


# ---------------------------------------------------------------------------------------
# Preview sheet
# ---------------------------------------------------------------------------------------

def preview():
    font = ImageFont.truetype(str(inter_ttf(600)), 26)
    small = ImageFont.truetype(str(inter_ttf(500)), 16)
    col_w, sheet_h = 460, 900
    sheet = Image.new("RGBA", (col_w * len(VARIANTS) + 40, sheet_h), (255, 255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, v in enumerate(VARIANTS.values()):
        x = 20 + i * col_w
        d.text((x + 20, 20), v.label, fill=INK_RGB + (255,), font=font)
        # Large mark.
        paths, _ = placed(mark(v), (0, 0, 300, 300))
        sheet.alpha_composite(raster((300, 300), [(paths, INK_RGB + (255,))]), (x + 60, 80))
        # App icon on a light and on a dark background.
        for j, bgc in enumerate([(232, 232, 232, 255), (32, 32, 32, 255)]):
            box = Image.new("RGBA", (200, 200), bgc)
            rl, _ = icon_layers(v, 180)
            box.alpha_composite(raster((180, 180), rl), (10, 10))
            sheet.alpha_composite(box, (x + 10 + j * 215, 420))
        d.text((x + 20, 630), "Inverse (web)", fill=(90, 90, 90, 255), font=small)
        rl, _ = icon_layers(v, 120, inverse=True)
        sheet.alpha_composite(raster((120, 120), rl), (x + 20, 660))
        # Real small sizes, shown at 1:1 and enlarged ×4 (nearest) to judge legibility.
        d.text((x + 170, 630), "48 / 32 / 16 px", fill=(90, 90, 90, 255), font=small)
        ox = x + 170
        for sz in (48, 32, 16):
            rl, _ = icon_layers(v, sz)
            sheet.alpha_composite(raster((sz, sz), rl), (ox, 665))
            ox += sz + 12
        rl, _ = icon_layers(v, 16)
        big = raster((16, 16), rl).resize((96, 96), Image.NEAREST)
        sheet.alpha_composite(big, (x + 170, 740))
        d.text((x + 275, 800), "16 px ×6", fill=(140, 140, 140, 255), font=small)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "explorations" / "kokon-variants.png"
    f.parent.mkdir(exist_ok=True)
    sheet.convert("RGB").save(f)
    print(f)


# ---------------------------------------------------------------------------------------
# Full asset set
# ---------------------------------------------------------------------------------------

def lockup(v: Variant, font_px: float, mark_rgb, text_hex_for_svg=None):
    """Mark + "Kokon". Returns (size, mark paths, text paths)."""
    cap = 0.727 * font_px  # Inter cap height
    mh = cap * 1.12
    gap = mh * 0.30
    pad = 0.0
    m_paths, _ = placed(mark(v), (pad, 0, mh * 2, mh), align="left")
    mx1 = bbox(m_paths)[2]
    baseline = mh / 2 + cap / 2
    t_paths = text_paths("Kokon", font_px, mx1 + gap, baseline)
    x0, y0, x1, y1 = bbox(m_paths + t_paths)
    w, h = math.ceil(x1 - min(0, x0)), math.ceil(max(y1, mh))
    return (w, h), m_paths, t_paths


def write_svg(path: Path, size, layers, bg=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg(size, layers, bg), encoding="utf-8")


def compose(size, mark_paths, text_paths_, rgb, bg=None):
    """Raster of the mark and the text in one colour."""
    img = raster(size, [(mark_paths, rgb + (255,))], bg)
    if text_paths_:
        m = raster_text(size, text_paths_, rgb).resize(size, Image.LANCZOS)
        img.alpha_composite(Image.composite(Image.new("RGBA", size, rgb + (255,)), Image.new("RGBA", size, (0, 0, 0, 0)), m))
    return img


def build(v: Variant):
    OUT.mkdir(parents=True, exist_ok=True)
    made = []

    def save(img, rel):
        p = OUT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p)
        made.append(p)

    # 1. Mark alone (transparent): ink and white.
    for name, rgb, hexc in [("kokon-mark", INK_RGB, INK), ("kokon-mark-white", PAPER_RGB, PAPER)]:
        paths, _ = placed(mark(v), (0, 0, 1024, 1024))
        x0, y0, x1, y1 = bbox(paths)
        write_svg(OUT / "mark" / f"{name}.svg", (1024, 1024), [(paths, hexc, "")])
        made.append(OUT / "mark" / f"{name}.svg")
        save(raster((1024, 1024), [(paths, rgb + (255,))]), f"mark/{name}-1024.png")

    # 2. App icon (light tile, desktop) and inverse tile (web, social, anywhere else).
    for name, inv in [("kokon-icon", False), ("kokon-icon-inverse", True)]:
        _, sl = icon_layers(v, 1024, inverse=inv)
        write_svg(OUT / "icon" / f"{name}.svg", (1024, 1024), sl)
        made.append(OUT / "icon" / f"{name}.svg")
        for sz in (1024, 512, 256):
            rl, _ = icon_layers(v, sz, inverse=inv)
            save(raster((sz, sz), rl), f"icon/{name}-{sz}.png")

    # 3. Horizontal logo (mark + wordmark), for light and for dark backgrounds.
    size, mp, tp = lockup(v, 400, INK_RGB)
    pad = 40
    mp = [p.transformed(1, pad, pad) for p in mp]
    tp = [p.transformed(1, pad, pad) for p in tp]
    size = (size[0] + 2 * pad, size[1] + 2 * pad)
    for name, rgb, hexc in [("kokon-logo", INK_RGB, INK), ("kokon-logo-white", PAPER_RGB, PAPER)]:
        write_svg(OUT / "logo" / f"{name}.svg", size, [(mp, hexc, ""), (tp, hexc, "")])
        made.append(OUT / "logo" / f"{name}.svg")
        save(compose(size, mp, tp, rgb), f"logo/{name}.png")

    # 4. Web: favicon (inverse tile reads best in browser tabs, light and dark), touch icons.
    web_sizes = {}
    for sz in (16, 32, 48, 180, 192, 512):
        rl, _ = icon_layers(v, sz, inverse=True)
        web_sizes[sz] = raster((sz, sz), rl)
    save(web_sizes[180], "web/apple-touch-icon.png")
    save(web_sizes[192], "web/icon-192.png")
    save(web_sizes[512], "web/icon-512.png")
    ico = OUT / "web" / "favicon.ico"
    web_sizes[48].save(ico, sizes=[(16, 16), (32, 32), (48, 48)], append_images=[web_sizes[16], web_sizes[32]])
    made.append(ico)
    _, sl = icon_layers(v, 64, inverse=True)
    write_svg(OUT / "web" / "favicon.svg", (64, 64), sl)
    made.append(OUT / "web" / "favicon.svg")

    # 5. Desktop app .ico, every size rendered directly (sharper than downscaling 1024).
    ico_sizes = (16, 20, 24, 32, 40, 48, 64, 128, 256)
    imgs = {sz: raster((sz, sz), icon_layers(v, sz)[0]) for sz in ico_sizes}
    app_ico = OUT / "icon" / "kokon-app.ico"
    imgs[256].save(app_ico, sizes=[(s, s) for s in ico_sizes], append_images=[imgs[s] for s in ico_sizes if s != 256])
    made.append(app_ico)

    # 7. macOS app icon set (Apple template grid), installed into src-tauri/icons/macos.
    mac = {sz: macos_icon(v, sz) for sz in (16, 32, 64, 128, 256, 512, 1024)}
    for sz, img in mac.items():
        save(img, f"icon/macos/kokon-macos-{sz}.png")
    icns = OUT / "icon" / "macos" / "kokon.icns"
    mac[1024].save(icns, format="ICNS", append_images=[mac[s] for s in (16, 32, 64, 128, 256, 512)])
    made.append(icns)
    tauri = ROOT / "src-tauri" / "icons" / "macos"
    tauri.mkdir(parents=True, exist_ok=True)
    for sz, name in [(32, "32x32.png"), (128, "128x128.png"), (256, "128x128@2x.png"), (512, "icon.png")]:
        mac[sz].save(tauri / name)
        made.append(tauri / name)
    shutil.copyfile(icns, tauri / "icon.icns")
    made.append(tauri / "icon.icns")

    # 6. Social preview (GitHub, links): dark, white logo, one line of copy.
    W_, H_ = 1280, 640
    size_l, mp, tp = lockup(v, 150, PAPER_RGB)
    ox, oy = (W_ - size_l[0]) / 2, 225
    mp = [p.transformed(1, ox, oy) for p in mp]
    tp = [p.transformed(1, ox, oy) for p in tp]
    img = compose((W_, H_), mp, tp, PAPER_RGB, bg=INK_RGB + (255,))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(str(inter_ttf(500)), 30)
    line = "Create and edit images by describing them. Privately, on your own GPU."
    tw = d.textlength(line, font=f)
    d.text(((W_ - tw) / 2, oy + size_l[1] + 70), line, font=f, fill=PAPER_RGB + (178,))
    save(img.convert("RGB"), "social/kokon-social-1280x640.png")

    for p in made:
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preview"
    if cmd == "preview":
        preview()
    elif cmd == "build":
        build(VARIANTS[sys.argv[2] if len(sys.argv) > 2 else "c"])
