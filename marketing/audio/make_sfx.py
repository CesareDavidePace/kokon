"""Synthesizes the launch video's sound effects (no samples, no licences) into video/public/sfx.

Every sound is soft and short, shares one small room reverb so they read as one family, and
is meant to sit well under the music. Run with any Python that has numpy + scipy, e.g. the
ACE-Step venv:  D:/AI/ACE-Step-1.5/.venv/Scripts/python make_sfx.py
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000
OUT = Path(__file__).parent.parent / "video" / "public" / "sfx"
rng = np.random.default_rng(7)


def t(sec: float) -> np.ndarray:
    return np.arange(int(sec * SR)) / SR


def noise(sec: float) -> np.ndarray:
    return rng.standard_normal(int(sec * SR))


def band(x: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)


def low(x: np.ndarray, f: float, order: int = 2) -> np.ndarray:
    return sosfilt(butter(order, f, btype="low", fs=SR, output="sos"), x)


def env(n: int, attack: float, decay: float) -> np.ndarray:
    """Linear attack (s), then exponential decay with time constant `decay` (s)."""
    x = np.arange(n) / SR
    a = np.clip(x / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(x - attack, 0) / decay)


def fade(x: np.ndarray, fin: float = 0.002, fout: float = 0.02) -> np.ndarray:
    x = x.copy()
    i, o = int(fin * SR), int(fout * SR)
    if i:
        x[:i] *= np.linspace(0, 1, i)
    if o:
        x[-o:] *= np.linspace(1, 0, o)
    return x


# One small, dark room for everything.
_ir_t = t(0.45)
_ir = low(rng.standard_normal(_ir_t.size), 5000) * np.exp(-_ir_t / 0.11)
_ir /= np.abs(_ir).sum() / 6


def room(x: np.ndarray, wet: float = 0.18) -> np.ndarray:
    """Stereo: dry signal plus slightly different reverb tails per side."""
    pad = np.concatenate([x, np.zeros(_ir.size)])
    l = fftconvolve(pad, _ir)[: pad.size]
    r = fftconvolve(pad, np.roll(_ir, 37))[: pad.size]
    return np.stack([pad + wet * l, pad + wet * r], axis=1)


def save(name: str, x: np.ndarray, peak_db: float) -> None:
    if x.ndim == 1:
        x = room(x)
    x = low(x.T, 12000).T  # keep the noise-based sounds soft on top
    x = x / np.abs(x).max() * 10 ** (peak_db / 20)
    OUT.mkdir(parents=True, exist_ok=True)
    wavfile.write(OUT / f"{name}.wav", SR, (x * 32767).astype(np.int16))
    print(f"{name}.wav  {x.shape[0] / SR:.2f} s")


def key_tap(pitch: float) -> np.ndarray:
    n = int(0.05 * SR)
    tick = band(rng.standard_normal(n), 1800 * pitch, 5200 * pitch) * env(n, 0.0005, 0.006)
    body = np.sin(2 * np.pi * 240 * pitch * t(0.05)) * env(n, 0.001, 0.012) * 0.5
    return tick + body


def click() -> np.ndarray:
    n = int(0.08 * SR)
    tick = band(rng.standard_normal(n), 2500, 7000) * env(n, 0.0003, 0.004)
    body = np.sin(2 * np.pi * 1450 * t(0.08)) * env(n, 0.0005, 0.018) * 0.35
    thump = np.sin(2 * np.pi * 170 * t(0.08)) * env(n, 0.001, 0.02) * 0.6
    return fade(tick + body + thump)


def typing(sec: float = 9.0) -> np.ndarray:
    """Irregular keystrokes, ~11 per second, a few heavier ones (space bar)."""
    out = np.zeros(int(sec * SR))
    pos = 0.02
    while pos < sec - 0.1:
        k = key_tap(rng.uniform(0.85, 1.2)) * rng.uniform(0.55, 1.0)
        if rng.random() < 0.15:
            k = key_tap(0.7) * 1.1
        i = int(pos * SR)
        out[i : i + k.size] += k[: out.size - i]
        pos += rng.uniform(0.055, 0.13)
    return out


def send() -> np.ndarray:
    x = t(0.32)
    f = 520 + 520 * (x / x[-1]) ** 1.5
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(x.size, 0.02, 0.08)
    air = band(noise(0.32), 2000, 8000) * env(x.size, 0.03, 0.06) * 0.25
    return fade(tone * 0.5 + air)


def tone(f: float, sec: float, attack: float, decay: float, partials=((1, 1.0),)) -> np.ndarray:
    """A soft sine voice with a few partials and a slow, slight vibrato."""
    x = t(sec)
    v = 1 + 0.002 * np.sin(2 * np.pi * 4.5 * x)
    out = sum(np.sin(2 * np.pi * f * k * x * v) * w for k, w in partials)
    return out * env(x.size, attack, decay)


def reveal(kind: str) -> np.ndarray:
    """A result appears: a breath of air that opens, with a low, warm note under it. Three
    variants (notes from A minor) take turns in the video so no two reveals in a row match."""
    sec = 1.6
    n = int(sec * SR)
    x = t(sec)
    if kind == "a":  # airy swell upward, A3 + E4
        air = swish(0.9, 500, 2000, 2500, 8000)
        note = tone(220, sec, 0.03, 0.45, ((1, 1), (2, 0.25))) + tone(329.6, sec, 0.05, 0.4) * 0.45
    elif kind == "b":  # breath falling, C4 + G4, a touch brighter
        air = swish(0.8, 2500, 8000, 600, 2500)
        note = tone(261.6, sec, 0.02, 0.38, ((1, 1), (3, 0.08))) + tone(392, sec, 0.04, 0.32) * 0.4
    else:  # soft felt mallet, E4 with its octave, little air
        air = swish(0.6, 1200, 4000, 1500, 6000) * 0.6
        note = tone(329.6, sec, 0.008, 0.3, ((1, 1), (2, 0.35), (4, 0.05))) + tone(220, sec, 0.02, 0.5) * 0.35
    out = np.zeros(n)
    out[: air.size] += air * 0.5
    out += low(note, 3000) * 0.55
    return fade(out * np.clip(x / 0.012, 0, 1), 0.002, 0.3)


def swish(sec: float, lo0: float, hi0: float, lo1: float, hi1: float) -> np.ndarray:
    """Noise through a moving band, rising then falling in level."""
    n = int(sec * SR)
    out = np.zeros(n)
    blocks = 48
    edges = np.linspace(0, n, blocks + 1).astype(int)
    src = noise(sec)
    for b in range(blocks):
        p = b / (blocks - 1)
        lo, hi = lo0 + (lo1 - lo0) * p, hi0 + (hi1 - hi0) * p
        seg = band(src[max(0, edges[b] - 2000) : edges[b + 1]], lo, hi)
        out[edges[b] : edges[b + 1]] = seg[-(edges[b + 1] - edges[b]) :]
    shape = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.6
    return fade(low(out, 9000) * shape, 0.01, 0.05)


def whoosh() -> np.ndarray:
    """Panned left to right: the transition between two examples."""
    x = swish(0.75, 250, 900, 1500, 6000)
    pan = np.linspace(0, 1, x.size)
    st = room(x, 0.25)[: x.size]
    return np.stack([st[:, 0] * np.cos(pan * np.pi / 2) * 1.3, st[:, 1] * np.sin(pan * np.pi / 2) * 1.3], axis=1)


def brush(sec: float = 3.6) -> np.ndarray:
    """Soft bristle texture with stroke-like level changes."""
    x = t(sec)
    strokes = 0.6 + 0.4 * np.sin(2 * np.pi * 1.7 * x) ** 2
    tex = band(noise(sec), 900, 4200) * strokes
    return fade(tex, 0.15, 0.3)


def hit() -> np.ndarray:
    """Low, warm impact with a soft top: the mark appearing."""
    x = t(2.4)
    f = 48 + 60 * np.exp(-x / 0.05)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(x.size, 0.003, 0.7)
    top = low(noise(2.4), 2500) * env(x.size, 0.002, 0.08) * 0.35
    air = band(noise(2.4), 3000, 10000) * env(x.size, 0.25, 0.6) * 0.06
    return fade(sub + top + air, 0.001, 0.3)


def riser(sec: float = 1.6) -> np.ndarray:
    """Short breath that swells into the next moment."""
    x = swish(sec, 300, 1200, 2500, 9000)
    return x * np.linspace(0.2, 1, x.size) ** 1.5


def pop() -> np.ndarray:
    """A soft, round pop: a caption landing. A short pitch drop on a sine, a breath of air."""
    x = t(0.18)
    f = 330 + 520 * np.exp(-x / 0.018)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(x.size, 0.001, 0.04)
    air = band(noise(0.18), 1500, 5000) * env(x.size, 0.0005, 0.008) * 0.25
    return fade(body + air, 0.0005, 0.04)


def swipe() -> np.ndarray:
    """A short, light whoosh panned left to right: a before/after wipe."""
    x = swish(0.42, 700, 2500, 2500, 8000)
    pan = np.linspace(0.1, 0.9, x.size)
    st = room(x, 0.15)[: x.size]
    return np.stack([st[:, 0] * np.cos(pan * np.pi / 2) * 1.3, st[:, 1] * np.sin(pan * np.pi / 2) * 1.3], axis=1)


def deep() -> np.ndarray:
    """A low, soft hit with a long dark tail, under the privacy line."""
    x = t(3.2)
    f = 41 + 30 * np.exp(-x / 0.08)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(x.size, 0.008, 1.1)
    body = low(noise(3.2), 600) * env(x.size, 0.004, 0.25) * 0.3
    return fade(sub + body, 0.002, 0.6)


# The "use cases" video's effects; write only these with:  python make_sfx.py pop swipe deep
EXTRA = {"pop": (pop, -8), "swipe": (swipe, -8), "deep": (deep, -4)}


# The "Kokon 0.11" video's effects, three variants each to pick by ear (video/public/sfx/011/
# NAME-K.wav, `python make_sfx.py 011`); sound.tsx plays the picked one.
def lock(k: int) -> np.ndarray:
    """A padlock shutting: a short metal click on a low knock (variant 2: two parts, ka-chunk)."""
    def clack(pitch: float, gain: float) -> np.ndarray:
        n = int(0.16 * SR)
        metal = band(rng.standard_normal(n), 2200 * pitch, 7000 * pitch) * env(n, 0.0003, 0.012)
        ring = tone(2900 * pitch, 0.16, 0.0005, 0.03, ((1, 1.0), (1.47, 0.6)))[:n] * 0.25
        knock = np.sin(2 * np.pi * 130 * pitch * t(0.16)) * env(n, 0.001, 0.025) * 0.8
        return (metal + ring + knock) * gain
    if k == 2:
        a, b = clack(1.1, 0.6), clack(0.85, 1.0)
        x = np.zeros(int(0.32 * SR))
        x[: a.size] += a
        i = int(0.085 * SR)
        x[i:i + b.size] += b
        return fade(x, 0.0005, 0.05)
    return fade(clack([1.0, 0.8][k], 1.0), 0.0005, 0.05)


def spinup(k: int, sec: float = 1.8) -> np.ndarray:
    """The computer's fans speeding up: air rising in pitch and level, a hum under it."""
    x = t(sec)
    r = (x / sec) ** [1.2, 0.8, 1.0][k]
    lo, hi = [(300, 1400), (180, 900), (400, 2200)][k]
    air = noise(sec)
    # The band moves up with the speed: two passes, crossfaded.
    a = band(air, lo, lo * 3) * (1 - r) + band(air, hi / 3, hi) * r
    hum = np.sin(2 * np.pi * np.cumsum(55 + 110 * r) / SR) * 0.25
    flutter = 1 + (0.35 * np.sin(2 * np.pi * np.cumsum(6 + 40 * r) / SR) if k == 2 else 0)
    return fade((a * flutter + hum) * (0.15 + 0.85 * r), 0.05, 0.25)


def power(k: int) -> np.ndarray:
    """The computer switched on: a soft button, then a warm swell."""
    x = t(1.4)
    btn = np.zeros(x.size)
    c = click()
    btn[: c.size] += c * 0.8
    f0 = [65.4, 82.4, 98.0][k]
    swell = tone(f0, 1.4, 0.35, 0.5, ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12))) * 0.7
    sweep = band(noise(1.4), 400, 2400) * env(x.size, 0.3, 0.25) * [0.0, 0.25, 0.12][k]
    return fade(btn + swell + sweep, 0.0005, 0.3)


def swipe011(k: int) -> np.ndarray:
    """A request going from the phone to the computer: a short whoosh panned left to right."""
    sec, lo0, hi0, lo1, hi1 = [(0.42, 700, 2500, 2500, 8000), (0.6, 400, 1600, 1600, 6000), (0.36, 900, 3000, 3000, 9000)][k]
    x = swish(sec, lo0, hi0, lo1, hi1)
    if k == 2:
        x = x + tone(2093, sec, 0.15, 0.12) * 0.15
    pan = np.linspace(0.1, 0.9, x.size)
    st = room(x, 0.15)[: x.size]
    return np.stack([st[:, 0] * np.cos(pan * np.pi / 2) * 1.3, st[:, 1] * np.sin(pan * np.pi / 2) * 1.3], axis=1)


V011 = {"lock": (lock, -6), "spinup": (spinup, -10), "power": (power, -6), "travel": (swipe011, -8)}


def save011() -> None:
    global OUT
    out, OUT = OUT, OUT / "011"
    for name, (make, peak) in V011.items():
        for k in range(3):
            save(f"{name}-{k + 1}", make(k), peak)
    OUT = out

if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["011"]:
        raise SystemExit(save011())
    if sys.argv[1:]:
        for name in sys.argv[1:]:
            make, peak = EXTRA[name]
            save(name, make(), peak)
        raise SystemExit
    save("click", click(), -6)
    save("typing", typing(), -6)
    save("send", send(), -6)
    for k in "abc":
        save(f"reveal-{k}", reveal(k), -9)
    save("slider", swish(1.1, 500, 2500, 1200, 5000), -6)
    save("whoosh", whoosh(), -6)
    save("brush", brush(), -6)
    save("hit", hit(), -3)
    save("riser", riser(), -6)
