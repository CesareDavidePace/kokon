"""Writes Tests/KokonEngineTests/Fixtures/quant.json: random blocks of each GGUF type the engine reads, and
the values gguf-py (ggml's Python reference) unpacks them to. The Swift tests check the CPU and GPU
unpacking against these values, so the oracle is not the engine's own transcription of ggml.

    python3 -m venv /tmp/gguf-venv && /tmp/gguf-venv/bin/pip install gguf numpy
    /tmp/gguf-venv/bin/python -I engine-mac/tools/make_quant_fixtures.py engine-mac/Tests/KokonEngineTests/Fixtures/quant.json
"""
import json
import sys

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import quants

TYPES = {"Q4_0": Q.Q4_0, "Q8_0": Q.Q8_0, "Q4_K": Q.Q4_K, "Q5_K": Q.Q5_K, "Q6_K": Q.Q6_K,
         "F16": Q.F16, "BF16": Q.BF16}
BLOCK = {"Q4_0": (32, 18), "Q8_0": (32, 34), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
         "F16": (1, 2), "BF16": (1, 2)}
# Where each block keeps its fp16 scales: random bytes there would often be NaN or huge.
HALVES = {"Q4_0": [0], "Q8_0": [0], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208]}


def blocks(name, count, rng):
    values, size = BLOCK[name]
    raw = rng.integers(0, 256, size=(count, size), dtype=np.uint8)
    if name in ("F16", "BF16"):
        f = rng.standard_normal(count * 64).astype(np.float32) * 0.05
        if name == "F16":
            return f.astype(np.float16).view(np.uint8).reshape(-1)
        return (f.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(-1)
    scales = [0.0, -0.0, 1.0, -0.5, 6.1e-5, 3e-8, 0.0123, -0.002, 65504.0]
    for b in range(count):
        for at in HALVES[name]:
            s = scales[b] if b < len(scales) else rng.standard_normal() * 0.01
            raw[b, at:at + 2] = np.frombuffer(np.float16(s).tobytes(), dtype=np.uint8)
    # Extreme quant bits in the first blocks: all zero, all ones.
    if count > 2:
        raw[0, max(HALVES[name]) + 2:] = 0
        raw[1, max(HALVES[name]) + 2:] = 255
    return raw.reshape(-1)


def main(path):
    rng = np.random.default_rng(20261025)
    out = {}
    for name, qtype in TYPES.items():
        count = 12 if BLOCK[name][0] > 1 else 1
        data = blocks(name, count, rng)
        values = quants.dequantize(data, qtype).astype(np.float32).reshape(-1)
        out[name] = {"bytes": data.tobytes().hex(), "values": [float(v) for v in values]}
    with open(path, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1])
