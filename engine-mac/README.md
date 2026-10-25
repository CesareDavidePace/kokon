# kokon-engine-mac

Kokon's own engine for Apple Silicon (in development). Goal: Qwen-Image 2.1 Turbo with the GPU, the CPU's
matrix unit (AMX) and the Neural Engine all busy for the whole render, reading the same GGUF files Kokon
already downloads for sd.cpp, and answering the same `/sdcpp/v1` API so the service, the app and MCP do not
change. sd.cpp stays the engine on Windows and Linux and the fallback on Macs.

```bash
cd engine-mac
swift build -c release
swift test                                   # no model needed
.build/release/kokon-engine-mac probe        # this Mac's matrix throughput, ~1 min, no model
.build/release/kokon-engine-mac inspect <file.gguf>
.build/release/kokon-engine-mac check-dequant <file.gguf> [tensor...]
```

## What is here

| File | What |
|---|---|
| `GGUF.swift` | GGUF reader; the file is mapped, never copied |
| `Quant.swift` | ggml's unpacking on the CPU (F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K), the reference |
| `Kernels/dequant.metal`, `GPUDequant.swift` | the same on the GPU, to `[n, k]` or `[k, n]`, fp32 or fp16 |
| `NeuralEngine.swift` | a matrix product on the Neural Engine: a small Core ML program written at run time, weight as an input |
| `Probe.swift` | GPU (MPS), AMX (Accelerate) and Neural Engine (Core ML) on Qwen's product shapes, alone and together |

Tests check the CPU and GPU unpacking against gguf-py's values (`tools/make_quant_fixtures.py` writes the
fixture).

## First measurements (M2 Pro 19-core GPU, 16 GB, macOS 27, 2026-10-25, `probe`)

m = 4 096 rows (a 1024² image), TFLOP/s:

| Product k × n | GPU fp32 (MPS) | AMX fp32 | Neural Engine fp16 | All three, cut by rows |
|---|---|---|---|---|
| 4096 × 4096 (q, k, v, out) | 5.1 | 1.7 | 7.4 | 11.0 |
| 4096 × 12288 (MLP gate, proj) | 5.1 | 1.9 | 7.4 | 11.5 |
| 12288 × 4096 (MLP out) | 4.8 | 1.6 | 7.4 | 9.4 |

- The Neural Engine must get its products in pieces of at most 4096 × 4096: a 4096 × 12288 weight in one
  program ran at 2.7.
- The products of a DiT step at 1024² (57 TFLOP) take 11.3 s on the GPU alone and 5.3 s on all three
  (sd.cpp today: 27.4 s per step, all included). The AMX is the one the others wait for: the split by rows needs
  the balancing loop.
- Neural Engine numbers leave out the fp32 ↔ fp16 conversions and scaling it needs; the plan is to have the
  GPU write its inputs straight into the Neural Engine's buffers.

Credits: the Neural Engine program technique and the GPU + AMX + Neural Engine split come from
[Siliconed](https://github.com/gwenn-ha-dev/Siliconed) (Apache-2.0), written again here.
