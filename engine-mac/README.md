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
.build/release/kokon-engine-mac probe-attention   # the target's attention at 1024² on the GPU, Neural Engine, CPU
.build/release/kokon-engine-mac inspect <file.gguf>
.build/release/kokon-engine-mac check-dequant <file.gguf> [tensor...]
.build/release/kokon-engine-mac generate <encoder.gguf> <dit.gguf> <vae.safetensors> "a red fox in the snow" \
    [--size 1024x1024] [--seed 42] [--out out.png]
```

## What is here

| File | What |
|---|---|
| `GGUF.swift` | GGUF reader; the file is mapped, never copied |
| `Quant.swift` | ggml's unpacking on the CPU (F32, F16, BF16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K), the reference |
| `Kernels/dequant.metal`, `GPUDequant.swift` | the same on the GPU, to `[n, k]` or `[k, n]`, fp32 or fp16 |
| `NeuralEngine.swift` | a matrix product on the Neural Engine: a small Core ML program written at run time, weight as an input |
| `Probe.swift` | GPU (MPS), AMX (Accelerate) and Neural Engine (Core ML) on Qwen's product shapes, alone and together |
| `QwenSequence.swift` | the joint text/image sequence, block-causal segments and RoPE positions |
| `QwenDiT.swift`, `Kernels/dit.metal`, `Attention.swift`, `Arena.swift` | the DiT: modulation on the CPU, everything per token on the GPU |
| `Coexecution.swift`, `Kernels/ane.metal` | the AMX and the Neural Engine beside the GPU on the DiT's products; the timeline of each split operation |
| `CPUProduct.swift` | the AMX's products: BNNS, fp32 or fp16 operands |
| `Balancer.swift` | the balancing loop: each evaluation's timeline sets the next one's rows and heads per unit |
| `Kernels/attention.metal` | fused block-causal attention |
| `AttentionSplit.swift`, `CPUAttention.swift` | the target's attention cut by heads: GPU, CPU (AMX + cores), optionally the Neural Engine |
| `AttentionProbe.swift` | `probe-attention`: the target's attention on each unit alone |
| `QwenVAE.swift`, `Safetensors.swift` | the VAE decoder as one MPSGraph, from Kokon's VAE file (Wan 2.2 names) |
| `QwenTokenizer.swift` | Qwen's byte-level BPE tokenizer, read from the encoder GGUF's metadata |
| `QwenTextEncoder.swift`, `Kernels/text.metal` | the text encoder: Qwen3-VL-8B's text decoder on the GPU, diffusers' prompt template |
| `QwenPipeline.swift` | text to image: noise, Turbo's 8 flow-Euler steps (`planner::TURBO_SIGMAS`), VAE, PNG |

Tests check the CPU and GPU unpacking against gguf-py's values (`tools/make_quant_fixtures.py` writes the
fixture), and the sequence layout against diffusers' rules worked out by hand.

**The DiT against diffusers.** `tools/reference_dit.py` runs diffusers' own Qwen-Image 2.1 modules in fp32 on the
CPU, on Kokon's GGUF weights unpacked by gguf-py, and saves every stage; `check-dit` compares the engine with it:

```bash
python3 -m venv ~/kokon-bench/venv-oracle && ~/kokon-bench/venv-oracle/bin/pip install torch diffusers transformers gguf numpy
~/kokon-bench/venv-oracle/bin/python -I tools/reference_dit.py <dit.gguf> ~/kokon-bench/ref/dit-t2i-256   # ~30 s
.build/release/kokon-engine-mac check-dit <dit.gguf> ~/kokon-bench/ref/dit-t2i-256
.build/release/kokon-engine-mac time-dit <dit.gguf> 64 32        # 1024², 32 text rows, plus a profile
```

256² text-to-image, σ 0.7, 2026-10-25: every stage within 2e-5 relative of diffusers fp32 (velocity 6e-6).
unsloth's Qwen-Image 2.1 Turbo Q4_K_M (the file Kokon will ship with this engine; ComfyUI layout, each block's MLP
`gate_up` fused, read as two views): **check it against an fp64 reference** (`--dtype float64 --all-layers`, ~1 min).
On this file torch's own fp32 run drifts from layer 6 on, to 7.3e-5 of fp64 at the velocity, while the engine stays
within 3.9e-6 (every layer within 6.3e-6); on the old uniform Q4_K file both are within 4.1e-6 of fp64. The 7.2e-5
once measured against the fp32 reference was the reference's error, not the engine's.

**The text encoder against transformers.** The reference tokenizes with the processor of the official
Qwen/Qwen-Image-2.1-Turbo repository (`processor/` and `text_encoder/config.json`, a few MB) and runs transformers'
Qwen3-VL decoder layers in fp32 on the GGUF's weights, one layer at a time; `--sanity` checks the language model
head answers " Paris" to a factual prompt (it does), so the weights are read right and not only read alike.

```bash
~/kokon-bench/venv-oracle/bin/pip install torchvision huggingface_hub   # the processor needs torchvision
~/kokon-bench/venv-oracle/bin/python -I tools/reference_tokenizer.py <processor dir> ~/kokon-bench/ref/text/tokenizer.json
.build/release/kokon-engine-mac check-tokenizer <encoder.gguf> ~/kokon-bench/ref/text/tokenizer.json
~/kokon-bench/venv-oracle/bin/python -I tools/reference_text_encoder.py <encoder.gguf> <processor dir> <text_encoder dir> ~/kokon-bench/ref/text/fox --sanity   # ~50 s
.build/release/kokon-engine-mac check-text <encoder.gguf> ~/kokon-bench/ref/text/fox
```

2026-10-25: tokenizer identical on 42 cases (accents, CJK, emoji, decomposed Unicode, special tokens in the text);
conditioning within 5.1e-6 relative (37 tokens) and 5.6e-6 (144 tokens). One encode: 0.45 s (37 tokens), 0.8 s (144).

**Text to image against diffusers.** `tools/reference_generate.py` runs QwenImage21Pipeline's loop with the
references above (diffusers' scheduler on the checkpoint's sigmas, the DiT layer by layer, the VAE, the image
processor) and saves its noise; `check-generate` starts the engine from that noise:

```bash
~/kokon-bench/venv-oracle/bin/python -I tools/reference_generate.py <dit.gguf> <vae.safetensors> ~/kokon-bench/ref/text/fox <repo dir> ~/kokon-bench/ref/generate-256   # ~5 min
.build/release/kokon-engine-mac check-generate <encoder.gguf> <dit.gguf> <vae.safetensors> ~/kokon-bench/ref/generate-256 [~/kokon-bench/ref/text/fox]
.build/release/kokon-engine-mac check-steps <dit.gguf> ~/kokon-bench/ref/generate-256 ~/kokon-bench/ref/text/fox
```

`check-generate` with the text reference starts from its conditioning instead of the engine's encoder;
`check-steps` runs each step alone on the reference's own latent, so one evaluation's error is not mixed with what
the trajectory makes of the steps before.

Unsloth's DiT + the texture-fix VAE, 256², fp64 reference (`--dtype float64`), 2026-10-25: each step alone within
1.4–2.9e-6 of fp64 except step 4 (σ 0.93) at 6.6e-5, an ill-conditioned evaluation (torch's fp32: 2.7e-5 there).
Along the trajectory the final latent ends within 5.8e-5 (torch fp32 1.2e-5), the image within 7.6e-5, 782 of
262 144 8-bit values off by 1; with the engine's own text encoder (5e-6 off) the same order. These shift by a few
times with any change of rounding order (the old flash kernel: 1.9e-5), as torch's fp32 shifts from fp64.

Old uniform Q4_K DiT + original VAE, 256², fp32 reference: each step's latent within 6.7e-6 relative, the image within 1.0e-5, 97 of 262 144 8-bit values
differ by 1. **First 1024² image (M2 Pro, `generate`): 100.7 s in all**, text encoder 1.7 s, 8 steps of 11.4–12.9 s,
VAE 5.4 s with its graph build; peak memory 4.1 GB (Kokon's sd.cpp: ~4.5 min).

**The seed's noise is sd.cpp's** (`Noise.swift`: its default Philox generator, the latent filled in the same
order), so a seed starts from the same latent as in Kokon today. Bit for bit against sd.cpp's own code compiled as
Kokon's macOS engine is (Apple clang, Release: `x · a + b` fused), 8 seeds × 263 144 values; without the fused
multiply-add 0.7 % of the values differ in the last bit:

```bash
clang++ -std=c++17 -O3 -I <stable-diffusion.cpp>/src tools/philox_reference.cpp -o philox && ./philox > philox.bin
.build/release/kokon-engine-mac check-noise philox.bin
```

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

**One DiT evaluation at 1024² (32 text rows), GPU only: 16.2 s** (sd.cpp: 27.4 s): products 11.6 s, attention
(MPSGraph) 4.1 s, unpacking 0.2 s, the rest 0.3 s. Every product goes to MPS on a multiple of 64 rows: one product
of 32 rows (the text projection) slowed all later ones to 18.5 s.

**With the AMX and the Neural Engine on the products** (`Coexecution.swift`; rows GPU 1856 / AMX 256 / Neural
Engine 2048 at 1024²): **11.3 s per evaluation at 1024²**, 8.8 s at 896². The GPU packs the Neural Engine's fp16
operands straight into its IOSurfaces and unpacks its output, so the CPU only runs the AMX. Accuracy at 896²
against diffusers fp32: the velocity within 3.6e-4 relative (GPU alone 4.4e-6); running the model in bf16, as
most tools do, deviates by roughly 1e-2. 
**Attention** is a fused (flash) kernel of our own (`Kernels/attention.metal`, fp32, online softmax in registers,
block-causal mask, 64 keys a tile): 120 ms for the target's 32 heads at 1024² alone (2.3 TFLOP/s; the first version
145 ms, MPSGraph ~128 ms) and without MPSGraph's score matrix (2.2 GB at 1 MP, ~34 GB at 4 MP).
`KOKON_ATTENTION=graph` keeps MPSGraph for comparison.

**The target's attention is cut by heads** (`AttentionSplit.swift`) as the products are by rows: GPU heads in the
flash kernel, CPU heads in `CPUAttention` (AMX for the two products, vvexpf on 8 threads for the softmax, fp32,
written straight into the GPU's output). `probe-attention`, ms per head alone: GPU 3.8, CPU 6.6, Neural Engine 5.1
in programs of 4 heads × 1 024 queries (head 0 within 2.2e-3 of exact: fp16). Together at 1024², per layer: GPU
alone 145 ms; GPU 19 / CPU 13 91 ms at no cost in accuracy; with Neural Engine heads ~1.5 % more, because the GPU's
heads slow from 4.5 to 6.3–7.0 ms beside it (memory and cache), for 2.5× the velocity error (9.3e-4 against 3.6e-4
at 896²): Neural Engine heads only with `KOKON_ANE_ATTENTION=1`.

**The balancing loop** (`Balancer.swift`): each evaluation's timeline (when the GPU released each split operation,
when each unit finished its share) gives each unit's rate per kind of operation beside the others, and the next
evaluation's shares follow: the AMX's rows per product shape (steps of 64, remembering measured counts: its rate
is not linear in rows), the Neural Engine's rows (steps of 256, a program set per count), the heads. On the M2 Pro at
1024² it moves from 1856 / 256 / 2048 rows and 19 / 13 heads to about 1728 / 320–448 / 2048 and 21 / 11 in two
evaluations; every product's units then finish within ~0.3 ms of each other (before: the GPU 3–15 ms after the
AMX). `KOKON_BALANCE=0` keeps the starting shares; `KOKON_HEADS=g,a,c` fixes the heads.

**One evaluation at 1024², all three units, unsloth's DiT: 8.7–8.9 s** (start of this pass 11.6 s): products ~5.3 s
(≈11 TFLOP/s across the units, the probe's ceiling), attention 32 × 78 ms, the GPU alone between operations 0.6 s
(`time-dit` prints it by transition). The Neural Engine's per-column weight scales are computed once per weight, the
next product's weight is unpacked and packed for the Neural Engine while the current one runs (between operations
1.12 → 0.60 s, the evaluation ~0.1 s faster: the units were already full), and the modulation of all of the
pipeline's timesteps is computed once (it was 0.2 s of CPU per evaluation). The balanced shares are remembered per
Mac and shape (`~/Library/Caches/kokon-engine-mac/balance.json`), so the first step starts balanced.

**1024² image, M2 Pro, unsloth's DiT + texture-fix VAE (`generate`), 2026-10-25: 79.9 s** (text encoder 2.0 s, step 1
10.2 s, steps 2–8 8.8–9.1 s, VAE 5.2 s); at the start of this pass 100.2 s; Kokon's sd.cpp ~4.5 min. With
fp16 operands in the flash kernel, **the default since 2026-10-25** (`KOKON_ATTENTION=fp32` for the checks): 76.0 s,
the velocity within 6.0e-4 of fp64 at 256² (fp32 kernel 1.0e-5; bf16 tools ~1e-2); the two 1024² images differ by 0.25
levels on average, 0.18 % of values by more than 8. The accuracy figures above are the fp32 kernel's.

**fp16 operands in the products, the default since 2026-10-25** (`KOKON_WEIGHTS=fp32` and `KOKON_ACTIVATIONS=fp32`
for the checks): the weights are unpacked to fp16 (half the bytes written and read by every unit), the GPU casts
the GPU's and the AMX's rows of each input to fp16 once (shared by q, k, v and by gate, proj), MPS multiplies fp16 ×
fp16 into fp32 and the AMX runs BNNS instead of `cblas_sgemm` (alone, on its shares: fp32 1.3–1.9× faster, fp16
inputs 2–2.7×). The three units share one limit when they run together (memory or power: BNNS in fp32 alone moved
an evaluation only 8.54 → 8.3–8.5 s), so halving the bytes is what pays. One evaluation at 1024²: **8.54 → 7.45–7.55
s** (fp16 weights alone 8.0–8.1). The velocity at 896² against diffusers fp32 (old Q4_K file, `dit-t2i-896`): 7.0e-4,
against 5.1e-4 before (fp16 weights alone 5.7e-4; the Neural Engine's rows were already fp16); 256² end to end
(GPU only, fp16 weights and attention) against fp64: image within 2.4e-3, at most 13 levels off (fp32 path 7.6e-5).
The 1024² image against the previous precision: 0.05 levels on average, at most 6. **1024² image: 69.8 s** (text
encoder 2.0 s, step 1 8.4 s, steps 2–8 7.4–8.0 s, VAE 6.0 s).

MFA's GEMM (libnnc, 48 × 48 blocks, 2 × 2 SIMD groups) was tried against MPS: its async copies are an AIR intrinsic
that macOS 27's run-time compiler refuses (libnnc ships prebuilt metallibs for it), and without them it ran at
3.6–4.0 TFLOP/s against MPS's 5.1–5.6 at 1472 × 4096 × 4096, so MPS stays.

**VAE decoder** (`check-vae`, against `tools/reference_vae.py`, diffusers fp32): within 1.1e-5; 1024² in 2.75 s. Its
graph builds and compiles in 0.3 s, but its first run at a size costs ~3.7 s more, GPU time (~1 s of CPU): it cannot
hide under the DiT, so the pipeline keeps its VAE between images and a long-lived engine pays it once per size.
fp16 convolutions (norms and attention kept fp32) were wrong (2.7e-1, an overflow) and 5.7× slower (MPSGraph tries to
move them to the Neural Engine): the VAE stays fp32.

Credits: the Neural Engine program technique and the GPU + AMX + Neural Engine split come from
[Siliconed](https://github.com/gwenn-ha-dev/Siliconed) (Apache-2.0), written again here.
