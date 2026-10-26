"""End-to-end text-to-image reference for the Mac engine: QwenImage21Pipeline.__call__'s loop (diffusers 0.42) with
the pieces the other references check, in fp32 on the CPU:

    conditioning of `reference_text_encoder.py` ─┐
    torch.randn noise [1, 1, 64, h, w], packed ──┴ FlowMatchEulerDiscreteScheduler over the checkpoint's sample_sigmas,
                                                   each step `reference_dit.ReferenceDiT` (timestep = t / 1000)
    unpacked, · std + mean ─ `reference_vae.load_vae` decode ─ VaeImageProcessor.postprocess ─ PNG

Saves the noise, every step's velocity and latent, the decoded image and the final pixels for
`kokon-engine-mac check-generate`. 256² takes ~5 min (8 DiT evaluations, each layer's weights unpacked by gguf-py).

**An edit** when the text reference has images (`reference_text_encoder.py --image`): each image as the pipeline hands it
to the VAE (RGBA, `VaeImageProcessor.preprocess` at its own size, which `resize_before_vae=false` keeps), encoded
(`_encode_vae_image`: the mode, (z − mean) / std), packed, and prepended to the target's tokens at every step; the
text reference's `image_pad_mask` places their slots. Saved as `reference<i>`.

    python -I engine-mac/tools/reference_generate.py <dit.gguf> <vae.safetensors> <text reference dir> <repo dir> <out>
        [--grid 16 | --size WxH] [--seed 1] [--dtype float64] [--init in.png [--mask mask.png] [--strength 1]]

**An init image and a mask** (`--init`, `--mask`, `--strength`; brushed edits and extending an image, as Kokon's
sd.cpp runs them for Qwen-Image 2.1, pipeline/image.cpp and diffusion_engine.cpp), written as diffusers' inpaint
pipelines loop (QwenImageInpaintPipeline): the init image encoded as the references are; the schedule cut by sd.cpp's
rule (`t_enc = int(steps · strength)`, one less at `steps`, start at sigma index `steps − t_enc − 1`: one step more
than diffusers' `get_timesteps`); the start `scale_noise(init, σ₀, noise)`; after each step the latent outside the
mask replaced by the init noised to the next σ (the clean init after the last). The mask as sd.cpp reads it: gray / 255
rounded, then the largest value of each 16 × 16 block (`NearestMax`), 1 = repaint. sd.cpp blends the x₀ prediction
with the clean init instead, the same latent algebraically. Saves `init.png`, `mask.png` and `init_latent`.

`--dtype float64`: the DiT and the latents in fp64 (the scheduler still rounds the latent to fp32 each step), the
VAE in fp32. Needed on unsloth's DiT, where torch's fp32 drifts to 7e-5 of fp64 in one evaluation.

`<repo dir>`: the official Qwen/Qwen-Image-2.1-Turbo files (model_index.json: sample_sigmas; scheduler/).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the sibling references
from reference_dit import ReferenceDiT, enable_float64  # noqa: E402
from reference_vae import load_vae  # noqa: E402
from reference_vae_encoder import load_full_vae  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dit")
    parser.add_argument("vae")
    parser.add_argument("text", help="output of reference_text_encoder.py")
    parser.add_argument("repo")
    parser.add_argument("out")
    parser.add_argument("--grid", type=int, default=16, help="latent grid side (16 = a 256² image)")
    parser.add_argument("--size", help="WxH of the image instead (multiples of 32)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--init", help="init image (PNG at the output's size)")
    parser.add_argument("--mask", help="mask (PNG at the output's size; white = repaint)")
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    if dtype == torch.float64:
        enable_float64()
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    def save(name, tensor):
        tensor.detach().to(torch.float32).contiguous().numpy().tofile(os.path.join(args.out, name + ".f32"))

    text_meta = json.load(open(os.path.join(args.text, "meta.json")))
    conditioning = torch.from_numpy(np.fromfile(os.path.join(args.text, "conditioning.f32"), dtype=np.float32))
    conditioning = conditioning.view(1, -1, 4096).to(dtype)
    h = w = args.grid
    if args.size:
        w, h = (int(v) // 16 for v in args.size.split("x"))

    # The edit's condition images (prepare_latents with `images`).
    refs, slots, sizes = [], None, text_meta.get("images") or []
    if sizes:
        slots = [bool(v) for v in text_meta["image_pad_mask"]]
        encoder_vae = load_full_vae(args.vae)
        processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
        mean = torch.tensor(encoder_vae.config.latents_mean).view(1, 64, 1, 1, 1)
        std = torch.tensor(encoder_vae.config.latents_std).view(1, 64, 1, 1, 1)
        for i, (rw, rh) in enumerate(sizes):
            image = Image.open(os.path.join(args.text, f"image{i + 1}.png")).convert("RGBA")
            pixels = processor.preprocess(image, width=rw, height=rh).unsqueeze(2)
            latent = (encoder_vae.encode(pixels).latent_dist.mode() - mean) / std
            packed = latent[0, :, 0].reshape(64, -1).transpose(0, 1)[None]
            save(f"reference{i}", packed[0])
            refs.append((packed.to(dtype), rh // 16, rw // 16))
        del encoder_vae

    # The init image and the mask (sd.cpp's rules, above).
    init, keep = None, None
    if args.init:
        encoder_vae = load_full_vae(args.vae)
        processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
        mean = torch.tensor(encoder_vae.config.latents_mean).view(1, 64, 1, 1, 1)
        std = torch.tensor(encoder_vae.config.latents_std).view(1, 64, 1, 1, 1)
        image = Image.open(args.init).convert("RGBA")
        assert image.size == (w * 16, h * 16), f"the init image must be {w * 16}x{h * 16}"
        image.save(os.path.join(args.out, "init.png"))
        latent = (encoder_vae.encode(processor.preprocess(image, width=w * 16, height=h * 16).unsqueeze(2)).latent_dist.mode() - mean) / std
        init = latent[0, :, 0].reshape(64, -1).transpose(0, 1)[None]
        save("init_latent", init[0])
        init = init.to(dtype)
        del encoder_vae
        if args.mask:
            gray = Image.open(args.mask).convert("L")
            assert gray.size == (w * 16, h * 16), f"the mask must be {w * 16}x{h * 16}"
            gray.save(os.path.join(args.out, "mask.png"))
            pixels = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255)
            blocks = torch.floor(pixels + 0.5).view(h, 16, w, 16).amax(dim=(1, 3))   # std::round, then NearestMax
            keep = (1 - blocks).reshape(1, h * w, 1).to(dtype)                       # 1 = the init stays

    # prepare_latents + _pack_latents
    noise = torch.randn((1, 1, 64, h, w), generator=torch.Generator().manual_seed(args.seed), dtype=torch.float32)
    save("noise", noise[0, 0])
    noise = noise.view(1, 64, h * w).transpose(1, 2).to(dtype)
    latents = noise

    # retrieve_timesteps with the checkpoint's sigmas
    index = json.load(open(os.path.join(args.repo, "model_index.json")))
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        json.load(open(os.path.join(args.repo, "scheduler", "scheduler_config.json"))))
    mu = calculate_shift(latents.shape[1], scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096), scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    scheduler.set_timesteps(sigmas=index["sample_sigmas"], mu=mu)
    start = 0
    if init is not None and args.strength < 1:
        steps = len(scheduler.timesteps)
        t_enc = int(steps * np.float32(args.strength))
        start = steps - (t_enc - 1 if t_enc == steps else t_enc) - 1
    scheduler.set_begin_index(start)
    timesteps = scheduler.timesteps[start:]
    if init is not None:
        latents = scheduler.scale_noise(init, timesteps[:1], noise)   # σ · noise + (1 − σ) · init

    dit = ReferenceDiT(args.dit, dtype=dtype)
    for i, t in enumerate(timesteps):
        velocity = dit.forward(conditioning, latents, h, w, t.expand(1).to(dtype) / 1000, progress=False,
                               refs=refs, slots=slots)
        save(f"velocity{i}", velocity[0])
        latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
        if keep is not None:
            proper = init if i == len(timesteps) - 1 else scheduler.scale_noise(init, timesteps[i + 1:i + 2], noise)
            latents = keep * proper + (1 - keep) * latents
        save(f"latent{i}", latents[0])
        print(f"step {i + 1}: t {float(t) / 1000:.6f}", flush=True)

    # _unpack_latents, denormalize, decode, postprocess
    vae = load_vae(args.vae)
    unpacked = latents.to(torch.float32).transpose(1, 2).reshape(1, 64, 1, h, w)
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    image = vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
    save("image", image[0])
    pil = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64).postprocess(image, output_type="pil")[0]
    pil.save(os.path.join(args.out, "image.png"))
    np.asarray(pil).tofile(os.path.join(args.out, "pixels.u8"))

    meta = {"prompt": text_meta["prompt"], "grid": [h, w], "seed": args.seed, "mode": pil.mode,
            "sigmas": scheduler.sigmas[start:].tolist(), "timesteps": (timesteps / 1000).tolist(),
            "dit": os.path.basename(args.dit), "dtype": args.dtype, "vae": os.path.basename(args.vae),
            "references": sizes, "init": init is not None, "mask": keep is not None, "strength": args.strength}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"image {pil.size} {pil.mode}")


if __name__ == "__main__":
    main()
