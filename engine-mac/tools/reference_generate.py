"""End-to-end text-to-image reference for the Mac engine: QwenImage21Pipeline.__call__'s loop (diffusers 0.42) with
the pieces the other references check, in fp32 on the CPU:

    conditioning of `reference_text_encoder.py` ─┐
    torch.randn noise [1, 1, 64, h, w], packed ──┴ FlowMatchEulerDiscreteScheduler over the checkpoint's sample_sigmas,
                                                   each step `reference_dit.ReferenceDiT` (timestep = t / 1000)
    unpacked, · std + mean ─ `reference_vae.load_vae` decode ─ VaeImageProcessor.postprocess ─ PNG

Saves the noise, every step's velocity and latent, the decoded image and the final pixels for
`kokon-engine-mac check-generate`. 256² takes ~5 min (8 DiT evaluations, each layer's weights unpacked by gguf-py).

    python -I engine-mac/tools/reference_generate.py <dit.gguf> <vae.safetensors> <text reference dir> <repo dir> <out>
        [--grid 16] [--seed 1] [--dtype float64]

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
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # the sibling references
from reference_dit import ReferenceDiT, enable_float64  # noqa: E402
from reference_vae import load_vae  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dit")
    parser.add_argument("vae")
    parser.add_argument("text", help="output of reference_text_encoder.py")
    parser.add_argument("repo")
    parser.add_argument("out")
    parser.add_argument("--grid", type=int, default=16, help="latent grid side (16 = a 256² image)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
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

    # prepare_latents + _pack_latents
    noise = torch.randn((1, 1, 64, h, w), generator=torch.Generator().manual_seed(args.seed), dtype=torch.float32)
    save("noise", noise[0, 0])
    latents = noise.view(1, 64, h * w).transpose(1, 2).to(dtype)

    # retrieve_timesteps with the checkpoint's sigmas
    index = json.load(open(os.path.join(args.repo, "model_index.json")))
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        json.load(open(os.path.join(args.repo, "scheduler", "scheduler_config.json"))))
    mu = calculate_shift(latents.shape[1], scheduler.config.get("base_image_seq_len", 256),
                         scheduler.config.get("max_image_seq_len", 4096), scheduler.config.get("base_shift", 0.5),
                         scheduler.config.get("max_shift", 1.15))
    scheduler.set_timesteps(sigmas=index["sample_sigmas"], mu=mu)
    scheduler.set_begin_index(0)

    dit = ReferenceDiT(args.dit, dtype=dtype)
    for i, t in enumerate(scheduler.timesteps):
        velocity = dit.forward(conditioning, latents, h, w, t.expand(1).to(dtype) / 1000, progress=False)
        save(f"velocity{i}", velocity[0])
        latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
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
            "sigmas": scheduler.sigmas.tolist(), "timesteps": (scheduler.timesteps / 1000).tolist(),
            "dit": os.path.basename(args.dit), "dtype": args.dtype, "vae": os.path.basename(args.vae)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"image {pil.size} {pil.mode}")


if __name__ == "__main__":
    main()
