"""The Qwen-Image 2.1 VAE decoder reference for the Mac engine: diffusers' `AutoencoderKLQwenImage21` in fp32 on
the CPU, on Kokon's VAE file. Decodes a random
normalized latent as the pipeline does (`latents * std + mean`, decode, clamp) and saves the input and the
RGBA output for `kokon-engine-mac check-vae`. Kokon's file uses Wan 2.2's names; `decoder_name` maps them to
diffusers' (the same mapping the engine reads them by).

    python -I engine-mac/tools/reference_vae.py <vae.safetensors> <out dir> [--grid 16]
"""
import argparse
import json
import os

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage21
from safetensors.torch import load_file


def decoder_name(key):
    """Wan 2.2 VAE name -> diffusers AutoencoderKLQwenImage21 name, decoder side; None for the encoder."""
    if key.startswith("conv2."):
        return "post_quant_conv." + key[6:]
    if not key.startswith("decoder."):
        return None
    k = key[len("decoder."):]
    residual = {"residual.0.gamma": "norm1.gamma", "residual.2.weight": "conv1.weight", "residual.2.bias": "conv1.bias",
                "residual.3.gamma": "norm2.gamma", "residual.6.weight": "conv2.weight", "residual.6.bias": "conv2.bias",
                "shortcut.weight": "conv_shortcut.weight", "shortcut.bias": "conv_shortcut.bias"}
    parts = k.split(".")
    if parts[0] == "conv1":
        return "decoder.conv_in." + parts[1]
    if parts[0] == "head":
        return "decoder.norm_out.gamma" if parts[1] == "0" else "decoder.conv_out." + parts[2]
    if parts[0] == "middle":
        rest = ".".join(parts[2:])
        if parts[1] == "1":
            return "decoder.mid_block.attentions.0." + rest
        return f"decoder.mid_block.resnets.{int(parts[1]) // 2}." + residual[rest]
    if parts[0] == "upsamples":
        i, j, rest = parts[1], int(parts[3]), ".".join(parts[4:])
        if j == 3:
            return f"decoder.up_blocks.{i}.upsampler." + rest
        return f"decoder.up_blocks.{i}.resnets.{j}." + residual[rest]
    raise KeyError(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vae")
    parser.add_argument("out")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    vae = AutoencoderKLQwenImage21()  # the defaults are Qwen-Image 2.1's config
    model = vae.state_dict()
    converted = {}
    for key, value in load_file(args.vae).items():
        name = decoder_name(key)
        if name is not None:
            converted[name] = value.float().reshape(model[name].shape)
    missing, unexpected = vae.load_state_dict(converted, strict=False)
    missing = [k for k in missing if not k.startswith("encoder.") and "time_conv" not in k and k != "quant_conv.weight"
               and k != "quant_conv.bias"]
    assert not missing, f"decoder weights missing: {missing[:10]}"
    assert not unexpected, f"unexpected: {unexpected[:10]}"
    vae.eval()

    g = args.grid
    latent = torch.randn(1, 64, 1, g, g)  # normalized, as the DiT produces
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    image = vae.decode(latent * std + mean, return_dict=False)[0][:, :, 0]  # [1, 4, H, W] in [-1, 1]

    latent[0, :, 0].contiguous().numpy().astype(np.float32).tofile(os.path.join(args.out, "latent.f32"))
    image[0].contiguous().numpy().astype(np.float32).tofile(os.path.join(args.out, "image.f32"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"grid": g, "image": list(image.shape[1:]), "seed": args.seed}, f)
    print("image", tuple(image.shape), float(image.min()), float(image.max()))


if __name__ == "__main__":
    main()
