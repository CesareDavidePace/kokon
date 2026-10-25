"""The Qwen-Image 2.1 VAE encoder reference for the Mac engine: diffusers' `AutoencoderKLQwenImage21.encode` in fp32 on
the CPU, on Kokon's VAE file, as the edit pipeline uses it (`QwenImage21Pipeline._encode_vae_image`: the posterior's
mode, then `(latent − mean) / std`). The input is a smooth random RGBA image in [−1, 1] (random low-resolution noise,
upsampled bilinearly, so it has edges and flat areas as photos do). Saves the input, every down block's output, the
mid block's and the normalized latent for `kokon-engine-mac check-vae-encoder`.

    python -I engine-mac/tools/reference_vae_encoder.py <vae.safetensors> <out dir> [--size 256x256]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_vae import decoder_name  # noqa: E402

RESIDUAL = {"residual.0.gamma": "norm1.gamma", "residual.2.weight": "conv1.weight", "residual.2.bias": "conv1.bias",
            "residual.3.gamma": "norm2.gamma", "residual.6.weight": "conv2.weight", "residual.6.bias": "conv2.bias",
            "shortcut.weight": "conv_shortcut.weight", "shortcut.bias": "conv_shortcut.bias"}


def encoder_name(key):
    """Wan 2.2 VAE name -> diffusers AutoencoderKLQwenImage21 name, encoder side; None for the decoder."""
    if key.startswith("conv1."):
        return "quant_conv." + key[6:]
    if not key.startswith("encoder."):
        return None
    parts = key[len("encoder."):].split(".")
    if parts[0] == "conv1":
        return "encoder.conv_in." + parts[1]
    if parts[0] == "head":
        return "encoder.norm_out.gamma" if parts[1] == "0" else "encoder.conv_out." + parts[2]
    if parts[0] == "middle":
        rest = ".".join(parts[2:])
        if parts[1] == "1":
            return "encoder.mid_block.attentions.0." + rest
        return f"encoder.mid_block.resnets.{int(parts[1]) // 2}." + RESIDUAL[rest]
    if parts[0] == "downsamples":
        i, j, rest = parts[1], int(parts[3]), ".".join(parts[4:])
        if j == 2:
            return f"encoder.down_blocks.{i}.downsampler." + rest
        return f"encoder.down_blocks.{i}.resnets.{j}." + RESIDUAL[rest]
    raise KeyError(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vae")
    parser.add_argument("out")
    parser.add_argument("--size", default="256x256", help="WxH, multiples of 16")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    from diffusers import AutoencoderKLQwenImage21
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)
    width, height = (int(v) for v in args.size.split("x"))

    vae = AutoencoderKLQwenImage21()  # the defaults are Qwen-Image 2.1's config
    model = vae.state_dict()
    converted = {}
    for key, value in load_file(args.vae).items():
        name = encoder_name(key) or decoder_name(key)
        if name is not None:
            converted[name] = value.float().reshape(model[name].shape)
    missing, unexpected = vae.load_state_dict(converted, strict=False)
    missing = [k for k in missing if k.startswith("encoder.") or k.startswith("quant_conv")]
    assert not missing and not unexpected, (missing[:10], unexpected[:10])
    vae.eval()

    image = torch.rand(1, 4, 1, height // 16, width // 16) * 2 - 1
    image = F.interpolate(image[:, :, 0], size=(height, width), mode="bilinear", align_corners=False)[:, :, None]
    image = image + 0.05 * torch.randn_like(image)   # some fine texture
    image = image.clamp(-1, 1)

    saved = {}

    def save(name, tensor):
        array = tensor.detach().to(torch.float32).contiguous().numpy()
        array.tofile(os.path.join(args.out, name + ".f32"))
        saved[name] = list(array.shape)

    hooks = [block.register_forward_hook(lambda m, a, o, i=i: save(f"down{i}", o[0, :, 0]))
             for i, block in enumerate(vae.encoder.down_blocks)]
    hooks.append(vae.encoder.conv_in.register_forward_hook(lambda m, a, o: save("conv_in", o[0, :, 0])))
    hooks.append(vae.encoder.mid_block.register_forward_hook(lambda m, a, o: save("mid", o[0, :, 0])))
    latent = vae.encode(image).latent_dist.mode()
    for h in hooks:
        h.remove()
    mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    save("image", image[0, :, 0])
    save("latent", ((latent - mean) / std)[0, :, 0])
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"size": [width, height], "seed": args.seed, "tensors": saved,
                   "torch": torch.__version__}, f, indent=1)
    print(f"{width}×{height}: latent {saved['latent']}")


if __name__ == "__main__":
    main()
