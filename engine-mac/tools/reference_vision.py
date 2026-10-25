"""The Qwen-Image 2.1 vision tower reference for the Mac engine: transformers' own `Qwen3VLVisionModel` in fp32 on the
CPU, on the weights of Kokon's mmproj GGUF (llama.cpp's layout) unpacked by gguf-py, on an image prepared by the
official processor as diffusers' QwenImage21Pipeline hands it over (`_get_qwen_prompt_embeds`: RGBA composited over
white, then `Qwen3VLProcessor`). The image's sides must be multiples of 32, as the pipeline makes them.

The input is a PNG (`--image`) or, by default, a smooth random image (random low-resolution noise upsampled
bilinearly, so it has edges and flat areas as photos do), saved as `image.png` for the engine to read.

    python -I engine-mac/tools/reference_vision.py <mmproj.gguf> <processor dir> <text_encoder dir> <out dir>
        [--image in.png] [--size 320x224] [--dtype float64]

Writes `<out>/meta.json` (grid, sizes) and one raw little-endian fp32 file per tensor (`<name>.f32`): pixels (the
processor's `pixel_values`), embeddings (patches + positions), block0/8/26 outputs, deepstack0..2, merged.
"""
import argparse
import json
import os

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

BLOCK = {"ln1": "norm1", "ln2": "norm2", "attn_qkv": "attn.qkv", "attn_out": "attn.proj",
         "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2"}
MERGER = {"norm": "norm", "fc1": "linear_fc1", "fc2": "linear_fc2"}


def tensor(reader, name):
    t = reader[name]
    values = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
    return torch.from_numpy(values.reshape([int(d) for d in reversed(t.shape)]).copy())


def state_dict(path, config):
    """mmproj names (llama.cpp's clip.cpp, projector qwen3vl_merger) -> Qwen3VLVisionModel names."""
    reader = {t.name: t for t in gguf.GGUFReader(path).tensors}
    get = lambda name: tensor(reader, name)  # noqa: E731
    # The Conv3d over 2 frames is stored as one 2D kernel per frame.
    state = {"patch_embed.proj.weight": torch.stack([get("v.patch_embd.weight"), get("v.patch_embd.weight.1")], dim=2),
             "patch_embed.proj.bias": get("v.patch_embd.bias"), "pos_embed.weight": get("v.position_embd.weight")}
    for i in range(config.depth):
        for g, h in BLOCK.items():
            for kind in ("weight", "bias"):
                state[f"blocks.{i}.{h}.{kind}"] = get(f"v.blk.{i}.{g}.{kind}")
    for kind in ("weight", "bias"):
        state[f"merger.norm.{kind}"] = get(f"v.post_ln.{kind}")
        state[f"merger.linear_fc1.{kind}"] = get(f"mm.0.{kind}")
        state[f"merger.linear_fc2.{kind}"] = get(f"mm.2.{kind}")
        for j, layer in enumerate(config.deepstack_visual_indexes):
            for g, h in MERGER.items():
                state[f"deepstack_merger_list.{j}.{h}.{kind}"] = get(f"v.deepstack.{layer}.{g}.{kind}")
    return state


def smooth_image(width, height, seed=0):
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 3, height // 16, width // 16, generator=g)
    img = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)[0]
    return Image.fromarray((img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("processor")
    parser.add_argument("text_encoder", help="text_encoder/ of the official repository (config.json)")
    parser.add_argument("out")
    parser.add_argument("--image")
    parser.add_argument("--size", default="320x224")
    parser.add_argument("--dtype", default="float32", help="float64: the model in fp64, to tell torch's fp32 drift apart")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    if args.image:
        image = Image.open(args.image)
    else:
        width, height = (int(v) for v in args.size.split("x"))
        image = smooth_image(width, height)
    assert image.width % 32 == 0 and image.height % 32 == 0, "sides must be multiples of 32"
    image.save(os.path.join(args.out, "image.png"))
    if image.mode == "RGBA":  # as the pipeline: alpha composited over white for the vision encoder
        white = Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.getchannel("A"))
        image = white
    image = image.convert("RGB")

    processor = AutoProcessor.from_pretrained(args.processor)
    inputs = processor.image_processor(images=[image], return_tensors="pt")
    dtype = getattr(torch, args.dtype)
    pixels, grid = inputs.pixel_values.to(dtype), inputs.image_grid_thw

    config = AutoConfig.from_pretrained(args.text_encoder).vision_config
    config._attn_implementation = "eager"
    model = Qwen3VLVisionModel(config).eval()
    model.load_state_dict(state_dict(args.gguf, config), strict=True)
    model = model.to(dtype)

    saved = {}

    def save(name, t):
        array = t.detach().to(torch.float32).contiguous().numpy()
        array.tofile(os.path.join(args.out, name + ".f32"))
        saved[name] = list(array.shape)

    save("pixels", pixels)
    hooks = [model.patch_embed.register_forward_hook(lambda m, a, out: save("patches", out))]
    for i in (0, 8, 26):
        hooks.append(model.blocks[i].register_forward_hook(lambda m, a, out, i=i: save(f"block{i}", out)))
    out = model(pixels, grid_thw=grid, return_dict=True)
    for h in hooks:
        h.remove()
    save("merged", out.pooler_output)
    for j, feature in enumerate(out.deepstack_features):
        save(f"deepstack{j}", feature)

    meta = {"grid": grid[0].tolist(), "dtype": args.dtype, "width": image.width, "height": image.height, "tensors": saved,
            "gguf": os.path.basename(args.gguf), "transformers": __import__("transformers").__version__,
            "torch": torch.__version__}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"grid {grid[0].tolist()}, merged {saved['merged']}")


if __name__ == "__main__":
    main()
