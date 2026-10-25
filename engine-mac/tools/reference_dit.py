"""The Qwen-Image 2.1 DiT reference for the Mac engine: diffusers' own modules, in fp32 on the CPU, on the
weights of Kokon's GGUF unpacked by gguf-py. One evaluation of a text-to-image case (text prefix, target image),
every stage saved for `kokon-engine-mac check-dit` to compare against.

The whole model in fp32 is 28 GB, more than a 16 GB Mac holds: the script follows
`QwenImage21Transformer2DModel.forward` step by step (diffusers 0.42, default `QwenImage21AttnProcessor`, no KV
cache) with one block module whose weights are swapped layer by layer.

    python -I engine-mac/tools/reference_dit.py <dit.gguf> <out dir> [--grid 16] [--text 24] [--sigma 0.7]

Writes `<out>/meta.json` and one raw little-endian fp32 file per tensor (`<name>.f32`).
"""
import argparse
import json
import math
import os

import gguf
import numpy as np
import torch
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AdaLayerNormContinuous,
    QwenImage21Rope,
    QwenImage21TextProjection,
    QwenImage21TimestepProjEmbeddings,
    QwenImage21Transformer2DModel,
    QwenImage21TransformerBlock,
    _qwenimage21_prefix_segments,
)

PREFIX = "model.diffusion_model."


class Weights:
    def __init__(self, path):
        self.reader = gguf.GGUFReader(path)
        self.tensors = {t.name.removeprefix(PREFIX): t for t in self.reader.tensors}

    def get(self, name):
        t = self.tensors[name]
        values = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        shape = [int(d) for d in reversed(t.shape)]  # ggml ne order -> torch order
        return torch.from_numpy(values.reshape(shape).copy())

    def load(self, module, prefix):
        state = {name: self.get(prefix + name) for name, _ in module.named_parameters()}
        module.load_state_dict(state, strict=True)


def layer0_stages(block, x, modulation, rotary_emb, target_token_mask, save):
    """QwenImage21TransformerBlock.forward and its processor, unrolled, saving each stage (t2i: one text segment)."""
    from diffusers.models.transformers.transformer_qwenimage21 import apply_rotary_emb_qwen
    mod1, mod2 = modulation.chunk(2, dim=-1)
    normed, gate1 = block._modulate(block.img_norm1(x), mod1, target_token_mask)
    save("l0_norm1", normed[0])
    attn = block.attn
    q = attn.norm_q(attn.to_q(normed).unflatten(-1, (-1, 128)))
    k = attn.norm_k(attn.to_k(normed).unflatten(-1, (-1, 128)))
    v = attn.to_v(normed).unflatten(-1, (-1, 128))
    q = apply_rotary_emb_qwen(q, rotary_emb, use_real=False)
    k = apply_rotary_emb_qwen(k, rotary_emb, use_real=False)
    save("l0_q", q[0].flatten(1))
    save("l0_k", k[0].flatten(1))
    save("l0_v", v[0].flatten(1))
    out = attn(hidden_states=normed, rotary_emb=rotary_emb,
               segments=[(0, int((~target_token_mask).sum()), True)])
    save("l0_attn_out", out[0])
    x = x + gate1.tanh() * out
    save("l0_res1", x[0])
    normed2, gate2 = block._modulate(block.img_norm2(x), mod2, target_token_mask)
    save("l0_norm2", normed2[0])
    save("l0_mlp", block.img_mlp(normed2)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("out")
    parser.add_argument("--grid", type=int, default=16, help="target latent grid side (16 = a 256² image)")
    parser.add_argument("--text", type=int, default=24, help="text tokens of the prefix")
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)
    w = Weights(args.gguf)
    d, heads, head_dim, layers = 4096, 32, 128, args.layers

    saved = {}

    def save(name, tensor):
        array = tensor.detach().to(torch.float32).contiguous().numpy()
        array.tofile(os.path.join(args.out, name + ".f32"))
        saved[name] = list(array.shape)

    # Inputs: the encoder's last layer for T positions (no image slots: text-to-image), the packed latent.
    h = w_ = args.grid
    tokens = h * w_
    encoder = torch.randn(1, args.text, d) * 4
    latent = torch.randn(1, tokens, 64)
    timestep = torch.tensor([args.sigma], dtype=torch.float32)
    save("in_encoder", encoder[0])
    save("in_latent", latent[0])
    img_shapes = [(1, h, w_)]
    img_mask = torch.tensor([[False] * args.text + [True] * (tokens // 4)])

    img_in = torch.nn.Linear(64, d, bias=False)
    w.load(img_in, "img_in.")
    txt_in = QwenImage21TextProjection(d, d, eps=1e-6)
    w.load(txt_in, "txt_in.")
    time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=d)
    w.load(time_text_embed, "time_text_embed.")
    modulation_linear = torch.nn.Linear(d, 4 * d, bias=False)
    modulation_linear.weight.copy_(w.get("modulation.1.weight"))
    norm_out = QwenImage21AdaLayerNormContinuous(d, d, eps=1e-6)
    w.load(norm_out, "norm_out.")
    proj_out = torch.nn.Linear(d, 64, bias=False)
    w.load(proj_out, "proj_out.")
    rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    block = QwenImage21TransformerBlock(d, heads, head_dim, mlp_ratio=3, eps=1e-6)

    # ── QwenImage21Transformer2DModel.forward, batch 1, no cache ──
    hidden_states = img_in(latent)
    encoder_hidden_states = txt_in(encoder)
    repeats = torch.where(img_mask, 4, 1)[0]
    image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
    joint = torch.cat([encoder_hidden_states, encoder_hidden_states.new_zeros(1, tokens // 4, d)], dim=1)
    joint = joint.repeat_interleave(repeats, dim=1)
    joint[:, image_pad_mask] = hidden_states
    save("joint_in", joint[0])

    rotary_emb = rope(img_shapes, image_pad_mask, device=torch.device("cpu"))
    save("rope", torch.view_as_real(rotary_emb).reshape(rotary_emb.shape[0], -1))
    image_ids, target_token_mask = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)
    t = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
    temb = time_text_embed(t, hidden_states)
    modulation = modulation_linear(torch.nn.functional.silu(temb))
    save("temb", temb)
    save("modulation", modulation)
    prefix_len = int((~target_token_mask).sum())
    segments = _qwenimage21_prefix_segments(image_ids, prefix_len)

    for layer in range(layers):
        w.load(block, f"transformer_blocks.{layer}.")
        if layer == 0:
            layer0_stages(block, joint, modulation, rotary_emb, target_token_mask, save)
        joint = block(hidden_states=joint, modulation=modulation, rotary_emb=rotary_emb, attention_mask=None,
                      target_token_mask=target_token_mask, segments=segments)
        if layer in (0, 1) or layer == layers - 1:
            save(f"layer{layer}_out", joint[0])
        print(f"layer {layer} done", flush=True)

    normed = norm_out(joint, temb, target_token_mask)
    save("norm_out", normed[0])
    output = proj_out(normed)
    save("model_out", output[0, prefix_len:])

    meta = {"grid": [h, w_], "text": args.text, "sigma": args.sigma, "layers": layers, "prefix": prefix_len,
            "seed": args.seed, "tensors": saved, "gguf": os.path.basename(args.gguf),
            "diffusers": __import__("diffusers").__version__, "torch": torch.__version__}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
