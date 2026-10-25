"""The Qwen-Image 2.1 DiT reference for the Mac engine: diffusers' own modules, in fp32 on the CPU, on the
weights of Kokon's GGUF unpacked by gguf-py. One evaluation of a text-to-image case (text prefix, target image),
every stage saved for `kokon-engine-mac check-dit` to compare against.

The whole model in fp32 is 28 GB, more than a 16 GB Mac holds: the script follows
`QwenImage21Transformer2DModel.forward` step by step (diffusers 0.42, default `QwenImage21AttnProcessor`, no KV
cache) with one block module whose weights are swapped layer by layer.

    python -I engine-mac/tools/reference_dit.py <dit.gguf> <out dir> [--grid 16] [--text 24] [--sigma 0.7]
        [--all-layers] [--dtype float64]

`--dtype float64` runs every module in fp64 (diffusers' own `.float()` casts made no-ops): the truth an fp32
engine and the fp32 reference are both measured against, to tell a bug from the weights' own sensitivity.

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


def enable_float64():
    """diffusers casts to fp32 in RoPE, its RMSNorms and the timestep path: `Tensor.float()` keeps fp64 tensors fp64."""
    to_float = torch.Tensor.float
    torch.Tensor.float = lambda self, *a, **k: self if self.dtype == torch.float64 else to_float(self, *a, **k)


class Weights:
    def __init__(self, path, dtype=torch.float32):
        self.dtype = dtype
        self.reader = gguf.GGUFReader(path)
        self.tensors = {t.name.removeprefix(PREFIX): t for t in self.reader.tensors}

    def get(self, name):
        if name not in self.tensors and name.endswith(("img_mlp.gate_layer.weight", "img_mlp.proj.weight")):
            # ComfyUI's layout (unsloth's GGUFs): one gate_up tensor, gate rows first, as diffusers' single-file
            # loader splits it.
            base = name.rsplit("img_mlp.", 1)[0]
            gate, up = self.get(base + "img_mlp.gate_up.weight").chunk(2, dim=0)
            return (gate if name.endswith("gate_layer.weight") else up).contiguous()
        t = self.tensors[name]
        values = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        shape = [int(d) for d in reversed(t.shape)]  # ggml ne order -> torch order
        return torch.from_numpy(values.reshape(shape).copy()).to(self.dtype)

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


class ReferenceDiT:
    """QwenImage21Transformer2DModel.forward, batch 1, no cache, text-to-image, one block module swapped per layer."""

    def __init__(self, path, layers=32, dtype=torch.float32):
        self.w = w = Weights(path, dtype)
        d, self.layers = 4096, layers
        self.img_in = torch.nn.Linear(64, d, bias=False)
        w.load(self.img_in, "img_in.")
        self.txt_in = QwenImage21TextProjection(d, d, eps=1e-6)
        w.load(self.txt_in, "txt_in.")
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(embedding_dim=d)
        w.load(self.time_text_embed, "time_text_embed.")
        self.modulation_linear = torch.nn.Linear(d, 4 * d, bias=False)
        self.modulation_linear.weight.copy_(w.get("modulation.1.weight"))
        self.norm_out = QwenImage21AdaLayerNormContinuous(d, d, eps=1e-6)
        w.load(self.norm_out, "norm_out.")
        self.proj_out = torch.nn.Linear(d, 64, bias=False)
        w.load(self.proj_out, "proj_out.")
        self.rope = QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
        self.block = QwenImage21TransformerBlock(d, 32, 128, mlp_ratio=3, eps=1e-6)
        for m in (self.img_in, self.txt_in, self.time_text_embed, self.modulation_linear, self.norm_out,
                  self.proj_out, self.block):
            m.to(dtype)

    all_layers = False

    def forward(self, encoder, latent, h, w_, timestep, save=None, progress=True):
        """`encoder` [1, T, d], `latent` [1, h * w, 64], `timestep` a [1] tensor (σ); returns the target's velocity
        [1, h * w, 64]. `save(name, tensor)` receives every stage when given."""
        d, text, tokens = 4096, encoder.shape[1], h * w_
        img_shapes = [(1, h, w_)]
        img_mask = torch.tensor([[False] * text + [True] * (tokens // 4)])
        hidden_states = self.img_in(latent)
        encoder_hidden_states = self.txt_in(encoder)
        repeats = torch.where(img_mask, 4, 1)[0]
        image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
        joint = torch.cat([encoder_hidden_states, encoder_hidden_states.new_zeros(1, tokens // 4, d)], dim=1)
        joint = joint.repeat_interleave(repeats, dim=1)
        joint[:, image_pad_mask] = hidden_states
        if save:
            save("joint_in", joint[0])

        rotary_emb = self.rope(img_shapes, image_pad_mask, device=torch.device("cpu"))
        image_ids, target_token_mask = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)
        t = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        temb = self.time_text_embed(t, hidden_states)
        modulation = self.modulation_linear(torch.nn.functional.silu(temb))
        if save:
            save("rope", torch.view_as_real(rotary_emb).reshape(rotary_emb.shape[0], -1))
            save("temb", temb)
            save("modulation", modulation)
        prefix_len = int((~target_token_mask).sum())
        segments = _qwenimage21_prefix_segments(image_ids, prefix_len)

        for layer in range(self.layers):
            self.w.load(self.block, f"transformer_blocks.{layer}.")
            if layer == 0 and save:
                layer0_stages(self.block, joint, modulation, rotary_emb, target_token_mask, save)
            joint = self.block(hidden_states=joint, modulation=modulation, rotary_emb=rotary_emb, attention_mask=None,
                               target_token_mask=target_token_mask, segments=segments)
            if save and (self.all_layers or layer in (0, 1) or layer == self.layers - 1):
                save(f"layer{layer}_out", joint[0])
            if progress:
                print(f"layer {layer} done", flush=True)

        normed = self.norm_out(joint, temb, target_token_mask)
        output = self.proj_out(normed)
        if save:
            save("norm_out", normed[0])
            save("model_out", output[0, prefix_len:])
        return output[:, prefix_len:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("out")
    parser.add_argument("--grid", type=int, default=16, help="target latent grid side (16 = a 256² image)")
    parser.add_argument("--text", type=int, default=24, help="text tokens of the prefix")
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-layers", action="store_true", help="save every layer's output")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    if dtype == torch.float64:
        enable_float64()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)

    saved = {}

    def save(name, tensor):
        array = tensor.detach().to(torch.float32).contiguous().numpy()
        array.tofile(os.path.join(args.out, name + ".f32"))
        saved[name] = list(array.shape)

    # Inputs: the encoder's last layer for T positions (no image slots: text-to-image), the packed latent.
    h = w_ = args.grid
    encoder = torch.randn(1, args.text, 4096) * 4
    latent = torch.randn(1, h * w_, 64)
    save("in_encoder", encoder[0])
    save("in_latent", latent[0])
    dit = ReferenceDiT(args.gguf, layers=args.layers, dtype=dtype)
    dit.all_layers = args.all_layers
    dit.forward(encoder.to(dtype), latent.to(dtype), h, w_, torch.tensor([args.sigma], dtype=dtype), save=save)

    meta = {"grid": [h, w_], "text": args.text, "sigma": args.sigma, "layers": args.layers, "prefix": args.text,
            "seed": args.seed, "dtype": args.dtype, "tensors": saved, "gguf": os.path.basename(args.gguf),
            "diffusers": __import__("diffusers").__version__, "torch": torch.__version__}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
