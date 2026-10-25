"""The Qwen-Image 2.1 text encoder reference for the Mac engine: transformers' own Qwen3-VL text decoder layers, in
fp32 on the CPU, on the weights of Kokon's GGUF unpacked by gguf-py, run on the prompt exactly as diffusers'
QwenImage21Pipeline builds it (0.42, `_get_qwen_prompt_embeds`, text-to-image):

    template "<|im_start|>system\\n{sys}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
    tokenized by the official processor; conditioning = the last decoder layer's output (before the final RMSNorm),
    without the system message's tokens

The 8B model in fp32 is 33 GB, more than a 16 GB Mac holds: one decoder layer module, its weights swapped layer
by layer (as `reference_dit.py` does for the DiT). `--sanity` also runs the language model head on a factual
prompt and prints its next token: a check that the GGUF weights are read right, which a comparison of two readers
of the same file cannot be.

    python -I engine-mac/tools/reference_text_encoder.py <encoder.gguf> <processor dir> <text_encoder dir> <out dir>
        [--prompt "..."] [--sanity]

Writes `<out>/meta.json` (prompt, ids, drop) and one raw little-endian fp32 file per tensor (`<name>.f32`).
"""
import argparse
import json
import os

import gguf
import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
)

SYSTEM = "Comprehend and analyze the provided prompt."
TEMPLATE = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
NAMES = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}


class Weights:
    def __init__(self, path):
        self.reader = gguf.GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}

    def get(self, name, rows=None):
        t = self.tensors[name]
        data = t.data if rows is None else t.data[rows]
        values = gguf.quants.dequantize(data, t.tensor_type).astype(np.float32)
        shape = [int(d) for d in reversed(t.shape)]  # ggml ne order -> torch order
        if rows is not None:
            shape[0] = len(rows)
        return torch.from_numpy(values.reshape(shape).copy())

    def load_layer(self, layer, index):
        state = {}
        for gname, hname in NAMES.items():
            state[hname + ".weight"] = self.get(f"blk.{index}.{gname}.weight")
        layer.load_state_dict(state, strict=True)


def run(weights, config, ids, save=None):
    """The text model's forward on one unpadded prompt; returns the last layer's output (no final norm)."""
    x = weights.get("token_embd.weight", rows=ids)[None]
    if save:
        save("embeddings", x[0])
    t = x.shape[1]
    position_ids = torch.arange(t).view(1, 1, -1).expand(3, 1, -1)  # text only: the three axes agree
    rope = Qwen3VLTextRotaryEmbedding(config)
    position_embeddings = rope(x, position_ids)
    mask = torch.full((t, t), torch.finfo(torch.float32).min).triu(1)[None, None]
    layer = Qwen3VLTextDecoderLayer(config, 0)
    for index in range(config.num_hidden_layers):
        weights.load_layer(layer, index)
        x = layer(x, position_embeddings=position_embeddings, attention_mask=mask)
        if save and index in (0, 1, 17, config.num_hidden_layers - 1):
            save(f"layer{index}_out", x[0])
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("processor")
    parser.add_argument("text_encoder", help="text_encoder/ of the official repository (config.json)")
    parser.add_argument("out")
    parser.add_argument("--prompt", default="A red fox sitting in fresh snow at dawn, soft golden light, photo")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    os.makedirs(args.out, exist_ok=True)
    config = AutoConfig.from_pretrained(args.text_encoder).text_config
    config._attn_implementation = "eager"
    processor = AutoProcessor.from_pretrained(args.processor)
    weights = Weights(args.gguf)

    # QwenImage21Pipeline.__init__ / _get_qwen_prompt_embeds, one prompt, no images.
    system = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    drop = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
    prompt = args.prompt or " "
    ids = processor(text=[TEMPLATE.format(prompt)], return_tensors="pt").input_ids[0].tolist()

    saved = {}

    def save(name, tensor):
        array = tensor.detach().to(torch.float32).contiguous().numpy()
        array.tofile(os.path.join(args.out, name + ".f32"))
        saved[name] = list(array.shape)

    hidden = run(weights, config, ids, save)
    save("conditioning", hidden[0, drop:])
    meta = {"prompt": prompt, "ids": ids, "drop": drop, "tensors": saved, "gguf": os.path.basename(args.gguf),
            "transformers": __import__("transformers").__version__, "torch": torch.__version__}

    if args.sanity:
        question = ("<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n"
                    "<|im_start|>assistant\nThe capital of France is")
        qids = processor(text=[question], return_tensors="pt").input_ids[0].tolist()
        h = run(weights, config, qids)
        norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight.copy_(weights.get("output_norm.weight"))
        logits = weights.get("output.weight") @ norm(h[0, -1])
        top = torch.topk(logits, 5)
        tokens = [processor.tokenizer.decode([i]) for i in top.indices.tolist()]
        print("sanity: next tokens after 'The capital of France is':", tokens)
        meta["sanity"] = tokens

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{len(ids)} tokens, drop {drop}, conditioning {saved['conditioning']}")


if __name__ == "__main__":
    main()
