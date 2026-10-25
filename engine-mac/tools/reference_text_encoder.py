"""The Qwen-Image 2.1 text encoder reference for the Mac engine: transformers' own Qwen3-VL text decoder layers, in
fp32 on the CPU, on the weights of Kokon's GGUF unpacked by gguf-py, run on the prompt exactly as diffusers'
QwenImage21Pipeline builds it (0.42, `_get_qwen_prompt_embeds`):

    template "<|im_start|>system\\n{sys}<|im_end|>\\n<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
    tokenized by the official processor; conditioning = the last decoder layer's output (before the final RMSNorm),
    without the system message's tokens

With `--image` (an edit, `prompt_template_ti2i`), "<image1><|vision_start|><|image_pad|><|vision_end|>" before the
prompt (" <imageN>…" for more images); the processor expands each `<|image_pad|>` to one token per 32 × 32 pixels, the
vision tower (`reference_vision.py`, on the mmproj GGUF) fills those rows, positions are Qwen3-VL's 3D M-RoPE
(`get_rope_index`), and its three deepstack outputs are added to the image rows after decoder layers 0, 1, 2. Also
saved: `image_pad_mask` (which conditioning rows are image slots) and the vision tower's outputs.

The 8B model in fp32 is 33 GB, more than a 16 GB Mac holds: one decoder layer module, its weights swapped layer
by layer (as `reference_dit.py` does for the DiT). `--sanity` also runs the language model head on a factual
prompt and prints its next token: a check that the GGUF weights are read right, which a comparison of two readers
of the same file cannot be.

    python -I engine-mac/tools/reference_text_encoder.py <encoder.gguf> <processor dir> <text_encoder dir> <out dir>
        [--prompt "..."] [--sanity] [--image in.png --mmproj mmproj.gguf]...

Writes `<out>/meta.json` (prompt, ids, drop) and one raw little-endian fp32 file per tensor (`<name>.f32`).
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import gguf
import numpy as np
import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
)
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_vision import state_dict as vision_state_dict  # noqa: E402

SYSTEM = "Comprehend and analyze the provided prompt."
TEMPLATE = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
IMAGE = "<image{}><|vision_start|><|image_pad|><|vision_end|>"
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


def run(weights, config, ids, save=None, vision=None):
    """The text model's forward on one unpadded prompt; returns the last layer's output (no final norm). `vision`:
    (image rows mask, merged rows, deepstack list, position ids [3, 1, T]) for a prompt with images."""
    x = weights.get("token_embd.weight", rows=ids)[None]
    t = x.shape[1]
    if vision is None:
        position_ids = torch.arange(t).view(1, 1, -1).expand(3, 1, -1)  # text only: the three axes agree
    else:
        visual, merged, deepstack, position_ids = vision
        x[0, visual] = merged
    if save:
        save("embeddings", x[0])
    rope = Qwen3VLTextRotaryEmbedding(config)
    position_embeddings = rope(x, position_ids)
    mask = torch.full((t, t), torch.finfo(torch.float32).min).triu(1)[None, None]
    layer = Qwen3VLTextDecoderLayer(config, 0)
    for index in range(config.num_hidden_layers):
        weights.load_layer(layer, index)
        x = layer(x, position_embeddings=position_embeddings, attention_mask=mask)
        if vision is not None and index < len(deepstack):  # Qwen3VLTextModel._deepstack_process
            x = x.clone()
            x[0, visual] = x[0, visual] + deepstack[index]
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
    parser.add_argument("--image", action="append", default=[], help="a condition image (sides multiples of 32)")
    parser.add_argument("--mmproj", help="the vision tower's GGUF, with --image")
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
    saved = {}

    def save(name, tensor):
        array = tensor.detach().to(torch.float32).contiguous().numpy()
        array.tofile(os.path.join(args.out, name + ".f32"))
        saved[name] = list(array.shape)

    full_config = AutoConfig.from_pretrained(args.text_encoder)
    tower = None

    def with_images(text, images, save=None):
        """Token ids and run()'s `vision` for `text` (a template with its image markers) and its images."""
        nonlocal tower
        flat = []
        for image in images:  # the pipeline: RGBA composited over white for the vision encoder
            if image.mode == "RGBA":
                white = Image.new("RGB", image.size, (255, 255, 255))
                white.paste(image, mask=image.getchannel("A"))
                image = white
            flat.append(image.convert("RGB"))
        inputs = processor(text=[text], images=flat, padding=True, padding_side="left", return_tensors="pt")
        if tower is None:
            vconfig = full_config.vision_config
            vconfig._attn_implementation = "eager"
            tower = Qwen3VLVisionModel(vconfig).eval()
            tower.load_state_dict(vision_state_dict(args.mmproj, vconfig), strict=True)
        out = tower(inputs.pixel_values.to(torch.float32), grid_thw=inputs.image_grid_thw, return_dict=True)
        shim = SimpleNamespace(config=full_config)
        shim.get_vision_position_ids = lambda *a, **k: Qwen3VLModel.get_vision_position_ids(shim, *a, **k)
        position_ids, _ = Qwen3VLModel.get_rope_index(shim, inputs.input_ids, inputs.mm_token_type_ids,
                                                      image_grid_thw=inputs.image_grid_thw,
                                                      attention_mask=inputs.attention_mask)
        if save:
            save("vision_merged", out.pooler_output)
            for j, feature in enumerate(out.deepstack_features):
                save(f"vision_deepstack{j}", feature)
            save("positions", position_ids[:, 0].to(torch.float32))
        ids = inputs.input_ids[0].tolist()
        visual = torch.tensor(ids) == full_config.image_token_id
        return ids, (visual, out.pooler_output, out.deepstack_features, position_ids)

    vision, images = None, []
    if args.image:
        images = [Image.open(p) for p in args.image]
        for i, image in enumerate(images):
            image.save(os.path.join(args.out, f"image{i + 1}.png"))
        template = TEMPLATE.replace("{}", " ".join(IMAGE.format(i + 1) for i in range(len(images))) + "{}")
        ids, vision = with_images(template.format(prompt), images, save)
    else:
        ids = processor(text=[TEMPLATE.format(prompt)], return_tensors="pt").input_ids[0].tolist()

    hidden = run(weights, config, ids, save, vision)
    save("conditioning", hidden[0, drop:])
    meta = {"prompt": prompt, "ids": ids, "drop": drop, "tensors": saved, "gguf": os.path.basename(args.gguf),
            "images": [[i.width, i.height] for i in images],
            "transformers": __import__("transformers").__version__, "torch": torch.__version__}
    if args.image:
        meta["image_pad_mask"] = [int(v) for v in vision[0][drop:].tolist()]

    if args.sanity:
        norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight.copy_(weights.get("output_norm.weight"))

        def next_tokens(text, images=None):
            if images:
                qids, qvision = with_images(text, images)
            else:
                qids, qvision = processor(text=[text], return_tensors="pt").input_ids[0].tolist(), None
            h = run(weights, config, qids, vision=qvision)
            logits = weights.get("output.weight") @ norm(h[0, -1])
            return [processor.tokenizer.decode([i]) for i in torch.topk(logits, 5).indices.tolist()]

        tokens = next_tokens("<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n"
                             "<|im_start|>assistant\nThe capital of France is")
        print("sanity: next tokens after 'The capital of France is':", tokens)
        meta["sanity"] = tokens
        if args.mmproj:
            # A red disk on white: the vision tower's weights are read right if the model sees it.
            yy, xx = np.mgrid[0:320, 0:320]
            disk = np.full((320, 320, 3), 255, np.uint8)
            disk[(yy - 160) ** 2 + (xx - 160) ** 2 < 100 ** 2] = (220, 30, 30)
            tokens = next_tokens("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What color is the shape? "
                                 "Answer in one word.<|im_end|>\n<|im_start|>assistant\n", [Image.fromarray(disk)])
            print("sanity: the shape's color:", tokens)
            meta["vision_sanity"] = tokens

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{len(ids)} tokens, drop {drop}, conditioning {saved['conditioning']}")


if __name__ == "__main__":
    main()
