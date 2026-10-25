"""The text generation reference for the Mac engine (Prompt+): transformers' own Qwen3-VL text decoder layers in fp32
on the CPU, on the GGUF's weights (as `reference_text_encoder.py`), run once on the prompt followed by the tokens the
engine generated (teacher forcing), with the final RMSNorm and the language model head on every generated position.

    kokon-engine-mac check-textgen <encoder.gguf> <dir>     # 1. the engine, greedy: <dir>/engine.json + its logits
    python -I engine-mac/tools/reference_text_generation.py <encoder.gguf> <processor dir> <text_encoder dir> <dir>
    kokon-engine-mac check-textgen <encoder.gguf> <dir>     # 2. the same tokens again, compared with the reference

The prompt is tokenized by the official processor as sd.cpp's template writes it (system turn, user turn, assistant),
and must give the engine's ids. Writes `<dir>/reference_logits.f32` ([generated, vocabulary] fp32: the logits that
predict each generated token) and `<dir>/reference.json` (its own greedy choice at each position).
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_text_encoder import Weights, run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf")
    parser.add_argument("processor")
    parser.add_argument("text_encoder", help="text_encoder/ of the official repository (config.json)")
    parser.add_argument("dir", help="where check-textgen wrote engine.json")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    with open(os.path.join(args.dir, "engine.json")) as f:
        engine = json.load(f)
    config = AutoConfig.from_pretrained(args.text_encoder).text_config
    config._attn_implementation = "eager"
    processor = AutoProcessor.from_pretrained(args.processor)

    # sd.cpp's template (src/pipeline/text.cpp), text only.
    system, prompt = engine["system"], engine["prompt"]
    text = (f"<|im_start|>system\n{system}<|im_end|>\n" if system else "") + f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = processor(text=[text], return_tensors="pt").input_ids[0].tolist()
    if ids != engine["prompt_ids"]:
        sys.exit(f"prompt ids differ from the engine's: {len(ids)} against {len(engine['prompt_ids'])}")
    generated = engine["generated"]
    sequence = ids + generated[:-1]   # each position predicts the next token

    weights = Weights(args.gguf)
    hidden = run(weights, config, sequence)[0]
    norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm.weight.copy_(weights.get("output_norm.weight"))
    head = weights.get("output.weight")
    logits = head @ norm(hidden[len(ids) - 1:]).T          # [vocabulary, generated]
    logits = logits.T.contiguous()
    logits.numpy().astype("float32").tofile(os.path.join(args.dir, "reference_logits.f32"))
    greedy = logits.argmax(dim=1).tolist()
    with open(os.path.join(args.dir, "reference.json"), "w") as f:
        json.dump({"greedy": greedy, "positions": len(generated), "transformers": __import__("transformers").__version__,
                   "torch": torch.__version__}, f, indent=1)
    same = sum(int(a == b) for a, b in zip(greedy, generated))
    print(f"{len(ids)} prompt tokens, {len(generated)} generated; the reference's greedy choice agrees at {same}")
    print("reference text:", processor.tokenizer.decode(greedy))


if __name__ == "__main__":
    main()
