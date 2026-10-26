"""Token ids from the processor of the official Qwen-Image 2.1 repository (what diffusers' QwenImage21Pipeline
tokenizes with), for `kokon-engine-mac check-tokenizer` to compare against.

    python -I engine-mac/tools/reference_tokenizer.py <processor dir> <out.json>

The processor dir is `processor/` of Qwen/Qwen-Image-2.1-Turbo (tokenizer.json, tokenizer_config.json, ...).
Writes a list of {"text", "ids"}: the pipeline's full text-to-image prompts and harder strings (accents, CJK,
emoji, digits, whitespace runs, contractions, special tokens inside the text).
"""
import json
import sys

from transformers import AutoProcessor

SYSTEM = "Comprehend and analyze the provided prompt."
TEMPLATE = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"

CASES = [
    "A red fox in the snow",
    " ",
    "a cozy wooden cabin by a lake at sunset, warm light, 35mm photo, f/1.8",
    "Replace the sky with a dramatic stormy sky; keep the house unchanged.",
    "Un gatto che dorme sulla sedia, perché è stanco. Città, più, caffè",
    "Ein schönes Haus — größer, weiß, Straße",
    "一只在雪地里的红狐狸，电影感光线",
    "日本語のテキスト「こんにちは」を看板に書いて",
    "Emoji 🦊🔥👍🏽 and symbols ©®™ ½ ∑ ≤ →",
    "Numbers 1234567890, 3.14159, 1,000,000 and 2026-10-25",
    "Multiple   spaces\tand\ttabs\n\nnew lines\r\nand trailing   ",
    "   leading spaces",
    "I'm sure it's they're we've you'll he'd IT'S I'M",
    "Text on the sign: \"KOKON\" in bold, 'single quotes', `backticks`",
    "café vs café (decomposed), Å vs Å",
    "special <|im_start|> inside and <|image_pad|><think> tags",
    "url https://kokon.app/docs?x=1&y=2#top and email a.b@c.de",
    "naïve résumé coöperate façade jalapeño",
    "Кириллица и ελληνικά και العربية עברית",
    "x" * 300,
    "The quick brown fox jumps over the lazy dog. " * 8,
]


def main():
    processor = AutoProcessor.from_pretrained(sys.argv[1])
    out = []
    for text in CASES:
        out.append({"text": text, "ids": processor.tokenizer.encode(text, add_special_tokens=False)})
        prompt = TEMPLATE.format(text)
        ids = processor(text=[prompt], return_tensors="pt").input_ids[0].tolist()
        out.append({"text": prompt, "ids": ids})
    with open(sys.argv[2], "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=0)
    print(f"{len(out)} cases")


if __name__ == "__main__":
    main()
