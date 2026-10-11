"""Step 2: caption every image in the manifest.

    python caption.py --manifest work/manifest.jsonl --out work/captioned.jsonl
    python caption.py ... --vlm D:\\models\\Huihui-Qwen3-VL-8B-Instruct-abliterated
    python caption.py ... --mode tags          # no VLM, captions built from the tags

The VLM writes a natural-language description (Iris's text encoder was trained on
those); the image's booru tags are passed in as hints and appended to the caption.
Any caption that mentions a child or a minor is dropped (safety.py), whatever the
tags said. Resumable: rows already in --out are skipped.
"""

import argparse
import json
import re
import time
from pathlib import Path

import common  # noqa: F401  (paths, AOTriton)
import torch
from PIL import Image

from common import TEXT_ENCODER, read_jsonl
from safety import caption_blocked

# tags that say nothing about the picture's content
META = {"highres", "absurdres", "lowres", "commentary", "commentary_request", "english_commentary", "translated",
        "bad_id", "bad_pixiv_id", "bad_twitter_id", "md5_mismatch", "artist_name", "signature", "watermark",
        "web_address", "patreon_username", "twitter_username", "censored", "uncensored", "jpeg_artifacts"}

PROMPT = (
    "Write a caption for this anime illustration to train an image generator. In 2 or 3 sentences (under 90 "
    "words), describe plainly and literally: the adult characters (always call them woman, women, man or men), "
    "their appearance, clothing or nudity, pose and what they are doing, any sexual activity, the setting, "
    "lighting and art style. Use direct, explicit words for anatomy and acts where they are visible. "
    "Do not guess names. Helpful tags for this image: {tags}"
)
MAX_TAGS = 30
# danbooru's people-count tags, said in adult words
COUNT_TAGS = {"1girl": "one woman", "2girls": "two women", "3girls": "three women", "4girls": "four women",
              "multiple_girls": "multiple women", "1boy": "one man", "2boys": "two men", "3boys": "three men",
              "multiple_boys": "multiple men", "1other": "one person", "solo": "solo"}
# whole words only (letters may not touch either side); "girl_on_top" style tags arrive with spaces
ADULT_WORDS = [(w, r) for w, r in (("girls", "women"), ("girl", "woman"), ("boys", "men"), ("boy", "man"))]
_EDGE = ('(?<![A-Za-z])', '(?![A-Za-z])')


def adult_words(text: str) -> str:
    for word, adult in ADULT_WORDS:
        text = re.sub(_EDGE[0] + word + _EDGE[1], adult, text, flags=re.I)
    return text


def tag_text(tags) -> str:
    out = [COUNT_TAGS.get(t, t.replace("_", " ")) for t in tags if t.lower() not in META]
    return ", ".join(adult_words(t) for t in out[:MAX_TAGS])


def load_vlm(path: str, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device).train(False)
    return processor, model


@torch.no_grad()
def vlm_caption(processor, model, image: Image.Image, tags: str, device, max_side: int) -> str:
    image = image.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": PROMPT.format(tags=tags or "none")}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=140, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="work/manifest.jsonl")
    ap.add_argument("--out", default="work/captioned.jsonl")
    ap.add_argument("--mode", choices=["vlm", "tags"], default="vlm")
    ap.add_argument("--vlm", default=str(TEXT_ENCODER), help="Qwen3-VL folder (an abliterated one avoids refusals)")
    ap.add_argument("--max-side", type=int, default=768, help="image size the VLM sees")
    ap.add_argument("--prefix", default="anime illustration. ", help="put in front of every caption")
    args = ap.parse_args()

    rows = read_jsonl(args.manifest)
    out = Path(args.out)
    done = {r["path"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in rows if r["path"] not in done]
    print(f"{len(rows)} images, {len(done)} already captioned, {len(todo)} to go")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "vlm" and todo:
        processor, model = load_vlm(args.vlm, device)
    dropped, t0 = 0, time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo):
            tags = tag_text(row.get("tags", []))
            if args.mode == "vlm":
                with Image.open(row["path"]) as im:
                    text = vlm_caption(processor, model, im, tags, device, args.max_side)
                if text.lower().startswith(("i can't", "i cannot", "i'm sorry", "sorry")):
                    text = ""  # refusal: fall back to the tags alone
            else:
                text = ""
            caption = args.prefix + " ".join(x for x in (adult_words(text), tags) if x)
            if reason := caption_blocked(caption):
                dropped += 1
                print(f"  dropped ({reason}): {row['path']}")
                continue
            f.write(json.dumps({**row, "caption": caption.strip()}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                rate = (time.time() - t0) / (i + 1)
                print(f"{i + 1}/{len(todo)}  {rate:.1f} s/image, about {rate * (len(todo) - i - 1) / 3600:.1f} h left",
                      flush=True)
    print(f"done, {dropped} dropped by the caption filter -> {out}")


if __name__ == "__main__":
    main()
