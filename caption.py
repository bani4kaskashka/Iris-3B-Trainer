"""Step 2: caption every image in the manifest.

    python caption.py --manifest work/manifest.jsonl --out work/captioned.jsonl
    python caption.py ... --vlm path\\to\\Huihui-Qwen3-VL-4B-Instruct-abliterated
    python caption.py ... --mode tags          # no VLM, captions built from the tags

The VLM writes a natural-language description (Iris's text encoder was trained on
those); the image's booru tags are passed in as hints and appended to the caption.
The prompt, tag handling, sampling, extra drop words and phrases to cut come from
the "caption" part of settings.json (the dashboard's Settings tab). Any caption that
mentions a child or a minor is dropped (safety.py), whatever the settings say.
Resumable: rows already in --out, or dropped earlier, are skipped.
"""

import argparse
import json
import re
import time
from pathlib import Path

import common  # noqa: F401  (paths, AOTriton)
import torch
from PIL import Image

import settings
from common import read_jsonl, write_jsonl
from safety import caption_blocked

# whole words only (letters may not touch either side). Only the prose is rewritten:
# the booru tags (1girl, 1boy, girl_on_top...) are kept exactly as people prompt them.
ADULT_WORDS = [("girls", "women"), ("girl", "woman"), ("boys", "men"), ("boy", "man")]
_EDGE = ("(?<![A-Za-z])", "(?![A-Za-z])")


def adult_words(text: str) -> str:
    for word, adult in ADULT_WORDS:
        text = re.sub(_EDGE[0] + word + _EDGE[1], adult, text, flags=re.I)
    return text


def tag_text(tags, cfg) -> str:
    skip = {t.lower().replace(" ", "_") for t in cfg["skip_tags"]}
    keep = [t for t in tags if t.lower().replace(" ", "_") not in skip]
    return ", ".join(t.replace("_", " ") for t in keep[:int(cfg["max_tags"])])


def word_list(words) -> re.Pattern | None:
    # spaces and underscores match each other, so "fox_tail" also catches the tag written as "fox tail"
    words = [re.split(r"[ _]+", w.strip()) for w in words if w.strip()]
    if not words:
        return None
    alts = "|".join("[ _]+".join(re.escape(part) for part in w) for w in words)
    return re.compile(_EDGE[0] + "(" + alts + ")" + _EDGE[1], re.I)


def remove_phrases(text: str, phrases) -> str:
    for ph in (x.strip() for x in phrases):
        if ph:
            text = re.sub(re.escape(ph), "", text, flags=re.I)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;:")
    return text[:1].upper() + text[1:]


def load_vlm(path: str, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device).train(False)
    return processor, model


@torch.no_grad()
def vlm_caption(processor, model, image: Image.Image, tags: str, device, cfg) -> str:
    image = image.convert("RGB")
    scale = min(1.0, int(cfg["max_side"]) / max(image.size))
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    prompt = cfg["prompt"].replace("{tags}", (tags or "none") if cfg["tags_as_hints"] else "none")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(device)
    temp = float(cfg["temperature"])
    sampling = {"do_sample": True, "temperature": temp, "top_p": float(cfg["top_p"])} if temp > 0 else {"do_sample": False}
    out = model.generate(**inputs, max_new_tokens=int(cfg["max_new_tokens"]),
                         repetition_penalty=float(cfg["repetition_penalty"]), **sampling)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def build_caption(text: str, tags: str, cfg) -> str:
    if cfg["rewrite_girl_boy"]:
        text = adult_words(text)
    text = remove_phrases(text, cfg["remove_phrases"]) if text else text
    parts = [cfg["prefix"].strip(), text, tags if cfg["append_tags"] or not text else ""]
    return " ".join(x for x in parts if x).strip()


def blocked(caption: str, extra: re.Pattern | None) -> str | None:
    if reason := caption_blocked(caption):  # built in, always on
        return reason
    if extra and (m := extra.search(caption)):
        return f"drop word '{m.group(0)}'"
    return None


def main():
    cfg = settings.load()["caption"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="work/manifest.jsonl")
    ap.add_argument("--out", default="work/captioned.jsonl")
    ap.add_argument("--mode", choices=["vlm", "tags"], default=cfg["mode"])
    ap.add_argument("--vlm", default=str(settings.captioner()), help="Qwen3-VL folder (an abliterated one avoids refusals)")
    ap.add_argument("--max-side", type=int, default=cfg["max_side"], help="image size the VLM sees")
    ap.add_argument("--prefix", default=cfg["prefix"], help="optional text in front of every caption")
    args = ap.parse_args()
    cfg = {**cfg, "max_side": args.max_side, "prefix": args.prefix}
    extra = word_list(cfg["drop_words"])

    rows = read_jsonl(args.manifest)
    out = Path(args.out)
    # dropped images are remembered, so a restart doesn't re-roll them until a caption slips through;
    # an old drop is let in only if its saved caption passes the current filter
    drops_file = out.with_name(out.stem + "_dropped.jsonl")
    drops = {r["path"]: r for r in read_jsonl(drops_file)} if drops_file.exists() else {}
    done = {r["path"] for r in read_jsonl(out)} if out.exists() else set()
    readmit = [d for p, d in drops.items() if p not in done and d.get("caption") and not blocked(d["caption"], extra)]
    if readmit:
        by_path = {r["path"]: r for r in rows}
        with open(out, "a", encoding="utf-8") as f:
            for d in readmit:
                if d["path"] in by_path:
                    f.write(json.dumps({**by_path[d["path"]], "caption": d["caption"]}, ensure_ascii=False) + "\n")
                    done.add(d["path"])
        for d in readmit:
            drops.pop(d["path"], None)
        write_jsonl(drops_file, drops.values())
        print(f"{len(readmit)} earlier drops pass the current filter and were added back")
    todo = [r for r in rows if r["path"] not in done and r["path"] not in drops]
    print(f"{len(rows)} images, {len(done)} already captioned, {len(drops)} dropped earlier, {len(todo)} to go")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "vlm" and todo:
        processor, model = load_vlm(args.vlm, device)
    dropped, t0 = 0, time.time()
    with open(out, "a", encoding="utf-8") as f, open(drops_file, "a", encoding="utf-8") as df:
        for i, row in enumerate(todo):
            tags = tag_text(row.get("tags", []), cfg)
            text = ""
            if args.mode == "vlm":
                with Image.open(row["path"]) as im:
                    text = vlm_caption(processor, model, im, tags, device, cfg)
                if text.lower().startswith(("i can't", "i cannot", "i'm sorry", "sorry")):
                    text = ""  # refusal: fall back to the tags alone
            caption = build_caption(text, tags, cfg)
            if reason := blocked(caption, extra):
                dropped += 1
                print(f"  dropped ({reason}): {row['path']}")
                df.write(json.dumps({"path": row["path"], "reason": reason, "caption": caption}, ensure_ascii=False) + "\n")
                df.flush()
                continue
            f.write(json.dumps({**row, "caption": caption}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                rate = (time.time() - t0) / (i + 1)
                print(f"{i + 1}/{len(todo)}  {rate:.1f} s/image, about {rate * (len(todo) - i - 1) / 3600:.1f} h left",
                      flush=True)
    print(f"done, {dropped} dropped by the caption filter -> {out}")


if __name__ == "__main__":
    main()
