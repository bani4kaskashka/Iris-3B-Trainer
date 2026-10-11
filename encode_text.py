"""Step 3: run Iris's text encoder over every caption once and cache the result.

    python encode_text.py --captions work/captioned.jsonl --cache work/text_cache

Training then never loads the 8 GB text encoder. Only the real (unpadded) tokens
are stored, in bf16: about 61 KB per token, so a 80-token caption is ~5 MB.
Also caches the empty prompt (for caption dropout / CFG) and the preview prompts.
"""

import argparse
import json
from pathlib import Path

import common  # noqa: F401
import torch
from safetensors.torch import save_file

from common import TEXT_ENCODER, load_base_config, read_jsonl

PREVIEW_PROMPTS = [
    "anime illustration. A woman with long silver hair and red eyes in a black dress, standing in a moonlit garden.",
    "anime illustration. A smiling woman with short brown hair in a white summer dress at the beach, bright sunlight.",
]


def build_encoder(device):
    import dataclasses

    from iris3b.text.qwen3_vl import Qwen3VLTextEncoder

    cfg = dataclasses.replace(load_base_config().text_encoder, pretrained=str(TEXT_ENCODER))
    enc = Qwen3VLTextEncoder(cfg, device=device)
    enc._null_cache_path = lambda: Path("work") / "null_embed_cache.pth"  # keep a Windows path out of the file name
    return enc


def trimmed(encoding):
    """[B, L, layers, D] padded -> list of [n_i, layers, D] bf16 on the CPU."""
    out = []
    for e, m in zip(encoding.embeddings, encoding.mask, strict=True):
        n = int(m.sum())
        out.append(e[:n].to(torch.bfloat16).cpu().contiguous())
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", default="work/captioned.jsonl")
    ap.add_argument("--cache", default="work/text_cache")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=256, help="captions per cache file")
    args = ap.parse_args()

    rows = read_jsonl(args.captions)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = build_encoder(device)

    extra = {"null": enc.null(""), **{f"preview{i}": enc.encode([p]) for i, p in enumerate(PREVIEW_PROMPTS)}}
    save_file({k: trimmed(v)[0] for k, v in extra.items()}, str(cache / "extra.safetensors"),
              metadata={"previews": json.dumps(PREVIEW_PROMPTS)})

    index = []
    for start in range(0, len(rows), args.shard):
        chunk = rows[start:start + args.shard]
        shard = f"shard_{start // args.shard:05d}.safetensors"
        tensors = {}
        for b in range(0, len(chunk), args.batch):
            batch = chunk[b:b + args.batch]
            for j, emb in enumerate(trimmed(enc.encode([r["caption"] for r in batch]))):
                key = f"s{start + b + j}"
                tensors[key] = emb
                index.append({"sample": start + b + j, "shard": shard, "key": key, "tokens": emb.shape[0]})
        save_file(tensors, str(cache / shard))
        print(f"{min(start + args.shard, len(rows))}/{len(rows)} encoded -> {shard}", flush=True)
    with open(cache / "index.jsonl", "w", encoding="utf-8") as f:
        for r in index:
            f.write(json.dumps(r) + "\n")
    toks = [r["tokens"] for r in index]
    print(f"done: {len(index)} captions, {sum(toks) / max(len(toks), 1):.0f} tokens on average, longest {max(toks, default=0)}")


if __name__ == "__main__":
    main()
