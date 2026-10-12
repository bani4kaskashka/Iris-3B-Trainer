"""Step 1: scan an image folder, filter it and write a manifest.

    python prep_dataset.py --images D:\\data\\danbooru --out work\\manifest.jsonl

Each image may have a sidecar with its tags:
    name.txt   booru tags, comma separated (or space separated with underscores)
    name.json  danbooru post metadata (tag_string, rating, ...)

Rejected: anything the minors filter in safety.py blocks (no override), images
whose short side is under --min-side, unreadable files and exact duplicates.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

from common import bucket_for, write_jsonl
from safety import minor_series_or_character, school_blocked, tags_blocked

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RATING = {"g": "general", "s": "sensitive", "q": "questionable", "e": "explicit"}


def read_tags(img: Path) -> tuple[list[str], str | None]:
    js, txt = img.with_suffix(".json"), img.with_suffix(".txt")
    if js.exists():
        meta = json.loads(js.read_text(encoding="utf-8"))
        tags = (meta.get("tag_string") or " ".join(
            meta.get(k, "") for k in ("tag_string_general", "tag_string_character", "tag_string_copyright", "tag_string_artist")
        )).split()
        return tags, RATING.get(meta.get("rating"), meta.get("rating"))
    if txt.exists():
        raw = txt.read_text(encoding="utf-8").strip()
        tags = [t.strip() for t in raw.split(",")] if "," in raw else raw.split()
        rating = next((t.split(":", 1)[1] for t in tags if t.startswith("rating:")), None)
        return [t for t in tags if t and not t.startswith("rating:")], RATING.get(rating, rating)
    return filename_tags(img.stem), None


# "yande.re 123456 sample tag1 tag2", "Konachan.com - 123456 tag1 tag2", "moe 123456 tag1"
BOORU_NAME = re.compile(r"^(?:yande\.re|konachan\.(?:com|net)\s*-|moe)\s+\d+\s+(.*)$", re.I)


def filename_tags(stem: str) -> list[str]:
    """Tags that yande.re / konachan / moe.imouto put in their download file names."""
    m = BOORU_NAME.match(stem.strip())
    if not m:
        return []
    return [t for t in m.group(1).split() if t.lower() not in ("sample", "jpeg", "png")]


def post_like(img: Path, tags) -> dict:
    """danbooru json sidecars carry character/copyright tags; otherwise check all tags."""
    js = img.with_suffix(".json")
    if js.exists():
        return json.loads(js.read_text(encoding="utf-8"))
    return {"tag_string": " ".join(tags)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, nargs="+")
    ap.add_argument("--out", default="work/manifest.jsonl")
    ap.add_argument("--min-side", type=int, default=832, help="drop images whose short side is smaller")
    ap.add_argument("--ratings", default="", help="keep only these ratings, e.g. explicit,questionable (empty = all)")
    ap.add_argument("--require-tags", action="store_true", help="drop images without a tag sidecar")
    ap.add_argument("--scale", type=float, default=1.0, help="bucket scale, 0.75 trains at about 768 px")
    args = ap.parse_args()

    keep_ratings = {r.strip() for r in args.ratings.split(",") if r.strip()}
    rows, why, seen = [], Counter(), set()
    files = [p for d in args.images for p in Path(d).rglob("*") if p.suffix.lower() in EXTS]
    for i, path in enumerate(sorted(files)):
        tags, rating = read_tags(path)
        if args.require_tags and not tags:
            why["no tags"] += 1
            continue
        if reason := tags_blocked(tags) or school_blocked(tags, rating) or (
                rating not in ("g", "general") and minor_series_or_character(post_like(path, tags))):
            why["minors filter: " + reason.split()[0]] += 1
            continue
        if keep_ratings and rating not in keep_ratings:
            why["rating"] += 1
            continue
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception:
            why["unreadable"] += 1
            continue
        if min(width, height) < args.min_side:
            why["too small"] += 1
            continue
        digest = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
        if digest in seen:
            why["duplicate"] += 1
            continue
        seen.add(digest)
        rows.append({"path": str(path.resolve()), "width": width, "height": height, "tags": tags,
                     "rating": rating, "bucket": list(bucket_for(width, height, args.scale))})
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(files)} scanned, {len(rows)} kept", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    print(f"kept {len(rows)} of {len(files)} -> {args.out}")
    for reason, n in why.most_common():
        print(f"  dropped {n:>7}  {reason}")
    buckets = Counter(tuple(r["bucket"]) for r in rows)
    print("buckets:", ", ".join(f"{w}x{h}:{n}" for (w, h), n in buckets.most_common()))


if __name__ == "__main__":
    main()
