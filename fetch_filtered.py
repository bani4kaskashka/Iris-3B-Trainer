"""Stream danbooru tar batches from Hugging Face and keep only allowed images.

    python fetch_filtered.py --meta data/meta/posts.json --out data/images --min-score 30

Each tar is read straight from the network. Members whose md5 is not on the
allow-list (danbooru.block_reason, which includes the minors rules) are skipped
in memory and never written to disk. Kept images get a <md5>.json sidecar with
their tags. If the connection drops, it resumes from the last member with an
HTTP range request.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from danbooru import block_reason, sidecar

REPO = "https://huggingface.co/datasets/AbdullahRian/Uncensored_Anime_Images/resolve/main"
BLOCK = 512


def open_at(url: str, offset: int):
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    return urllib.request.urlopen(req, timeout=60)


def read_exact(stream, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = stream.read(min(n - len(out), 1 << 20))
        if not chunk:
            raise ConnectionError("stream ended early")
        out += chunk
    return bytes(out)


def stream_tar(url: str, allow: dict, out: Path, state_file: Path, max_bytes: int | None):
    """Parse the tar by hand so a dropped connection can resume at a member boundary."""
    state = json.loads(state_file.read_text()) if state_file.exists() else {"offset": 0, "done": False}
    if state["done"]:
        print(f"{url.rsplit('/', 1)[1]}: already done")
        return
    kept = skipped = 0
    t0, start = time.time(), state["offset"]
    retries = 0
    while True:
        try:
            stream = open_at(url, state["offset"])
            pax_name, pending = None, 0  # pending = pax bytes read for the member that follows
            while True:
                header = read_exact(stream, BLOCK)
                if not header.strip(b"\0"):
                    state["done"] = True
                    break
                name = header[:100].rstrip(b"\0").decode("utf-8", "replace")
                size = int(header[124:136].rstrip(b"\0 ") or b"0", 8)
                kind = header[156:157]
                padded = (size + BLOCK - 1) // BLOCK * BLOCK
                if kind == b"x":  # pax header: may carry the real path
                    pax = read_exact(stream, padded)[:size].decode("utf-8", "replace")
                    for line in pax.splitlines():
                        if " path=" in line:
                            pax_name = line.split(" path=", 1)[1]
                    pending += BLOCK + padded
                    continue  # the member it describes follows; state stays at the pax header
                member = Path(pax_name or name).name
                pax_name = None
                md5 = member.split(".", 1)[0]
                if kind in (b"0", b"\0") and md5 in allow and not (out / member).exists():
                    data = read_exact(stream, padded)[:size]
                    (out / member).write_bytes(data)
                    (out / f"{md5}.json").write_text(json.dumps(allow[md5], ensure_ascii=False), encoding="utf-8")
                    kept += 1
                else:
                    remaining = padded
                    while remaining:  # discard without keeping anything
                        chunk = stream.read(min(remaining, 1 << 20))
                        if not chunk:
                            raise ConnectionError("stream ended early")
                        remaining -= len(chunk)
                    skipped += 1
                # the next member (or its pax header) starts here
                state["offset"] += pending + BLOCK + padded
                pending = 0
                if (kept + skipped) % 500 == 0:
                    state_file.write_text(json.dumps(state))
                    done = (state["offset"] - start) / 2**30
                    rate = done / max(time.time() - t0, 1e-6) * 60
                    print(f"  {state['offset'] / 2**30:.2f} GB in, kept {kept}, skipped {skipped}, {rate:.2f} GB/min", flush=True)
                if max_bytes and state["offset"] - start >= max_bytes:
                    print("  --max-bytes reached")
                    state_file.write_text(json.dumps(state))
                    return
            state_file.write_text(json.dumps(state))
            print(f"  finished: kept {kept}, skipped {skipped}")
            return
        except (ConnectionError, OSError, TimeoutError) as exc:
            retries += 1
            state_file.write_text(json.dumps(state))
            if retries > 20:
                raise
            print(f"  connection problem ({exc}); resuming at {state['offset']} in 10 s")
            time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default="data/meta/posts.json")
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--tars", default="0,1,2,3")
    ap.add_argument("--min-side", type=int, default=832)
    ap.add_argument("--min-score", type=int, default=30)
    ap.add_argument("--ratings", default="e,q,s,g")
    ap.add_argument("--max-bytes", type=int, default=None, help="stop each tar after this many bytes (testing)")
    args = ap.parse_args()

    ratings = tuple(r.strip() for r in args.ratings.split(","))
    allow = {}
    for line in open(args.meta, encoding="utf-8"):
        if line.strip():
            p = json.loads(line)
            if block_reason(p, args.min_side, args.min_score, ratings) is None:
                allow[p["md5"]] = sidecar(p)
    print(f"allow-list: {len(allow)} images")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in (int(x) for x in args.tars.split(",")):
        url = f"{REPO}/batch_{i}.tar"
        print(f"batch_{i}.tar")
        stream_tar(url, allow, out, out / f".batch_{i}.state.json", args.max_bytes)
    print(f"images on disk: {sum(1 for _ in out.glob('*.json'))}")


if __name__ == "__main__":
    main()
