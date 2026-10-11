"""Stream danbooru tar batches from a Hugging Face dataset and keep only allowed images.

    python fetch_filtered.py --repo https://huggingface.co/datasets/AbdullahRian/Uncensored_Anime_Images ^
        --meta-file metadata/posts.json --out D:\\Iris-data\\images --min-score 30

The dataset needs .tar files of images named <md5>.<ext> or <post id>.<ext> and a
danbooru posts metadata file (JSON lines or a JSON array) somewhere in the repo.
The tar list and sizes come from the Hugging Face API, and the metadata file is
downloaded once into data/meta/<repo>/.

Each tar is read straight from the network. Members that are not on the allow-list
(danbooru.block_reason, which includes the minors rules) are skipped in memory and
never written to disk. Kept images get a <name>.json sidecar with their tags. If the
connection drops, it resumes from the last member with an HTTP range request.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from danbooru import block_reason, sidecar

HF = "https://huggingface.co"
BLOCK = 512
NUL = bytes(1)


def repo_id(repo: str) -> str:
    """'owner/name' from an id or any huggingface.co/datasets/... URL."""
    repo = repo.strip().rstrip("/")
    if "huggingface.co" in repo:
        parts = repo.split("huggingface.co/", 1)[1].split("/")
        parts = parts[1:] if parts[0] == "datasets" else parts
        return "/".join(parts[:2])
    return repo


def list_tars(rid: str) -> list[tuple[str, int]]:
    with urllib.request.urlopen(f"{HF}/api/datasets/{rid}/tree/main?recursive=true", timeout=60) as r:
        entries = json.loads(r.read())
    return sorted((e["path"], int(e.get("size") or 0)) for e in entries
                  if e.get("type") == "file" and e["path"].endswith(".tar"))


def read_posts(path: Path):
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            yield from json.load(f)
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)


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


def stream_tar(url: str, allow: dict, out: Path, state_file: Path, max_bytes: int | None, size_total: int = 0):
    """Parse the tar by hand so a dropped connection can resume at a member boundary."""
    state = json.loads(state_file.read_text()) if state_file.exists() else {"offset": 0, "done": False}
    state.update(url=url, size=size_total)
    if state["done"]:
        print(f"{url.rsplit('/', 1)[1]}: already done")
        state_file.write_text(json.dumps(state))
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
                if not header.strip(NUL):
                    state["done"] = True
                    break
                name = header[:100].rstrip(NUL).decode("utf-8", "replace")
                size = int(header[124:136].rstrip(NUL + b" ") or b"0", 8)
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
                stem = member.split(".", 1)[0]
                if kind in (b"0", NUL) and stem in allow and not (out / member).exists():
                    data = read_exact(stream, padded)[:size]
                    (out / member).write_bytes(data)
                    (out / f"{stem}.json").write_text(json.dumps(allow[stem], ensure_ascii=False), encoding="utf-8")
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
                    print(f"  {state['offset'] / 2**30:.2f} GB in, kept {kept}, skipped {skipped}, {rate:.2f} GB/min",
                          flush=True)
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
    ap.add_argument("--repo", default="AbdullahRian/Uncensored_Anime_Images", help="dataset id or huggingface.co URL")
    ap.add_argument("--meta-file", default="metadata/posts.json", help="metadata path inside the repo")
    ap.add_argument("--meta", default=None, help="local metadata file (downloaded here when missing)")
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--tars", default="all", help="all, or indexes into the repo's sorted tar list, e.g. 0,1")
    ap.add_argument("--min-side", type=int, default=832)
    ap.add_argument("--min-score", type=int, default=30)
    ap.add_argument("--ratings", default="e,q,s,g")
    ap.add_argument("--max-bytes", type=int, default=None, help="stop each tar after this many bytes (testing)")
    args = ap.parse_args()

    rid = repo_id(args.repo)
    here = Path(__file__).resolve().parent
    meta = Path(args.meta) if args.meta else here / "data" / "meta" / rid.replace("/", "__") / Path(args.meta_file).name
    if not meta.exists():
        meta.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading metadata {rid}/{args.meta_file}", flush=True)
        urllib.request.urlretrieve(f"{HF}/datasets/{rid}/resolve/main/{args.meta_file}", str(meta))
    tars = list_tars(rid)
    if not tars:
        raise SystemExit(f"no .tar files found in {rid}")
    pick = list(range(len(tars))) if args.tars.strip() == "all" else [int(x) for x in args.tars.split(",")]
    print(f"{rid}: {len(tars)} tars, {sum(s for _, s in tars) / 2**30:.1f} GB; fetching {len(pick)}", flush=True)

    ratings = tuple(r.strip() for r in args.ratings.split(","))
    allow, n = {}, 0
    for p in read_posts(meta):
        if block_reason(p, args.min_side, args.min_score, ratings) is None:
            side = sidecar(p)
            allow[p["md5"]] = side
            allow[str(p["id"])] = side  # some datasets name files by post id
            n += 1
    print(f"allow-list: {n} images", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in pick:
        path, size = tars[i]
        slug = (rid + "/" + path).replace("/", "__")
        print(path, flush=True)
        stream_tar(f"{HF}/datasets/{rid}/resolve/main/{path}", allow, out, out / f".{slug}.state.json",
                   args.max_bytes, size)
    print(f"images on disk: {sum(1 for _ in out.glob('*.json'))}")


if __name__ == "__main__":
    main()
