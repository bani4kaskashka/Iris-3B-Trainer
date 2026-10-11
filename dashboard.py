"""Local dashboard for the Iris LoRA pipeline: progress, jobs, logs and a dataset browser.

    python dashboard.py                      # http://127.0.0.1:7861, data in D:\\Iris-data
    python dashboard.py --data E:\\iris --port 7862

Everything the page does goes through a small JSON API, so scripts (or an agent)
can drive it too:
    GET  /api/status                    progress of every stage + running jobs
    GET  /api/jobs                      job definitions and their default settings
    POST /api/jobs/start  {"job": "caption", "params": {...}}
    POST /api/jobs/stop   {"job": "caption"}
    GET  /api/log?job=caption&lines=200
    GET  /api/dataset?offset=0&limit=60&q=text
    POST /api/exclude     {"path": "...", "excluded": true}
    GET  /api/runs        training runs with loss history and previews
It binds to 127.0.0.1 only. One GPU job runs at a time. Jobs run detached: closing
or restarting the dashboard never stops them, and a new dashboard picks them up.
"""

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
GPU_JOBS = {"caption", "encode", "train", "merge"}
# starts a job with no parent attached (and outside the caller's job object when allowed),
# prints its pid and exits, so nothing that kills the dashboard's process tree reaches the job
LAUNCHER = r"""
import json, subprocess, sys
a = json.loads(sys.argv[1])
log = open(a["log"], "a", encoding="utf-8")
base = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
for flags in (base | 0x01000000, base):  # try CREATE_BREAKAWAY_FROM_JOB first
    try:
        p = subprocess.Popen(a["cmd"], cwd=a["cwd"], env=a["env"], stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, creationflags=flags)
        break
    except OSError:
        continue
print(p.pid, flush=True)
"""


def comfy_scripts() -> str:
    return str(Path(os.environ["LOCALAPPDATA"]) / "Comfy-Desktop" / "ComfyUI-Installs" / "ComfyUI" / "ComfyUI" / ".venv" / "Scripts")


class Paths:
    def __init__(self, data: Path):
        self.data = data
        self.images = data / "images"
        self.work = data / "work"
        self.runs = data / "runs"
        self.logs = data / "logs"
        self.thumbs = data / "thumbs"
        self.vlm = data / "models" / "Huihui-Qwen3-VL-4B-Instruct-abliterated"
        self.manifest = self.work / "manifest.jsonl"
        self.captions = self.work / "captioned.jsonl"
        self.cache = self.work / "text_cache"
        self.excluded = self.work / "excluded.txt"
        for d in (self.images, self.work, self.runs, self.logs, self.thumbs):
            d.mkdir(parents=True, exist_ok=True)


def job_defs(P: Paths) -> dict:
    """name -> (description, default params). Params map to script flags in build_cmd()."""
    return {
        "fetch": ("Download: stream the filtered images from a Hugging Face dataset (any danbooru-style tar dataset)",
                  {"repo": "https://huggingface.co/datasets/AbdullahRian/Uncensored_Anime_Images",
                   "meta_file": "metadata/posts.json", "out": str(P.images), "min_score": 30, "tars": "all"}),
        "prep": ("Prep: tags, minors filter, size check, duplicates, buckets",
                 {"images": str(P.images), "out": str(P.manifest), "min_side": 832, "scale": 1.0}),
        "caption": ("Captions: natural-language captions with the VLM, plus tags (resumable)",
                    {"manifest": str(P.manifest), "out": str(P.captions), "vlm": str(P.vlm), "max_side": 768}),
        "encode": ("Text cache: run Iris's text encoder over every caption",
                   {"captions": str(P.captions), "cache": str(P.cache)}),
        "train": ("Train the LoRA (resume continues a run; init_lora starts from a finished LoRA)",
                  {"name": "anime", "steps": 10000, "rank": 128, "lr": 0.00008, "batch": 1, "accum": 1,
                   "save_every": 500, "preview_every": 500, "resume": True, "init_lora": ""}),
        "merge": ("Merge a LoRA into a full checkpoint for ComfyUI",
                  {"lora": "", "name": "anime", "strength": 1.0}),
    }


def build_cmd(job: str, p: dict, P: Paths) -> list[str]:
    py = str(HERE.parent / "iris-3b" / ".venv" / "Scripts" / "python.exe")
    if not Path(py).exists():
        py = sys.executable
    if job == "fetch":
        return [py, "-u", "fetch_filtered.py", "--repo", p["repo"], "--meta-file", p["meta_file"], "--out", p["out"],
                "--min-score", str(p["min_score"]), "--tars", str(p["tars"])]
    if job == "prep":
        return [py, "-u", "prep_dataset.py", "--images", p["images"], "--out", p["out"], "--require-tags",
                "--min-side", str(p["min_side"]), "--scale", str(p["scale"])]
    if job == "caption":
        return [py, "-u", "caption.py", "--manifest", p["manifest"], "--out", p["out"], "--vlm", p["vlm"],
                "--max-side", str(p["max_side"])]
    if job == "encode":
        return [py, "-u", "encode_text.py", "--captions", p["captions"], "--cache", p["cache"]]
    if job == "train":
        out = P.runs / p["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "run.json").write_text(json.dumps(p, indent=1))
        cmd = [py, "-u", "train_lora.py", "--captions", str(P.captions), "--cache", str(P.cache), "--out", str(out),
               "--steps", str(p["steps"]), "--rank", str(p["rank"]), "--lr", str(p["lr"]), "--batch", str(p["batch"]),
               "--accum", str(p["accum"]), "--save-every", str(p["save_every"]), "--preview-every", str(p["preview_every"]),
               "--exclude", str(P.excluded)]
        if p.get("init_lora"):
            cmd += ["--init-lora", p["init_lora"]]
        return cmd + (["--resume"] if p.get("resume") else [])
    if job == "merge":
        return [py, "-u", "merge_lora.py", p["lora"], "--name", p["name"], "--strength", str(p["strength"])]
    raise ValueError(f"unknown job {job}")


class Proc:
    """A detached job process, tracked by pid (survives dashboard restarts)."""

    def __init__(self, pid: int, log: Path):
        self.pid, self.log, self.returncode = pid, log, None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        import psutil

        try:
            if psutil.Process(self.pid).status() != psutil.STATUS_ZOMBIE:
                return None
        except psutil.Error:
            pass
        lines = tail(self.log, 25).splitlines()
        last_run = "\n".join(lines[max((i for i, l in enumerate(lines) if l.startswith("=== ")), default=0):])
        if lines and "stopped from the dashboard" in lines[-1]:
            self.returncode = -1
        elif "Traceback" in last_run:
            self.returncode = 1
        else:
            self.returncode = 0
        return self.returncode


class Jobs:
    def __init__(self, P: Paths):
        self.P, self.procs, self.meta, self.lock = P, {}, {}, threading.Lock()
        self.state = P.logs / "jobs.json"
        if self.state.exists():  # pick up jobs that are still running
            for name, m in json.loads(self.state.read_text()).items():
                proc = Proc(m["pid"], P.logs / f"{name}.log")
                if proc.poll() is None:
                    self.procs[name], self.meta[name] = proc, m

    def save(self):
        live = {n: {**self.meta[n], "pid": p.pid} for n, p in self.procs.items() if p.poll() is None}
        self.state.write_text(json.dumps(live))

    def running(self) -> dict:
        return {n: {"pid": p.pid, "since": self.meta[n]["since"], "params": self.meta[n]["params"]}
                for n, p in list(self.procs.items()) if p.poll() is None}

    def last(self) -> dict:
        return {n: {"exit": p.poll(), "since": self.meta[n]["since"]} for n, p in self.procs.items() if p.poll() is not None}

    def start(self, job: str, params: dict) -> dict:
        with self.lock:
            run = self.running()
            if job in run:
                return {"ok": False, "error": f"{job} is already running"}
            if job in GPU_JOBS and (busy := GPU_JOBS & set(run)):
                return {"ok": False, "error": f"the GPU is busy with {', '.join(busy)}; stop it first"}
            p = {**job_defs(self.P)[job][1], **(params or {})}
            cmd = build_cmd(job, p, self.P)
            env = {**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONIOENCODING": "utf-8",
                   "PATH": comfy_scripts() + os.pathsep + os.environ.get("PATH", ""),
                   "IRIS_MODELS": os.environ.get("IRIS_MODELS", r"E:\ComfyUI\models")}
            log = self.P.logs / f"{job}.log"
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start {job}: {' '.join(cmd)}\n")
            args = json.dumps({"cmd": cmd, "cwd": str(HERE), "env": env, "log": str(log)})
            out = subprocess.run([sys.executable, "-c", LAUNCHER, args], capture_output=True, text=True, timeout=60)
            if out.returncode != 0 or not out.stdout.strip().isdigit():
                return {"ok": False, "error": f"could not start {job}: {out.stderr[-400:]}"}
            self.procs[job] = Proc(int(out.stdout.strip()), log)
            self.meta[job] = {"since": time.time(), "params": p}
            self.save()
            return {"ok": True, "pid": self.procs[job].pid}

    def stop(self, job: str) -> dict:
        proc = self.procs.get(job)
        if not proc or proc.poll() is not None:
            return {"ok": False, "error": f"{job} is not running"}
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        with open(self.P.logs / f"{job}.log", "a", encoding="utf-8") as log:
            log.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} stopped from the dashboard\n")
        time.sleep(0.5)
        self.save()
        return {"ok": True}


# ---------- progress ----------

_count_cache: dict[str, tuple[float, float, int]] = {}


def line_count(path: Path, ttl: float = 5.0) -> int:
    if not path.exists():
        return 0
    key, now, mtime = str(path), time.time(), path.stat().st_mtime
    hit = _count_cache.get(key)
    if hit and hit[0] == mtime and now - hit[1] < ttl:
        return hit[2]
    with open(path, "rb") as f:
        n = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))
    _count_cache[key] = (mtime, now, n)
    return n


_img_cache = {"t": 0.0, "n": 0}


def image_count(P: Paths) -> int:
    if time.time() - _img_cache["t"] > 30:
        _img_cache["n"] = sum(1 for e in os.scandir(P.images) if e.name.endswith(".json") and not e.name.startswith("."))
        _img_cache["t"] = time.time()
    return _img_cache["n"]


def tail(path: Path, lines: int = 200) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 200_000))
        text = f.read().decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-lines:])


def rate_from_log(text: str) -> dict:
    m = re.findall(r"([\d.]+) s/image, about ([\d.]+) h left", text)
    return {"s_per_item": float(m[-1][0]), "eta_h": float(m[-1][1])} if m else {}


def fetch_state(P: Paths) -> dict:
    tars, got, total = [], 0, 0
    for st in sorted(P.images.glob(".*.state.json")):
        s = json.loads(st.read_text())
        size = s.get("size") or 0
        done = size if s.get("done") else min(s.get("offset", 0), size or s.get("offset", 0))
        got += done
        total += size
        name = (s.get("url") or st.name).rsplit("/", 1)[-1]
        tars.append({"tar": name, "done": bool(s.get("done")), "pct": round(100 * done / size, 1) if size else None})
    return {"pct": round(100 * got / total, 1) if total else 0, "images": image_count(P), "tars": tars}


def status(P: Paths, jobs: Jobs) -> dict:
    s = {"data": str(P.data), "running": jobs.running(), "finished": jobs.last(), "time": time.time()}
    s["fetch"] = fetch_state(P)
    images, manifest = image_count(P), line_count(P.manifest)
    s["prep"] = {"manifest": manifest, "images": images}
    caps = line_count(P.captions)
    s["caption"] = {"done": caps, "total": manifest, "pct": round(100 * caps / manifest, 1) if manifest else 0,
                    **rate_from_log(tail(P.logs / "caption.log", 30))}
    enc = line_count(P.cache / "index.jsonl")
    s["encode"] = {"done": enc, "total": caps, "pct": round(100 * enc / caps, 1) if caps else 0,
                   "shards": len(list(P.cache.glob("shard_*.safetensors"))) if P.cache.exists() else 0}
    s["excluded"] = line_count(P.excluded, ttl=0)
    s["train"] = [run_summary(r) for r in sorted(P.runs.iterdir()) if r.is_dir()] if P.runs.exists() else []
    return s


def run_summary(run: Path, history: bool = False) -> dict:
    cfg = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    rows = []
    if (run / "log.csv").exists():
        for line in tail(run / "log.csv", 100000 if history else 1).splitlines():
            parts = line.split(",")
            if len(parts) >= 3:
                rows.append([int(parts[0]), float(parts[1]), float(parts[2])])
    step = rows[-1][0] if rows else 0
    total = int(cfg.get("steps", 0)) or None
    out = {"name": run.name, "step": step, "steps": total, "pct": round(100 * step / total, 1) if total else None,
           "loss_avg": rows[-1][2] if rows else None,
           "loras": sorted(p.name for p in run.glob("lora_*.safetensors")),
           "previews": sorted(p.name for p in run.glob("preview_*.png"))}
    if history:
        stride = max(1, len(rows) // 1500)
        out["history"] = rows[::stride]
        out["config"] = cfg
    return out


# ---------- dataset ----------

def excluded_set(P: Paths) -> set[str]:
    if not P.excluded.exists():
        return set()
    return {line.strip() for line in P.excluded.read_text(encoding="utf-8").splitlines() if line.strip()}


_cap_cache = {"mtime": None, "map": {}}


def captions_by_path(P: Paths) -> dict:
    if not P.captions.exists():
        return {}
    m = P.captions.stat().st_mtime
    if _cap_cache["mtime"] != m:
        out = {}
        for line in open(P.captions, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["path"]] = r.get("caption")
        _cap_cache.update(mtime=m, map=out)
    return _cap_cache["map"]


def dataset(P: Paths, offset: int, limit: int, q: str, only_excluded: bool) -> dict:
    caps, ex = captions_by_path(P), excluded_set(P)
    rows, total = [], 0
    if P.manifest.exists():
        for line in open(P.manifest, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            cap = caps.get(r["path"])
            text = (cap or "") + " " + " ".join(r.get("tags", []))
            if q and q.lower() not in text.lower():
                continue
            if only_excluded and r["path"] not in ex:
                continue
            if offset <= total < offset + limit:
                rows.append({"path": r["path"], "caption": cap, "tags": r.get("tags", [])[:40], "bucket": r.get("bucket"),
                             "size": [r.get("width"), r.get("height")], "excluded": r["path"] in ex})
            total += 1
    return {"source": f"{len(caps)} captioned", "total": total, "rows": rows}


def set_excluded(P: Paths, path: str, excluded: bool) -> dict:
    ex = excluded_set(P)
    (ex.add if excluded else ex.discard)(path)
    P.excluded.write_text("".join(p + "\n" for p in sorted(ex)), encoding="utf-8")
    return {"ok": True, "excluded": len(ex)}


def allowed_file(P: Paths, path: str) -> Path | None:
    """Only files inside the pipeline's own folders are served."""
    src = Path(path).resolve()
    if (P.data.resolve() in src.parents or HERE.resolve() in src.parents) and src.is_file():
        return src
    return None


def thumbnail(P: Paths, path: str) -> Path | None:
    from hashlib import md5

    from PIL import Image

    src = allowed_file(P, path)
    if not src:
        return None
    th = P.thumbs / (md5(str(src).encode()).hexdigest() + ".jpg")
    if not th.exists():
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((320, 320))
            im.save(th, quality=82)
    return th


# ---------- http ----------

def make_handler(P: Paths, jobs: Jobs):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def file(self, path: Path | None):
            if not path or not path.exists():
                return self.send(404, {"error": "not found"})
            self.send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    return self.file(HERE / "dashboard.html")
                if u.path == "/api/status":
                    return self.send(200, status(P, jobs))
                if u.path == "/api/jobs":
                    return self.send(200, {k: {"desc": d, "params": p, "gpu": k in GPU_JOBS} for k, (d, p) in job_defs(P).items()})
                if u.path == "/api/log":
                    name = re.sub(r"[^a-z]", "", q.get("job", ""))
                    return self.send(200, {"text": tail(P.logs / f"{name}.log", int(q.get("lines", 200)))})
                if u.path == "/api/dataset":
                    return self.send(200, dataset(P, int(q.get("offset", 0)), int(q.get("limit", 60)), q.get("q", ""),
                                                  q.get("excluded") == "1"))
                if u.path == "/api/runs":
                    return self.send(200, [run_summary(r, history=True) for r in sorted(P.runs.iterdir()) if r.is_dir()])
                if u.path == "/thumb":
                    return self.file(thumbnail(P, q.get("path", "")))
                if u.path == "/image":
                    return self.file(allowed_file(P, q.get("path", "")))
                if u.path == "/preview":
                    run = P.runs / re.sub(r"[\\/:]", "", q.get("run", ""))
                    return self.file(run / re.sub(r"[\\/:]", "", q.get("file", "")))
                return self.send(404, {"error": "not found"})
            except Exception as exc:  # keep the dashboard alive
                return self.send(500, {"error": str(exc)})

        def do_POST(self):
            u = urlparse(self.path)
            try:
                b = self.body()
                if u.path == "/api/jobs/start":
                    if b.get("job") not in job_defs(P):
                        return self.send(400, {"ok": False, "error": "unknown job"})
                    return self.send(200, jobs.start(b["job"], b.get("params") or {}))
                if u.path == "/api/jobs/stop":
                    return self.send(200, jobs.stop(b.get("job", "")))
                if u.path == "/api/exclude":
                    return self.send(200, set_excluded(P, b["path"], bool(b.get("excluded", True))))
                return self.send(404, {"error": "not found"})
            except Exception as exc:
                return self.send(500, {"ok": False, "error": str(exc)})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("IRIS_DATA", r"D:\Iris-data"))
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()
    P = Paths(Path(args.data))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(P, Jobs(P)))
    print(f"Iris dashboard on http://127.0.0.1:{args.port}  (data: {P.data})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
