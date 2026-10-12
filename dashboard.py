"""Local dashboard for the Iris LoRA pipeline: progress, jobs, logs and a dataset browser.

    start.bat                                # first run sets everything up, then opens the page
    python dashboard.py --open               # http://127.0.0.1:7861, data folder from settings.json
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
    POST /api/jobs/pause  training saves at the end of its current step and stops
    GET  /api/settings    settings.json, what empty fields resolve to, and missing pieces
    POST /api/settings    {"caption": {"drop_words": [...]}, ...}  (merged into settings.json)
    GET  /api/presets     training presets
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

import settings

HERE = Path(__file__).resolve().parent
GPU_JOBS = {"caption", "encode", "train", "merge", "compare"}
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


def job_python() -> str:
    py = settings.load()["python"]
    return py if py and Path(py).exists() else sys.executable


def job_env() -> dict:
    s = settings.load()
    py = Path(job_python())
    # the job Python's own folders first (venvs keep tools like hipInfo.exe in Scripts)
    path = [str(x) for x in s["extra_path"] if str(x).strip()] + [str(py.parent), str(py.parent / "Scripts")]
    return {**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONIOENCODING": "utf-8",
            "PATH": os.pathsep.join(path + [os.environ.get("PATH", "")]),
            "IRIS_MODELS": str(settings.models_dir(s)), "IRIS_REPO": str(settings.iris_repo(s)),
            "IRIS_SETTINGS": str(settings.FILE), **{k: str(v) for k, v in s["env"].items()}}


class Paths:
    def __init__(self, data: Path):
        self.data = data
        self.images = data / "images"
        self.work = data / "work"
        self.runs = data / "runs"
        self.logs = data / "logs"
        self.thumbs = data / "thumbs"
        self.manifest = self.work / "manifest.jsonl"
        self.captions = self.work / "captioned.jsonl"
        self.cache = self.work / "text_cache"
        self.excluded = self.work / "excluded.txt"
        for d in (self.images, self.work, self.runs, self.logs, self.thumbs):
            d.mkdir(parents=True, exist_ok=True)


def job_defs(P: Paths) -> dict:
    """name -> (description, default params). Params map to script flags in build_cmd()."""
    return {
        "models": ("Models: download Iris-3B and its text encoder into the models folder (skips what you have). "
                   "captioner also gets the abliterated Qwen3-VL, which refuses far less",
                   {"captioner": True}),
        "fetch": ("Download: stream the filtered images from a Hugging Face dataset (any danbooru-style tar dataset)",
                  {"repo": "https://huggingface.co/datasets/AbdullahRian/Uncensored_Anime_Images",
                   "meta_file": "metadata/posts.json", "out": str(P.images), "min_score": 30, "tars": "all"}),
        "prep": ("Prep: tags (sidecars or booru file names), minors filter, size check, duplicates, buckets. "
                 "Several image folders: separate them with ;",
                 {"images": str(P.images), "out": str(P.manifest), "min_side": 832, "scale": 1.0, "require_tags": False}),
        "caption": ("Captions: natural-language captions with the VLM, plus tags (resumable). "
                    "Prompt, words and sampling are in the Settings tab",
                    {"manifest": str(P.manifest), "out": str(P.captions)}),
        "encode": ("Text cache: run Iris's text encoder over every caption",
                   {"captions": str(P.captions), "cache": str(P.cache)}),
        "train": ("Train the LoRA (resume continues a run; init_lora starts from a finished LoRA)",
                  {"name": "my-lora", **PRESETS["Balanced"], "batch": 1, "accum": 1, "resume": True, "init_lora": ""}),
        "merge": ("Merge a LoRA into a full checkpoint for ComfyUI",
                  {"lora": "", "name": "my-lora", "strength": 1.0}),
        "compare": ("Compare: the same prompts and seeds with the base model and with a LoRA (Training tab)",
                    {"lora": "", "prompts": "\n".join(COMPARE_PROMPTS), "steps": 50, "cfg": 3.0, "seed": 100,
                     "strength": 1.0}),
    }


# Compare defaults: two booru-tag prompts, one plain sentence, one mix of both
COMPARE_PROMPTS = [
    "1girl, solo, mature female, long hair, black hair, large breasts, nude, nipples, navel, lying, on bed, "
    "looking at viewer, blush, smile, bedroom, window, sunlight",
    "1girl, 1boy, hetero, sex, vaginal, missionary, mature female, brown hair, medium breasts, nude, sweat, "
    "open mouth, blush, on bed, pov",
    "anime illustration. A tall adult woman with wavy red hair and green eyes, wearing a black lace lingerie set, "
    "sits on the edge of a velvet sofa in a dim lounge, warm lamplight, confident smile, looking at the viewer.",
    "anime illustration of a muscular adult woman with short white hair and a scar over one eye, topless, leaning "
    "against a stone wall in a rainy alley at night, neon reflections. 1girl, mature female, abs, topless, wet, "
    "night, rain, neon lights, looking at viewer",
]


# training presets (the Train card's preset menu fills these in). VRAM is the peak measured
# on a 16 GB card at 832x1216; leave a couple of GB free for the desktop and browser.
PRESETS = {
    "Quick test": {"steps": 1500, "rank": 32, "lr": 0.0001, "save_every": 250, "preview_every": 250},
    "Character or style": {"steps": 3000, "rank": 32, "lr": 0.0001, "save_every": 500, "preview_every": 500},
    "Balanced": {"steps": 8000, "rank": 64, "lr": 0.0001, "save_every": 500, "preview_every": 500},
    "Big dataset": {"steps": 20000, "rank": 64, "lr": 0.00008, "save_every": 1000, "preview_every": 1000},
    "High capacity": {"steps": 12000, "rank": 128, "lr": 0.00008, "save_every": 1000, "preview_every": 1000},
    "Keep training a LoRA": {"steps": 4000, "rank": 64, "lr": 0.00005, "save_every": 500, "preview_every": 500},
}
PRESET_NOTES = {
    "Quick test": "a short run to check the data and captions, about 1.5 h",
    "Character or style": "a few hundred to a few thousand images of one thing",
    "Balanced": "rank 64, about 9.4 GB VRAM and 4.3 s/step",
    "Big dataset": "10k+ images, teaching new concepts; about a day",
    "High capacity": "rank 128 needs about 15 GB VRAM: close games, browsers and anything else on the GPU",
    "Keep training a LoRA": "set init_lora to a finished LoRA of the same rank; new images or more steps",
}


def build_cmd(job: str, p: dict, P: Paths) -> list[str]:
    py = job_python()
    if job == "models":
        return [py, "-u", "download_models.py"] + (["--captioner"] if p.get("captioner") else [])
    if job == "fetch":
        return [py, "-u", "fetch_filtered.py", "--repo", p["repo"], "--meta-file", p["meta_file"], "--out", p["out"],
                "--min-score", str(p["min_score"]), "--tars", str(p["tars"])]
    if job == "prep":
        folders = [f.strip() for f in str(p["images"]).split(";") if f.strip()]
        return [py, "-u", "prep_dataset.py", "--images", *folders, "--out", p["out"], "--min-side", str(p["min_side"]),
                "--scale", str(p["scale"])] + (["--require-tags"] if p.get("require_tags") else [])
    if job == "caption":
        return [py, "-u", "caption.py", "--manifest", p["manifest"], "--out", p["out"]]
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
    if job == "compare":
        if not p["lora"]:
            raise ValueError("pick a LoRA to compare")
        prompts = p["prompts"] if isinstance(p["prompts"], list) else str(p["prompts"]).splitlines()
        file = P.work / "compare_prompts.txt"
        file.write_text("\n".join(x.strip() for x in prompts if x.strip()) + "\n", encoding="utf-8")
        return [py, "-u", "compare_lora.py", p["lora"], "--prompts", str(file), "--steps", str(p["steps"]),
                "--cfg", str(p["cfg"]), "--seed", str(p["seed"]), "--strength", str(p["strength"])]
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
            env = job_env()
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

    def pause(self) -> dict:
        """Ask the running training to save at the end of its current step and stop."""
        run = self.running().get("train")
        if not run:
            return {"ok": False, "error": "training is not running"}
        (self.P.runs / run["params"]["name"] / "PAUSE").write_text("")
        return {"ok": True}

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
    s["models"] = settings_view(P)["checks"]
    s["fetch"] = fetch_state(P)
    images, manifest = image_count(P), line_count(P.manifest)
    s["prep"] = {"manifest": manifest, "images": images}
    caps = line_count(P.captions)
    s["caption"] = {"done": caps, "total": manifest, "pct": round(100 * caps / manifest, 1) if manifest else 0,
                    **rate_from_log(tail(P.logs / "caption.log", 30))}
    enc = line_count(P.cache / "index.jsonl")
    if "encode" in s["running"]:  # index.jsonl is complete only at the end, the log has the live count
        live = re.findall(r"(\d+)/\d+ (?:already )?encoded", tail(P.logs / "encode.log", 5))
        enc = int(live[-1]) if live else 0
    s["encode"] = {"done": enc, "total": caps, "pct": round(100 * enc / caps, 1) if caps else 0,
                   "shards": len(list(P.cache.glob("shard_*.safetensors"))) if P.cache.exists() else 0}
    s["excluded"] = line_count(P.excluded, ttl=0)
    last_train = tail(P.logs / "train.log", 4)
    s["train_paused"] = "train" not in s["running"] and "paused at step" in last_train
    s["train_pausing"] = "train" in s["running"] and (P.runs / s["running"]["train"]["params"]["name"] / "PAUSE").exists()
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
        out["compares"] = [compare_summary(d) for d in sorted(run.glob("compare_*"), key=lambda d: d.stat().st_mtime,
                                                                 reverse=True) if d.is_dir()]
    if history:
        stride = max(1, len(rows) // 1500)
        out["history"] = rows[::stride]
        out["config"] = cfg
    return out


def preview_prompts(P: Paths) -> list[str]:
    """The prompts behind the preview sheets, as stored in the text cache."""
    extra = P.cache / "extra.safetensors"
    if not extra.exists():
        return []
    from safetensors import safe_open

    with safe_open(str(extra), framework="numpy") as f:
        return json.loads((f.metadata() or {}).get("previews", "[]"))


def compare_summary(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
    n = len(list(d.glob("base_*.png")))
    return {"dir": d.name, "meta": meta, "pairs": [[f"base_{i}.png" if (d / f"base_{i}.png").exists() else None,
                                                   f"lora_{i}.png" if (d / f"lora_{i}.png").exists() else None]
                                                  for i in range(1, n + 1)]}


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


# ---------- settings ----------

def settings_view(P: Paths) -> dict:
    """The saved settings plus what each empty one resolves to right now."""
    s = settings.load()
    repo, models, cap = settings.iris_repo(s), settings.models_dir(s), settings.captioner(s)
    return {"settings": s, "file": str(settings.FILE), "defaults": settings.DEFAULTS, "resolved": {
        "data_dir": str(P.data), "python": job_python(), "iris_repo": str(repo), "models_dir": str(models),
        "captioner": str(cap)}, "checks": {
        "iris_repo": (repo / "src" / "iris3b").is_dir(),
        "iris_weights": (models / "diffusion_models" / "iris-3b.safetensors").exists(),
        "text_encoder": (models / "text_encoders" / "Qwen3-VL-4B-Instruct" / "config.json").exists(),
        "captioner": (cap / "config.json").exists(),
        "python": Path(job_python()).exists()}}


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
                if u.path == "/api/presets":
                    return self.send(200, {k: {"params": v, "note": PRESET_NOTES.get(k, "")} for k, v in PRESETS.items()})
                if u.path == "/api/settings":
                    return self.send(200, settings_view(P))
                if u.path == "/api/log":
                    name = re.sub(r"[^a-z]", "", q.get("job", ""))
                    return self.send(200, {"text": tail(P.logs / f"{name}.log", int(q.get("lines", 200)))})
                if u.path == "/api/dataset":
                    return self.send(200, dataset(P, int(q.get("offset", 0)), int(q.get("limit", 60)), q.get("q", ""),
                                                  q.get("excluded") == "1"))
                if u.path == "/api/runs":
                    prompts = preview_prompts(P)
                    return self.send(200, [{**run_summary(r, history=True), "preview_prompts": prompts}
                                           for r in sorted(P.runs.iterdir()) if r.is_dir()])
                if u.path == "/thumb":
                    return self.file(thumbnail(P, q.get("path", "")))
                if u.path == "/image":
                    return self.file(allowed_file(P, q.get("path", "")))
                if u.path == "/preview":
                    run = P.runs / re.sub(r"[\\/:]", "", q.get("run", ""))
                    if q.get("dir"):  # a compare folder inside the run
                        run = run / re.sub(r"[\\/:]|\.\.", "", q["dir"])
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
                if u.path == "/api/jobs/pause":
                    return self.send(200, jobs.pause())
                if u.path == "/api/jobs/stop":
                    return self.send(200, jobs.stop(b.get("job", "")))
                if u.path == "/api/settings":
                    settings.save(b)
                    return self.send(200, {"ok": True, **settings_view(P)})
                if u.path == "/api/exclude":
                    return self.send(200, set_excluded(P, b["path"], bool(b.get("excluded", True))))
                return self.send(404, {"error": "not found"})
            except Exception as exc:
                return self.send(500, {"ok": False, "error": str(exc)})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="data folder (default: data_dir in settings.json, else ./data)")
    ap.add_argument("--port", type=int, default=None, help="default: port in settings.json, else 7861")
    ap.add_argument("--open", action="store_true", help="open the page in the browser")
    args = ap.parse_args()
    P = Paths(Path(args.data) if args.data else settings.data_dir())
    port = args.port or int(settings.load()["port"])
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(P, Jobs(P)))
    url = f"http://127.0.0.1:{port}"
    print(f"Iris dashboard on {url}  (data: {P.data}, settings: {settings.FILE})", flush=True)
    if args.open:
        import webbrowser

        threading.Timer(1.0, webbrowser.open, [url]).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
