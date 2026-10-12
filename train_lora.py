"""Step 4: train a LoRA on Iris-3B.

    python train_lora.py --captions work/captioned.jsonl --cache work/text_cache --out runs/anime

Uses upstream's own objective (rectified-flow v-prediction, logit-normal timesteps
on the shift-4 schedule) with 10% caption dropout so CFG keeps working. The base
model stays frozen in bf16; only the LoRA (fp32) trains. Gradient checkpointing
keeps 1024 px at batch 1 inside 16 GB.

Saves runs/<name>/lora_<step>.safetensors (same layout as the Iris Turbo LoRA:
"<layer>.lora_down" / "<layer>.lora_up", rank/alpha metadata), a resumable
state file, and preview images every --preview-every steps.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import common  # noqa: F401  (paths, AOTriton, upstream src on sys.path)
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from common import BASE_WEIGHTS, load_base_config, read_jsonl

DEFAULT_TARGETS = ("blocks", "pixel_blocks", "final_layer")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # starts as an exact no-op
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


def inject(model: nn.Module, targets, rank: int, alpha: float) -> dict[str, LoRALinear]:
    wrapped = {}

    def walk(module: nn.Module, prefix: str):
        for name, child in module.named_children():
            path = f"{prefix}.{name}"
            if isinstance(child, nn.Linear):
                layer = LoRALinear(child, rank, alpha)
                setattr(module, name, layer)
                wrapped[path] = layer
            elif not isinstance(child, LoRALinear):
                walk(child, path)

    for t in targets:
        walk(model.get_submodule(t), t)
    return wrapped


def lora_state(layers: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    out = {}
    for prefix, layer in layers.items():
        out[f"{prefix}.lora_down"] = layer.down.weight.detach().float().cpu().contiguous()
        out[f"{prefix}.lora_up"] = layer.up.weight.detach().float().cpu().contiguous()
    return out


def load_base(device):
    from iris3b.models.dit import IrisDiT

    cfg = load_base_config()
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    with safe_open(str(BASE_WEIGHTS), framework="pt", device="cpu") as f:
        weights = {k: f.get_tensor(k).to(torch.bfloat16) for k in f.keys()}
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model.requires_grad_(False)
    return model.to(device), cfg


class TextCache:
    def __init__(self, cache_dir: Path, text_len: int):
        self.dir, self.text_len, self.files = cache_dir, text_len, {}
        self.index = {r["sample"]: r for r in read_jsonl(cache_dir / "index.jsonl")}
        extra = load_file(str(cache_dir / "extra.safetensors"))
        with safe_open(str(cache_dir / "extra.safetensors"), framework="pt") as f:
            self.preview_prompts = json.loads((f.metadata() or {}).get("previews", "[]"))
        self.null = self.pad(extra["null"])
        self.previews = [self.pad(extra[f"preview{i}"]) for i in range(len(self.preview_prompts))]

    def pad(self, emb: torch.Tensor):
        n = min(emb.shape[0], self.text_len)
        y = torch.zeros(self.text_len, *emb.shape[1:], dtype=emb.dtype)
        y[:n] = emb[:n]
        mask = torch.zeros(self.text_len, dtype=torch.long)
        mask[:n] = 1
        return y, mask

    def get(self, sample: int):
        r = self.index[sample]
        if r["shard"] not in self.files:
            if len(self.files) > 8:
                self.files.pop(next(iter(self.files)))
            self.files[r["shard"]] = safe_open(str(self.dir / r["shard"]), framework="pt")
        return self.pad(self.files[r["shard"]].get_tensor(r["key"]))


def load_image(path: str, size: tuple[int, int], flip: bool) -> torch.Tensor:
    """Cover-resize to (w, h), random crop, [-1, 1] CHW."""
    w, h = size
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = max(w / im.width, h / im.height)
        rw, rh = max(w, round(im.width * scale)), max(h, round(im.height * scale))
        im = im.resize((rw, rh), Image.LANCZOS)
        left, top = random.randint(0, rw - w), random.randint(0, rh - h)
        im = im.crop((left, top, left + w, top + h))
        if flip and random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(h, w, 3)
    return x.permute(2, 0, 1).float() / 127.5 - 1.0


class Batches:
    """Endless batches of one bucket each, buckets drawn in proportion to their size."""

    def __init__(self, rows, batch: int, skip: set[str] = frozenset()):
        self.by_bucket = {}
        for i, r in enumerate(rows):  # i stays the row's index into the text cache
            if r["path"] not in skip:
                self.by_bucket.setdefault(tuple(r["bucket"]), []).append(i)
        self.batch, self.keys = batch, list(self.by_bucket)
        self.weights = [len(self.by_bucket[k]) for k in self.keys]
        self.queues = {k: [] for k in self.keys}

    def next(self):
        k = random.choices(self.keys, self.weights)[0]
        q = self.queues[k]
        if len(q) < self.batch:
            fresh = self.by_bucket[k][:]
            random.shuffle(fresh)
            q.extend(fresh)
        out = q[:self.batch]
        del q[:self.batch]
        return k, out


@torch.no_grad()
def preview(model, cfg, cache: TextCache, device, path: Path, steps: int = 20, size=(1024, 1024)):
    from iris3b.flow.solver import FlowDPMSolver

    model.train(False)
    null_y, null_m = cache.null
    images = []
    for i, (y, m) in enumerate(cache.previews):
        cond, mask = y[None].to(device), m[None].to(device)
        uncond, umask = null_y[None].to(device), null_m[None].to(device)
        cfg_mask = torch.cat([umask, mask])

        def fn(x, t, yy, cond=cond, mask=mask, cfg_mask=cfg_mask):
            return model(x.to(torch.bfloat16), t, yy, y_mask=mask if yy is cond else cfg_mask).x.float()

        solver = FlowDPMSolver(fn, num_timesteps=cfg.flow.num_train_timesteps, cfg_scale=3.0, prediction=cfg.flow.prediction)
        g = torch.Generator(device=device).manual_seed(1234 + i)
        z = torch.randn(1, 3, size[1], size[0], generator=g, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            x = solver.sample(z, cond, uncond, steps=steps, order=2, shift=cfg.flow.shift).clamp(-1, 1)
        images.append(((x[0] + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())
    sheet = Image.new("RGB", (sum(im.shape[1] for im in images), images[0].shape[0]))
    xoff = 0
    for im in images:
        sheet.paste(Image.fromarray(im), (xoff, 0))
        xoff += im.shape[1]
    sheet.save(path)
    model.train(True)
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", default="work/captioned.jsonl")
    ap.add_argument("--cache", default="work/text_cache")
    ap.add_argument("--out", default="runs/anime")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=None, help="default = rank")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation (effective batch = batch * accum)")
    ap.add_argument("--text-dropout", type=float, default=0.1)
    ap.add_argument("--flip", action="store_true", help="random horizontal flips (avoid if images contain text)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--preview-every", type=int, default=500)
    ap.add_argument("--preview-steps", type=int, default=40, help="sampling steps for the preview images")
    ap.add_argument("--checkpointing", default="full", choices=["none", "full", "selective_op", "selective_layer"])
    ap.add_argument("--resume", action="store_true", help="continue from <out>/state.pt")
    ap.add_argument("--init-lora", default=None,
                    help="start from this LoRA file (same rank) with a fresh optimizer and schedule, e.g. to keep "
                         "training a finished LoRA on more or new images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", default=None, help="text file of image paths to leave out (one per line)")
    args = ap.parse_args()

    from iris3b.flow.transport import RectifiedFlow

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.captions)
    model, cfg = load_base(device)
    alpha = float(args.alpha or args.rank)
    layers = inject(model, [t.strip() for t in args.targets.split(",") if t.strip()], args.rank, alpha)
    params = [p for layer in layers.values() for p in (layer.down.weight, layer.up.weight)]
    for p in params:
        p.requires_grad_(True)
    print(f"LoRA rank {args.rank} on {len(layers)} layers, {sum(p.numel() for p in params) / 1e6:.1f}M trainable params")
    model.train(True)
    model.activation_checkpointing = args.checkpointing

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    step = 0
    if args.init_lora and not (args.resume and (out / "state.pt").exists()):
        with safe_open(args.init_lora, framework="pt") as f:
            init_meta = f.metadata() or {}
        if int(init_meta.get("rank", -1)) != args.rank:
            raise SystemExit(f"--init-lora has rank {init_meta.get('rank')}, but --rank is {args.rank}")
        init = load_file(args.init_lora)
        missing = [p for p in layers if f"{p}.lora_down" not in init]
        if missing:
            raise SystemExit(f"--init-lora lacks {len(missing)} layers (different --targets?), e.g. {missing[:3]}")
        for prefix, layer in layers.items():
            layer.down.weight.data.copy_(init[f"{prefix}.lora_down"])
            layer.up.weight.data.copy_(init[f"{prefix}.lora_up"])
        print(f"starting from {args.init_lora}")
    if args.resume and (out / "state.pt").exists():
        state = torch.load(out / "state.pt", map_location="cpu", weights_only=True)
        lora = state["lora"]
        for prefix, layer in layers.items():
            layer.down.weight.data.copy_(lora[f"{prefix}.lora_down"])
            layer.up.weight.data.copy_(lora[f"{prefix}.lora_up"])
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"resumed at step {step}")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        progress = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))  # cosine to 10%

    flow = RectifiedFlow(cfg.flow)
    cache = TextCache(Path(args.cache), cfg.model.text_len)
    skip = set()
    if args.exclude and Path(args.exclude).exists():
        skip = {l.strip() for l in Path(args.exclude).read_text(encoding="utf-8").splitlines() if l.strip()}
    # captions written after the last text cache run have no embedding yet, leave them out
    unencoded = {r["path"] for i, r in enumerate(rows) if i not in cache.index}
    batches = Batches(rows, args.batch, skip | unencoded)
    used = len(rows) - len((skip | unencoded) & {r["path"] for r in rows})
    print(f"{used} images to train on ({len(skip)} excluded, {len(unencoded)} not in the text cache yet)")
    meta = {"rank": str(args.rank), "alpha": str(alpha), "extra": json.dumps(
        {"base": "speridlabs/iris-3b", "targets": args.targets, "kind": "iris3b-lora"})}

    def save(tag):
        lora = lora_state(layers)
        save_file(lora, str(out / f"lora_{tag}.safetensors"), metadata=meta)
        torch.save({"lora": lora, "opt": opt.state_dict(), "step": step}, out / "state.pt")

    log = open(out / "log.csv", "a", encoding="utf-8")
    loss_ema, t0, t_last = None, time.time(), time.time()
    torch.cuda.reset_peak_memory_stats()
    pause = out / "PAUSE"  # the dashboard's Pause button creates it: save now and stop, --resume continues
    pause.unlink(missing_ok=True)
    if step == 0 and args.preview_every:  # what the model makes before any training, to compare against
        preview(model, cfg, cache, device, out / "preview_000000.png", steps=args.preview_steps)
    while step < args.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        total = 0.0
        for _ in range(args.accum):
            (w, h), idx = batches.next()
            x0 = torch.stack([load_image(rows[i]["path"], (w, h), args.flip) for i in idx]).to(device)
            ys, masks = zip(*[cache.null if random.random() < args.text_dropout else cache.get(i) for i in idx])
            y = torch.stack(ys).to(device)
            mask = torch.stack(masks).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = flow.training_loss(model, x0, y, model_kwargs={"y_mask": mask}).loss / args.accum
            loss.backward()
            total += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        loss_ema = total if loss_ema is None else 0.98 * loss_ema + 0.02 * total
        log.write(f"{step},{total:.5f},{loss_ema:.5f},{lr_at(step - 1):.2e}\n")
        if step % 10 == 0 or step == 1:
            dt = (time.time() - t_last) / (10 if step > 1 else 1)
            t_last = time.time()
            eta = dt * (args.steps - step) / 3600
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"step {step}/{args.steps}  loss {total:.4f} (avg {loss_ema:.4f})  lr {lr_at(step - 1):.1e}  "
                  f"{dt:.2f} s/step  peak {peak:.1f} GB  eta {eta:.1f} h", flush=True)
            log.flush()
        if step % args.save_every == 0 or step == args.steps:
            save(f"{step:06d}")
        if args.preview_every and (step % args.preview_every == 0 or step == args.steps):
            preview(model, cfg, cache, device, out / f"preview_{step:06d}.png", steps=args.preview_steps)
        if pause.exists():
            log.flush()
            save(f"{step:06d}")
            pause.unlink(missing_ok=True)
            print(f"paused at step {step}, saved; start with resume on to continue", flush=True)
            return
    print(f"finished {step} steps in {(time.time() - t0) / 3600:.2f} h -> {out}")


if __name__ == "__main__":
    main()
