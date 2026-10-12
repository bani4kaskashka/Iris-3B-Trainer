"""Base model vs LoRA on the same prompts and seeds, as one comparison sheet.

    python compare_lora.py runs/my-lora/lora_001000.safetensors
    python compare_lora.py runs/my-lora/lora_001000.safetensors --prompts my_prompts.txt --steps 50

Top row is the base model, bottom row the LoRA. Every column uses the same prompt
and seed, so any difference comes from the LoRA. --prompts takes a text file with
one prompt per line. Writes compare_<lora name>_<time>/ next to the LoRA: the sheet,
each image on its own, prompts.txt and meta.json (the dashboard's Training tab shows them).
"""

import argparse
import json
import time
from pathlib import Path

import common  # noqa: F401  (paths, AOTriton)
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file

PROMPTS = [
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


def encode(prompts, device):
    from encode_text import build_encoder, trimmed

    enc = build_encoder(device)
    cond = [trimmed(enc.encode([p]))[0] for p in prompts]
    null = trimmed(enc.null(""))[0]
    del enc
    torch.cuda.empty_cache()
    return cond, null


def pad(emb, text_len):
    n = min(emb.shape[0], text_len)
    y = torch.zeros(text_len, *emb.shape[1:], dtype=emb.dtype)
    y[:n] = emb[:n]
    mask = torch.zeros(text_len, dtype=torch.long)
    mask[:n] = 1
    return y[None], mask[None]


@torch.no_grad()
def render(model, cfg, cond, null, seed, steps, cfg_scale, size, device):
    from iris3b.flow.solver import FlowDPMSolver

    y, m = (t.to(device) for t in pad(cond, cfg.model.text_len))
    uy, um = (t.to(device) for t in pad(null, cfg.model.text_len))
    cfg_mask = torch.cat([um, m])

    def fn(x, t, yy):
        return model(x.to(torch.bfloat16), t, yy, y_mask=m if yy is y else cfg_mask).x.float()

    solver = FlowDPMSolver(fn, num_timesteps=cfg.flow.num_train_timesteps, cfg_scale=cfg_scale,
                           prediction=cfg.flow.prediction)
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(1, 3, size[1], size[0], generator=g, device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        x = solver.sample(z, y, uy, steps=steps, order=2, shift=cfg.flow.shift).clamp(-1, 1)
    return Image.fromarray(((x[0] + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy())


def main():
    from train_lora import inject, load_base

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lora")
    ap.add_argument("--prompts", default=None, help="text file, one prompt per line")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=100, help="column i uses seed + i")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = PROMPTS
    if args.prompts:
        prompts = [line.strip() for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    size = tuple(int(v) for v in args.size.lower().split("x"))
    lora_path = Path(args.lora)
    out = Path(args.out) if args.out else lora_path.parent / f"compare_{lora_path.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    cond, null = encode(prompts, device)
    model, cfg = load_base(device)
    with safe_open(str(lora_path), framework="pt") as f:
        meta = f.metadata() or {}
    extra = json.loads(meta.get("extra", "{}"))
    rank, alpha = int(meta["rank"]), float(meta.get("alpha", meta["rank"]))
    layers = inject(model, extra.get("targets", "blocks,pixel_blocks,final_layer").split(","), rank, alpha)
    weights = load_file(str(lora_path))
    for prefix, layer in layers.items():
        layer.down.weight.data.copy_(weights[f"{prefix}.lora_down"])
        layer.up.weight.data.copy_(weights[f"{prefix}.lora_up"])
    model.train(False)

    rows = {}
    for label, scale in (("base", 0.0), ("lora", alpha / rank * args.strength)):
        for layer in layers.values():
            layer.scale = scale  # 0 turns the LoRA off exactly: the base model
        rows[label] = []
        for i, p in enumerate(prompts):
            im = render(model, cfg, cond[i], null, args.seed + i, args.steps, args.cfg, size, device)
            im.save(out / f"{label}_{i + 1}.png")
            rows[label].append(im)
            print(f"{label} {i + 1}/{len(prompts)}", flush=True)

    w, h, bar = size[0], size[1], 48
    sheet = Image.new("RGB", (w * len(prompts), (h + bar) * 2), (16, 17, 21))
    draw = ImageDraw.Draw(sheet)
    for r, label in enumerate(("base", "lora")):
        for i, im in enumerate(rows[label]):
            sheet.paste(im, (i * w, r * (h + bar) + bar))
            draw.text((i * w + 12, r * (h + bar) + 14), f"{label.upper()}  {i + 1}  seed {args.seed + i}",
                      fill=(230, 232, 238))
    sheet.save(out / "sheet.png")
    (out / "prompts.txt").write_text(
        f"lora: {lora_path}\nsteps {args.steps}, cfg {args.cfg}, strength {args.strength}, size {args.size}\n\n"
        + "\n".join(f"{i + 1} (seed {args.seed + i}): {p}" for i, p in enumerate(prompts)) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "lora": str(lora_path), "steps": args.steps, "cfg": args.cfg, "strength": args.strength, "size": args.size,
        "prompts": [{"prompt": p, "seed": args.seed + i} for i, p in enumerate(prompts)]}, indent=1), encoding="utf-8")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
