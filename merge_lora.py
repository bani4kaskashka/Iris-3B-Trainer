"""Step 5: fold a trained LoRA into the base weights and save a new checkpoint.

    python merge_lora.py runs/anime/lora_010000.safetensors --name anime

writes E:\\ComfyUI\\models\\diffusion_models\\iris-3b-anime.safetensors, which the
ComfyUI Iris model loader picks up like the base model (same config).
"""

import argparse
import json

import common  # noqa: F401
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from common import BASE_WEIGHTS, MODELS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lora")
    ap.add_argument("--name", required=True, help="checkpoint becomes iris-3b-<name>.safetensors")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="bf16 = 6 GB, fp32 = 12 GB")
    ap.add_argument("--out", default=None, help="output file (default: ComfyUI diffusion_models/iris-3b-<name>.safetensors)")
    args = ap.parse_args()

    with safe_open(args.lora, framework="pt") as f:
        meta = f.metadata() or {}
    scale = args.strength * float(meta["alpha"]) / int(meta["rank"])
    lora = load_file(args.lora)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    out, merged = {}, 0
    with safe_open(str(BASE_WEIGHTS), framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key).float()
            prefix = key[: -len(".weight")] if key.endswith(".weight") else None
            if prefix and f"{prefix}.lora_down" in lora:
                t += scale * (lora[f"{prefix}.lora_up"].float() @ lora[f"{prefix}.lora_down"].float())
                merged += 1
            out[key] = t.to(dtype)
    pairs = sum(1 for k in lora if k.endswith(".lora_down"))
    if merged != pairs:
        raise SystemExit(f"only {merged} of {pairs} LoRA layers matched the base weights")
    target = args.out or MODELS / "diffusion_models" / f"iris-3b-{args.name}.safetensors"
    save_file(out, str(target), metadata={"merged_lora": args.lora, "strength": str(args.strength),
                                          "lora_meta": json.dumps(meta)})
    print(f"merged {merged} layers (scale {scale:g}) -> {target}")


if __name__ == "__main__":
    main()
