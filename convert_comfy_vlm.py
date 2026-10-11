"""Turn a ComfyUI single-file Qwen3-VL (fp8_scaled, with its vision tower) back into a
Hugging Face folder that caption.py can load.

    python convert_comfy_vlm.py E:\\ComfyUI\\models\\text_encoders\\Huihui-...-fp8_scaled.safetensors ^
        --like E:\\ComfyUI\\models\\text_encoders\\Qwen3-VL-4B-Instruct --out D:\\Iris-data\\models\\Huihui-Qwen3-VL-4B

fp8 weights are multiplied by their per-tensor scale back to bf16; config,
tokenizer and image preprocessor come from a stock folder of the same model.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comfy_file")
    ap.add_argument("--like", required=True, help="stock HF folder of the same architecture")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    like, out = Path(args.like), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    expected = set(json.loads((like / "model.safetensors.index.json").read_text())["weight_map"])
    tensors = {}
    with safe_open(args.comfy_file, framework="pt") as f:
        keys = set(f.keys())
        for k in sorted(keys):
            if k.endswith((".weight_scale", ".comfy_quant", ".input_scale")):
                continue
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                scale = f.get_tensor(k + "_scale").float() if k + "_scale" in keys else 1.0
                t = (t.float() * scale).to(torch.bfloat16)
            tensors[k] = t.contiguous()
    missing, extra = expected - set(tensors), set(tensors) - expected
    if missing or extra:
        raise SystemExit(f"layout mismatch: missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}")
    # two shards, like the stock repo, plus an index
    names = sorted(tensors)
    half = len(names) // 2
    weight_map = {}
    for i, part in enumerate((names[:half], names[half:]), 1):
        shard = f"model-0000{i}-of-00002.safetensors"
        save_file({k: tensors[k] for k in part}, str(out / shard), metadata={"format": "pt"})
        weight_map.update({k: shard for k in part})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    for p in like.iterdir():
        if p.is_file() and not p.name.startswith("model") and p.suffix in (".json", ".txt", ".jinja"):
            shutil.copy(p, out / p.name)
    print(f"wrote {len(tensors)} tensors ({total / 2**30:.1f} GB) to {out}")


if __name__ == "__main__":
    main()
