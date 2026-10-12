"""Shared paths and setup for the Iris-3B LoRA pipeline."""

import os
import sys
from pathlib import Path

# AOTriton flash attention on ROCm must be switched on before torch loads
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import settings

HERE = Path(__file__).resolve().parent
# upstream clone (github.com/speridlabs/iris-3b): model, flow loss and text encoder code
IRIS_REPO = settings.iris_repo()
if not (IRIS_REPO / "src" / "iris3b").is_dir():
    raise SystemExit(f"upstream iris-3b not found at {IRIS_REPO}; run start.bat once, or set iris_repo in the "
                     "dashboard's Settings tab")
sys.path.insert(0, str(IRIS_REPO / "src"))

# a ComfyUI-style models folder: diffusion_models/iris-3b.safetensors, text_encoders/Qwen3-VL-4B-Instruct
MODELS = settings.models_dir()
BASE_WEIGHTS = MODELS / "diffusion_models" / "iris-3b.safetensors"
# the release config.yaml; the copy next to these scripts is used when the clone has none
BASE_CONFIG = next((p for p in (IRIS_REPO / "checkpoints" / "iris-3b" / "config.yaml", HERE / "iris-3b.yaml") if p.exists()),
                   HERE / "iris-3b.yaml")
TEXT_ENCODER = MODELS / "text_encoders" / "Qwen3-VL-4B-Instruct"

# ~1 MP training buckets, multiples of 16 (the sizes Iris was fine-tuned around)
BUCKETS = [(1024, 1024), (1120, 896), (896, 1120), (1152, 864), (864, 1152), (1216, 832), (832, 1216),
           (1344, 768), (768, 1344), (1536, 640), (640, 1536)]


def bucket_for(width: int, height: int, scale: float = 1.0) -> tuple[int, int]:
    """Closest aspect bucket; ``scale`` < 1 shrinks every bucket (e.g. 0.75 for ~768 px training)."""
    ar = width / height
    w, h = min(BUCKETS, key=lambda b: abs((b[0] / b[1]) / ar - 1))
    return max(256, round(w * scale / 16) * 16), max(256, round(h * scale / 16) * 16)


def load_base_config():
    from omegaconf import OmegaConf

    from iris3b.config import inference_config

    return inference_config(OmegaConf.to_container(OmegaConf.load(BASE_CONFIG)))


def read_jsonl(path):
    import json

    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    import json

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
