"""Download what the trainer needs into the models folder (skips files that are already there).

    python download_models.py                  # Iris-3B + its text encoder
    python download_models.py --captioner      # also the abliterated Qwen3-VL captioner (fewer refusals)

Layout (the same one ComfyUI uses, so the Comfy nodes can share it):
    <models>/diffusion_models/iris-3b.safetensors
    <models>/text_encoders/Qwen3-VL-4B-Instruct/
    <models>/text_encoders/Huihui-Qwen3-VL-4B-Instruct-abliterated/
"""

import argparse
import shutil
from pathlib import Path

import settings

IRIS = ("speridlabs/iris-3b", "model.safetensors", "diffusion_models/iris-3b.safetensors")
FOLDERS = {
    "text encoder": ("Qwen/Qwen3-VL-4B-Instruct", "text_encoders/Qwen3-VL-4B-Instruct"),
    "captioner": ("huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
                  "text_encoders/Huihui-Qwen3-VL-4B-Instruct-abliterated"),
}


def complete(folder: Path) -> bool:
    return (folder / "config.json").exists() and (folder / "model.safetensors.index.json").exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(settings.models_dir()))
    ap.add_argument("--captioner", action="store_true", help="also download the abliterated captioner")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    models = Path(args.models)
    repo, name, rel = IRIS
    target = models / rel
    if target.exists():
        print(f"have {target}")
    else:
        print(f"downloading {repo}/{name} (6 GB) -> {target}", flush=True)
        staging = target.parent / ".iris-download"
        got = hf_hub_download(repo, name, local_dir=staging)  # local_dir: no symlinks, they fail on many Windows setups
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(got, target)
        shutil.rmtree(staging, ignore_errors=True)
    wanted = ["text encoder"] + (["captioner"] if args.captioner else [])
    for what in wanted:
        repo, rel = FOLDERS[what]
        folder = models / rel
        if complete(folder):
            print(f"have {folder}")
            continue
        print(f"downloading the {what} {repo} (9 GB) -> {folder}", flush=True)
        snapshot_download(repo, local_dir=folder, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    print("done")


if __name__ == "__main__":
    main()
