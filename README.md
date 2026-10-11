# Iris-3B LoRA Trainer

Train LoRAs (and merged checkpoints) for [Iris-3B](https://huggingface.co/speridlabs/iris-3b) on a single consumer GPU. It uses upstream's own training objective, so a LoRA learns the same way Iris itself was trained.

Results load in [Iris-3B Comfy Nodes](https://github.com/bani4kaskashka/Iris-3B-Comfy-Nodes) with **Load Iris LoRA**, or as a merged checkpoint in **Load Iris-3B Model**.

Tested on an AMD RX 9060 XT (16 GB, ROCm on Windows): **about 3.8 s per step at 832x1216, peak 9.2 GB VRAM**. It should work the same on NVIDIA.

## Setup

```
git clone https://github.com/bani4kaskashka/Iris-3B-Trainer
```

1. Clone [speridlabs/iris-3b](https://github.com/speridlabs/iris-3b) next to this folder (or point `IRIS_REPO` at it). The trainer imports its model, flow loss and text encoder code.
2. Use a Python with PyTorch (a ComfyUI venv works), then `pip install -r requirements.txt`.
3. Point `IRIS_MODELS` at a ComfyUI-style models folder with:
   - `diffusion_models/iris-3b.safetensors` (upstream's `model.safetensors`)
   - `text_encoders/Qwen3-VL-4B-Instruct/` (the whole HF repo)

The Comfy pack downloads both on first use, so if you already run it, point `IRIS_MODELS` at ComfyUI's `models` folder.

## The five steps

```
python prep_dataset.py --images D:\data\my_set --out work/manifest.jsonl
python caption.py --manifest work/manifest.jsonl --out work/captioned.jsonl
python encode_text.py --captions work/captioned.jsonl --cache work/text_cache
python train_lora.py --captions work/captioned.jsonl --cache work/text_cache --out runs/my_lora --steps 10000
python merge_lora.py runs/my_lora/lora_010000.safetensors --name my_model
```

1. **prep_dataset.py** scans images and their sidecar tags (`name.txt` booru tags or `name.json` danbooru metadata), filters by size, rating and duplicates, and sorts every image into a ~1 MP aspect bucket. `--scale 0.75` trains at about 768 px instead.
2. **caption.py** writes a natural-language caption per image with a Qwen3-VL model (Iris's text encoder was trained on those) and appends the tags. `--vlm` takes any Qwen3-VL folder; `--mode tags` skips the VLM. On the 9060 XT it's about 8 s per image. Resumable.
3. **encode_text.py** runs Iris's text encoder over every caption once and caches the result, so training never loads it. Plan for disk space: about 7 MB per caption, so 10k images is roughly 70 GB.
4. **train_lora.py** trains the LoRA. Rank 64 by default on the transformer blocks, pixel head and final layer (112M params), AdamW with warmup and cosine decay, 10% caption dropout so CFG keeps working, gradient checkpointing, and a preview sheet and save every 500 steps. `--resume` continues from the last save.
5. **merge_lora.py** folds a LoRA into the base weights and writes `iris-3b-<name>.safetensors` (bf16, 6 GB) into your models folder.

Useful options: `--rank` (128 is a good default for big datasets, 64 for small ones), `--lr` (1e-4 at rank 64, 8e-5 at 128), `--init-lora` (start from a finished LoRA with a fresh schedule, to keep training it on more or new images), `--batch` / `--accum`, `--steps`, `--checkpointing` (`full` saves the most memory), `--flip` (only if your images have no text).

## Dashboard

```
python dashboard.py --data D:\Iris-data
```

Opens on http://127.0.0.1:7861 (local only). It shows the progress of every step with ETAs, starts and stops each job with its settings, follows the logs, charts the training loss with the preview images, and has a dataset browser where you can read captions and exclude images from training: full-size viewer, arrow keys to move, X to exclude (thumbnails are blurred until you untick it). Jobs run detached: they keep going if you close the page or restart the dashboard, and it picks them back up. Only one GPU job runs at a time.

Everything it does goes through a small JSON API (`/api/status`, `/api/jobs/start`, `/api/jobs/stop`, `/api/log`, `/api/dataset`, `/api/exclude`, `/api/runs`), so you can script it too.

## Danbooru metadata tools

- **analyze_meta.py** counts what a Danbooru `posts.json` would give after each filter.
- **fetch_filtered.py** takes any Hugging Face dataset link (`--repo`) with danbooru-style tar batches and a posts metadata file (`--meta-file`). It finds the tars and their sizes itself and streams them, and saves only the allowed images, each with a json sidecar of its tags. Rejected files are never written to disk, and a dropped connection resumes where it left off.
- **convert_comfy_vlm.py** turns a ComfyUI single-file Qwen3-VL (fp8_scaled, with its vision tower) back into a Hugging Face folder that `caption.py` can use.

## Content rule

`safety.py` drops anything that depicts or suggests a minor: hard-blocked tags (loli, shota, child, school uniforms and more), suspicious tag combinations, and any generated caption that mentions a child, teen, student or an age. Explicit and suggestive images are also dropped when they're school-coded (uniforms, classrooms and so on) or come from student-cast series or canonically underage characters; clean general-rated images keep their school uniforms. Captions call adult characters women and men. It runs in prep, fetching and captioning and has no off switch. You can add tags to block in `blocked_tags_extra.txt`, but not remove built-in ones.

## How much data

For a style or a character, a few hundred to a few thousand captioned images. For teaching the model concepts it barely knows, think 10k to 30k good images and 10k+ steps (about 10 hours at 1024 px). Keep images at 832 px or more on the short side; don't train at 512 px, where Iris shows block artifacts.

## Credits

Made by Bucky ([bani4kaskashka](https://github.com/bani4kaskashka)). Iris-3B and the training objective are by [Speridlabs](https://github.com/speridlabs/iris-3b) (Apache-2.0). The LoRA file layout matches [pentaCoxian's Iris Turbo](https://github.com/pentaCoxian/iris-3b-turbo) adapters. Apache-2.0.
