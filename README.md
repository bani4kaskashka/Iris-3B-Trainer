# Iris-3B LoRA Trainer

Train LoRAs (and merged checkpoints) for [Iris-3B](https://huggingface.co/speridlabs/iris-3b) on a single consumer GPU. It uses upstream's own training objective, so a LoRA learns the same way Iris itself was trained.

Results load in [Iris-3B Comfy Nodes](https://github.com/bani4kaskashka/Iris-3B-Comfy-Nodes) with **Load Iris LoRA**, or as a merged checkpoint in **Load Iris-3B Model**.

Tested on an AMD RX 9060 XT (16 GB, ROCm on Windows): **about 4.3 s per step at rank 64, peak 9.4 GB VRAM**. It should work the same on NVIDIA.

## Quick start (Windows)

```
git clone https://github.com/bani4kaskashka/Iris-3B-Trainer
```

Double-click **start.bat**. The first run:

1. makes a Python environment in `.venv` (needs Python 3.12 or 3.13 from python.org),
2. asks which GPU you have and installs the matching PyTorch (NVIDIA CUDA, or AMD ROCm for RX 7000 / RX 9000),
3. installs the rest of `requirements.txt`,
4. fetches the upstream Iris-3B code into `iris-3b/` (needs Git),

and then opens the dashboard on http://127.0.0.1:7861. Later runs go straight to the dashboard. Press **Start** on the **Models** card once to download Iris-3B, its text encoder and the abliterated captioner, or point `models_dir` in the Settings tab at a ComfyUI models folder that already has them.

Already have a Python with PyTorch (a ComfyUI venv works)? Set `IRIS_PYTHON` to it before running start.bat, or run `python dashboard.py --open` with it after `pip install -r requirements.txt`.

## Dashboard

Everything runs from the dashboard, step by step: **Models, Download, Prep, Captions, Text cache, Train, Merge**. Each card shows progress and ETA, starts and stops its job and has its own settings. Jobs run detached, so they keep going if you close the page or restart the dashboard, and it picks them back up. Only one GPU job runs at a time. It binds to 127.0.0.1 only.

- **Dataset**: browse every image with its caption and tags, full-size viewer (arrow keys to move, X to exclude from training). Thumbnails are blurred until you untick it.
- **Training**: loss chart, preview sheets (click one for full size, 1-3 for a single image), and **Compare base vs LoRA**: pick a saved LoRA, write your prompts, and it renders each prompt with the base model and with the LoRA on the same seed, side by side.
- **Logs**: live log of every job.
- **Settings**: paths, extra PATH folders and environment variables for jobs, and every captioning option (see below). Saved to `settings.json`.

### Training presets

The Train card has presets: **Quick test** (rank 32, 1500 steps), **Character or style** (rank 32, 3000), **Balanced** (rank 64, 8000), **Big dataset** (rank 64, 20000), **High capacity** (rank 128, needs about 15 GB free VRAM) and **Keep training a LoRA** (continue from a finished LoRA of the same rank with `init_lora`). Every value can still be changed by hand.

**Pause** saves at the end of the current step and stops; **Start** with `resume` on continues from there. Each run also renders a step-0 preview with the untouched base model, to compare the later previews against.

### Captioning options

All in the Settings tab: the captioner model, the prompt it gets (`{tags}` becomes the image's tags), whether tags are shown as hints and appended, how many, which tags to skip, extra **drop words** (captions containing them are dropped), **phrases to cut** from every caption, the girl/boy to woman/man prose rewrite (booru tags like 1girl always stay), a prefix or trigger word, the image size the captioner sees, and sampling (temperature, top_p, repetition penalty, length). Dropped images are remembered, and an earlier drop comes back on its own if its saved caption passes your current settings.

## Command line

Every step is also a plain script; they all read `settings.json`.

```
python download_models.py --captioner
python prep_dataset.py --images D:\data\my_set D:\data\more --out work/manifest.jsonl
python caption.py --manifest work/manifest.jsonl --out work/captioned.jsonl
python encode_text.py --captions work/captioned.jsonl --cache work/text_cache
python train_lora.py --captions work/captioned.jsonl --cache work/text_cache --out runs/my_lora --steps 8000
python compare_lora.py runs/my_lora/lora_002000.safetensors --steps 50
python merge_lora.py runs/my_lora/lora_008000.safetensors --name my_model
```

1. **prep_dataset.py** scans image folders and their tags (`name.txt` booru tags, `name.json` danbooru metadata, or the tags in yande.re / konachan file names), filters by size, rating and duplicates, and sorts every image into a ~1 MP aspect bucket. `--scale 0.75` trains at about 768 px instead.
2. **caption.py** writes a natural-language caption per image with a Qwen3-VL model (Iris's text encoder was trained on those) and appends the tags. About 8 s per image on the 9060 XT. Resumable.
3. **encode_text.py** runs Iris's text encoder over every caption once and caches the result, so training never loads it. Plan for disk space: about 10 MB per caption. Finished shards are reused when you add more captions later.
4. **train_lora.py** trains the LoRA on the transformer blocks, pixel head and final layer, AdamW with warmup and cosine decay, 10% caption dropout so CFG keeps working, gradient checkpointing, a preview sheet and save every few hundred steps. Captions that aren't in the text cache yet are skipped, so you can train on what's done while captioning continues. `--resume` continues from the last save, `--init-lora` starts from a finished LoRA.
5. **compare_lora.py** renders the same prompts and seeds with and without a LoRA.
6. **merge_lora.py** folds a LoRA into the base weights and writes `iris-3b-<name>.safetensors` (bf16, 6 GB) into your models folder.

Danbooru helpers: **analyze_meta.py** counts what a `posts.json` would give after each filter, **fetch_filtered.py** streams only the allowed images out of any Hugging Face dataset of danbooru-style tars (resumes dropped connections), and **convert_comfy_vlm.py** turns a ComfyUI single-file Qwen3-VL back into a Hugging Face folder.

## Content rule

`safety.py` drops anything that depicts or suggests a minor: hard-blocked tags (loli, shota, child and more), suspicious tag combinations, and any generated caption that mentions a child, teen, student or an age. Explicit and suggestive images are also dropped when they're school-coded (uniforms, classrooms and so on) or come from student-cast series or canonically underage characters; clean general-rated images keep their school uniforms. Adult content of any kind is kept. It runs in prep, fetching and captioning and has no off switch. You can add tags to block in `blocked_tags_extra.txt` and words in the caption settings, but not remove built-in ones.

## How much data

For a style or a character, a few hundred to a few thousand captioned images. For teaching the model concepts it barely knows, think 10k to 30k good images and 10k+ steps. Keep images at 832 px or more on the short side; don't train at 512 px, where Iris shows block artifacts. Iris works in pixels with no VAE, so it learns JPEG artifacts and blur straight from the data: use the cleanest, largest files you have.

## Credits

Made by Bucky ([bani4kaskashka](https://github.com/bani4kaskashka)). Iris-3B and the training objective are by [Speridlabs](https://github.com/speridlabs/iris-3b) (Apache-2.0). The LoRA file layout matches [pentaCoxian's Iris Turbo](https://github.com/pentaCoxian/iris-3b-turbo) adapters. Apache-2.0.
