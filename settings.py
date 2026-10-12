"""Machine settings and captioning options, kept in settings.json next to these scripts.

Every script reads it, the dashboard edits it (Settings tab). Anything left empty
falls back to a default inside this folder, so a fresh clone works without one.
IRIS_SETTINGS points at a different file.
"""

import copy
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILE = Path(os.environ.get("IRIS_SETTINGS", HERE / "settings.json"))

CAPTION_PROMPT = (
    "You are captioning images to train an image generator. Describe this picture in 2 to 4 natural sentences. "
    "Start with what kind of image it is (anime illustration, 2.5D render, CG, sketch, painting, screenshot...) "
    "only if it adds something, then cover whatever matters most here: the characters (call people women or men) "
    "and how they look, clothing or nudity, pose and expression, what they are doing including any sexual acts "
    "in plain explicit words, composition and camera angle, background, lighting, colors and mood. "
    "Write it the way a person would, vary your sentence structure, and don't open with 'This image' or "
    "'The image'. Do not guess names. Tags from the image's source, for reference: {tags}"
)

# tags that say nothing about the picture's content
SKIP_TAGS = ["highres", "absurdres", "lowres", "commentary", "commentary_request", "english_commentary", "translated",
             "bad_id", "bad_pixiv_id", "bad_twitter_id", "md5_mismatch", "artist_name", "signature", "watermark",
             "web_address", "patreon_username", "twitter_username", "censored", "uncensored", "jpeg_artifacts"]

DEFAULTS = {
    "data_dir": "",      # empty: ./data
    "python": "",        # Python that runs the jobs; empty: the one running the dashboard
    "iris_repo": "",     # clone of github.com/speridlabs/iris-3b; empty: ./iris-3b or ../iris-3b
    "models_dir": "",    # ComfyUI-style models folder; empty: ./models
    "extra_path": [],    # folders put in front of PATH for jobs (ROCm tools and so on)
    "env": {},           # extra environment variables for jobs
    "port": 7861,
    "caption": {
        "captioner": "",           # Qwen3-VL folder; empty: Iris's own text encoder
        "mode": "vlm",             # vlm, or tags (no VLM, captions are the tags)
        "prompt": CAPTION_PROMPT,  # {tags} becomes the image's tags
        "tags_as_hints": True,     # show the tags to the VLM
        "append_tags": True,       # put the tags after the caption
        "max_tags": 30,
        "skip_tags": SKIP_TAGS,    # never shown or appended
        "drop_words": [],          # captions with any of these words are dropped (on top of the built-in filter)
        "remove_phrases": [],      # cut out of every caption, e.g. "The image shows"
        "rewrite_girl_boy": True,  # prose "girl"/"boy" becomes "woman"/"man"; booru tags (1girl...) stay as they are
        "prefix": "",              # text in front of every caption, e.g. a trigger word
        "max_side": 768,           # image size the VLM sees
        "max_new_tokens": 170,
        "temperature": 0.7,        # 0 = greedy, the same wording style every time
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    },
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load() -> dict:
    return _merge(DEFAULTS, json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {})


def save(new: dict) -> dict:
    s = _merge(load(), new)
    if "env" in new:  # replaced, so variables can be removed
        s["env"] = dict(new["env"])
    FILE.write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    return s


def _path(value: str, default: Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))) if value else default


def data_dir(s=None) -> Path:
    s = s or load()
    return _path(os.environ.get("IRIS_DATA") or s["data_dir"], HERE / "data")


def models_dir(s=None) -> Path:
    s = s or load()
    return _path(os.environ.get("IRIS_MODELS") or s["models_dir"], HERE / "models")


def iris_repo(s=None) -> Path:
    s = s or load()
    if value := os.environ.get("IRIS_REPO") or s["iris_repo"]:
        return _path(value, HERE)
    return next((p for p in (HERE / "iris-3b", HERE.parent / "iris-3b") if (p / "src" / "iris3b").is_dir()),
                HERE / "iris-3b")


def captioner(s=None) -> Path:
    """The set captioner, else the abliterated one if downloaded, else Iris's own text encoder."""
    s = s or load()
    tes = models_dir(s) / "text_encoders"
    fallback = next((p for p in (tes / "Huihui-Qwen3-VL-4B-Instruct-abliterated",) if (p / "config.json").exists()),
                    tes / "Qwen3-VL-4B-Instruct")
    return _path(s["caption"]["captioner"], fallback)
