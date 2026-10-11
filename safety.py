"""Hard content filter: nothing that depicts or suggests a minor enters the dataset.

This runs in prep_dataset.py (on tags) and again in caption.py (on the generated
captions). There is deliberately no switch to turn it off. Extra blocked tags can
be added in blocked_tags_extra.txt next to this file (one per line); nothing can
be removed from the built-in lists.
"""

import re
from pathlib import Path

# danbooru-style tags (underscores or spaces both match)
HARD_BLOCK = {
    "loli", "lolicon", "shota", "shotacon", "child", "children", "kid", "kids", "toddler", "infant", "baby",
    "female_child", "male_child", "little_girl", "little_boy", "young_girl", "young_boy", "minor",
    "aged_down", "younger", "child_on_child", "kindergarten", "kindergartener", "kindergarten_uniform",
    "elementary_school", "elementary_school_student", "middle_school", "middle_school_student",
    "randoseru", "school_uniform", "serafuku", "school_swimsuit", "gym_uniform", "bloomers",
    "high_school_student", "student", "schoolgirl", "schoolboy", "teen", "teenager", "underage",
    "cub", "mesugaki", "onee-shota", "oppai_loli",
}
# body tags that are only blocked in combination (petite adult art exists, but these
# pairs are the usual way minor-coded images get around the hard tags)
COMBO_BLOCK = [
    ({"flat_chest"}, {"petite", "short", "small", "small_stature", "short_stature", "tiny", "chibi"}),
    ({"chibi"}, {"nude", "sex", "nipples", "pussy", "penis", "explicit"}),
]
CAPTION_BLOCK = re.compile(
    r"\b(child|children|kid|kids|toddler|infant|baby|minor|underage|loli|lolicon|shota|teen|teens|teenage|"
    r"teenager|preteen|pre-teen|schoolgirl|schoolboy|school\s*girl|school\s*boy|little\s+girl|little\s+boy|"
    r"young\s+girl|young\s+boy|girl\s+of\s+\d|boy\s+of\s+\d|\d{1,2}\s*-?\s*years?\s*-?\s*old|elementary|"
    r"middle\s+school|high\s+school|kindergarten|student|pupil|youthful|childlike|child-like|"
    r"adolescent|juvenile|prepubescent|pubescent)\b",
    re.I,
)


def _norm(tag: str) -> str:
    return tag.strip().lower().replace(" ", "_")


def _extra() -> set[str]:
    path = Path(__file__).with_name("blocked_tags_extra.txt")
    if not path.exists():
        return set()
    return {_norm(t) for t in path.read_text(encoding="utf-8").splitlines() if t.strip() and not t.startswith("#")}


_BLOCK = {_norm(t) for t in HARD_BLOCK} | _extra()


def tags_blocked(tags) -> str | None:
    """The reason a tag set is rejected, or None if it passes."""
    tags = {_norm(t) for t in tags}
    hit = tags & _BLOCK
    if hit:
        return f"tag {sorted(hit)[0]}"
    for need, with_any in COMBO_BLOCK:
        if need <= tags and tags & with_any:
            return f"combo {sorted(need)[0]}+{sorted(tags & with_any)[0]}"
    return None


def caption_blocked(text: str) -> str | None:
    m = CAPTION_BLOCK.search(text or "")
    return f"caption word '{m.group(0)}'" if m else None
