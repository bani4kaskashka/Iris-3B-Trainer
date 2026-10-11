"""Filtering for danbooru post metadata, shared by analyze_meta.py, fetch_filtered.py and prep."""

from safety import minor_series_or_character, school_blocked, tags_blocked

IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")


def block_reason(post: dict, min_side: int = 832, min_score: int = 0, ratings=("e", "q", "s", "g")) -> str | None:
    """Why a danbooru post is left out, or None if it's allowed. The minors rules always apply."""
    tags = set((post.get("tag_string") or "").split())
    rating = post.get("rating")
    if post.get("is_deleted") or post.get("is_banned"):
        return "deleted or banned"
    if post.get("file_ext") not in IMAGE_EXTS:
        return "not a still image"
    if tags_blocked(tags):
        return "minors filter (loli/shota/child... tags)"
    if school_blocked(tags, rating):
        return "school-coded + not general-rated"
    if rating != "g" and minor_series_or_character(post):
        return "minor character or student-cast series + not general-rated"
    if rating not in ratings:
        return "rating"
    if min(post.get("image_width", 0), post.get("image_height", 0)) < min_side:
        return f"short side < {min_side} px"
    if post.get("score", 0) < min_score:
        return f"score < {min_score}"
    return None


def sidecar(post: dict) -> dict:
    """The metadata kept next to each extracted image (prep_dataset.py reads it)."""
    keys = ("id", "md5", "rating", "score", "image_width", "image_height", "tag_string", "tag_string_general",
            "tag_string_character", "tag_string_copyright", "tag_string_artist", "source")
    return {k: post.get(k) for k in keys}
