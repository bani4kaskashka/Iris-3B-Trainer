"""Count what a Danbooru posts.json (JSONL) would give after each filter tier.

    python analyze_meta.py data/meta/posts.json
"""

import json
import sys
from collections import Counter

from danbooru import block_reason

ADULT_HINTS = {"mature_female", "milf", "office_lady", "nurse", "teacher", "adult", "aged_up", "mother",
               "wife", "married_woman", "old", "mature_male", "muscular_female", "bodysuit", "business_suit"}


def main(path):
    total, by_rating, kept = 0, Counter(), []
    stage = Counter()
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        p = json.loads(line)
        total += 1
        by_rating[p.get("rating")] += 1
        tags = set(p.get("tag_string", "").split())
        rating = p.get("rating")
        reason = block_reason(p)
        if reason:
            stage[reason] += 1
            continue
        kept.append(p)

    print(f"posts: {total}")
    print("ratings:", ", ".join(f"{k}={v}" for k, v in by_rating.most_common()))
    print("\nremoved, in order:")
    for k, v in stage.items():
        print(f"  {v:>7}  {k}")
    print(f"\nleft: {len(kept)}")
    kr = Counter(p["rating"] for p in kept)
    print("  by rating:", ", ".join(f"{k}={v}" for k, v in kr.most_common()))
    for score in (10, 30, 100):
        print(f"  score >= {score}: {sum(1 for p in kept if p.get('score', 0) >= score)}")
    adult = sum(1 for p in kept if set(p['tag_string'].split()) & ADULT_HINTS)
    print(f"  with an explicit adult hint tag: {adult}")

    chars, series = Counter(), Counter()
    for p in kept:
        if p["rating"] in ("e", "q"):
            chars.update(p.get("tag_string_character", "").split())
            series.update(p.get("tag_string_copyright", "").split())
    print("\ntop characters left in explicit/questionable (check for minors):")
    print("  " + ", ".join(f"{c}:{n}" for c, n in chars.most_common(60)))
    print("\ntop series left in explicit/questionable:")
    print("  " + ", ".join(f"{c}:{n}" for c, n in series.most_common(40)))


if __name__ == "__main__":
    main(sys.argv[1])
