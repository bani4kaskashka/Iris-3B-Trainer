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
    "randoseru", "teen", "teenager", "underage", "cub", "mesugaki", "onee-shota", "oppai_loli",
}
# school-coded clothing and settings: fine in general-rated (clean) images, blocked in
# sensitive, questionable and explicit ones
SCHOOL_CODED = {
    "school_uniform", "serafuku", "sailor_collar", "neckerchief", "pleated_skirt", "plaid_skirt", "school_bag",
    "school_desk", "school_chair", "classroom", "chalkboard", "school", "school_hat", "school_swimsuit",
    "gym_uniform", "gym_shirt", "gym_shorts", "buruma", "bloomers", "high_school_student", "student",
    "schoolgirl", "schoolboy", "student_council", "school_briefcase", "track_jacket", "kneehighs",
}
# series whose main cast are students or otherwise underage (danbooru copyright tags)
MINOR_SERIES = {
    "blue_archive", "love_live!", "love_live!_sunshine!!", "love_live!_school_idol_project",
    "love_live!_nijigasaki_high_school_idol_club", "love_live!_superstar!!", "bang_dream!",
    "bang_dream!_it's_mygo!!!!!", "bang_dream!_ave_mujica", "danganronpa_(series)", "danganronpa_1", "danganronpa_2",
    "danganronpa_v3", "persona_3", "persona_4", "persona_5", "k-on!", "bocchi_the_rock!", "boku_no_hero_academia",
    "neon_genesis_evangelion", "evangelion:_3.0+1.0_thrice_upon_a_time", "rebuild_of_evangelion", "kill_la_kill",
    "oshi_no_ko", "lycoris_recoil", "sono_bisque_doll_wa_koi_wo_suru", "bishoujo_senshi_sailor_moon", "toradora!",
    "go-toubun_no_hanayome", "kaguya-sama_wa_kokurasetai_~tensai-tachi_no_renai_zunousen~", "uma_musume",
    "hibike!_euphonium", "yuru_yuri", "lucky_star", "suzumiya_haruhi_no_yuuutsu", "toaru_kagaku_no_railgun",
    "toaru_majutsu_no_index", "yahari_ore_no_seishun_lovecome_wa_machigatteiru.", "kimi_no_na_wa.",
    "chuunibyou_demo_koi_ga_shitai!", "kono_subarashii_sekai_ni_shukufuku_wo!", "zombie_land_saga",
    "seishun_buta_yarou", "nisekoi", "saenai_heroine_no_sodatekata", "idolmaster_cinderella_girls_u149",
    "the_idolm@ster:_cinderella_girls", "the_idolm@ster:_shiny_colors", "the_idolm@ster:_million_live!",
    "gakuen_idolmaster", "girls_und_panzer", "assault_lily", "sword_art_online", "darling_in_the_franxx",
    "made_in_abyss", "mahou_shoujo_madoka_magica", "precure", "pretty_cure", "ojamajo_doremi",
    "kantai_collection", "princess_connect!", "project_sekai", "hololive_dev_is", "touhou",
    "ijiranaide_nagatoro-san", "rwby", "tokyo_afterschool_summoners", "idolmaster", "idolmaster_cinderella_girls",
    "idolmaster_shiny_colors", "idolmaster_million_live!", "idolmaster_cinderella_girls_u149",
    "naruto_(series)", "naruto", "naruto_shippuuden", "kemono_friends",
}
# canonically underage characters from otherwise mixed-age series
MINOR_CHARACTERS = {
    "hatsune_miku", "kagamine_rin", "kagamine_len", "megumin", "kitagawa_marin", "anya_(spy_x_family)",
    "souryuu_asuka_langley", "ayanami_rei", "makinami_mari_illustrious", "kamado_nezuko", "kugisaki_nobara",
    "zen'in_maki", "klee_(genshin_impact)", "qiqi_(genshin_impact)", "nahida_(genshin_impact)",
    "diona_(genshin_impact)", "sayu_(genshin_impact)", "paimon_(genshin_impact)", "yaoyao_(genshin_impact)",
    "illyasviel_von_einzbern", "chloe_von_einzbern", "miyu_edelfelt", "abigail_williams_(fate)",
    "jack_the_ripper_(fate/apocrypha)", "nursery_rhyme_(fate)", "tohsaka_rin", "matou_sakura", "hoshino_ruby",
    "arima_kana", "hoshino_ai_(oshi_no_ko)", "gotoh_hitori", "ijichi_nijika", "yamada_ryou", "kita_ikuyo",
    "niijima_makoto", "takamaki_ann", "sakura_futaba", "okumura_haru", "satonaka_chie", "amagi_yukiko",
    "kujikawa_rise", "shirogane_naoto", "yuuki_asuna", "kirigaya_suguha", "asada_shino", "tsukino_usagi",
    "serena_(pokemon)", "may_(pokemon)", "dawn_(pokemon)", "misty_(pokemon)", "lillie_(pokemon)",
    "gloria_(pokemon)", "marnie_(pokemon)", "juliana_(pokemon)", "nemona_(pokemon)", "rosa_(pokemon)",
    "hilda_(pokemon)", "selene_(pokemon)", "hilbert_(pokemon)", "iono_(pokemon)", "penny_(pokemon)",
    "kafuu_chino", "hoto_cocoa", "nakano_ichika", "nakano_nino", "nakano_miku", "nakano_yotsuba", "nakano_itsuki",
    "shinomiya_kaguya", "fujiwara_chika", "aisaka_taiga", "kousaka_kirino", "gokou_ruri", "yukinoshita_yukino",
    "yuigahama_yui", "isshiki_iroha", "makima_(chainsaw_man)_child", "kobeni_higashiyama_child",
    "nishikigi_chisato", "inoue_takina", "uraraka_ochako", "asui_tsuyu", "yaoyorozu_momo", "jirou_kyouka",
    "toga_himiko", "ryuuko_matoi", "kiryuuin_satsuki", "mankanshoku_mako", "hyuuga_hinata_(young)",
    "haruno_sakura_(young)", "nami_(one_piece)_(young)", "yor_briar_(young)",
    "nagatoro_hayase", "ruby_rose", "weiss_schnee", "yang_xiao_long", "blake_belladonna", "rem_(re:zero)",
    "ram_(re:zero)", "fujimaru_ritsuka_(male)", "fujimaru_ritsuka_(female)", "mash_kyrielight", "princess_zelda",
    "link", "purah", "mona_(genshin_impact)", "hu_tao_(genshin_impact)", "kiana_kaslana", "raiden_mei", "bronya_zaychik", "theresa_apocalypse", "seele_vollerei",
}


def minor_series_or_character(post) -> str | None:
    """Reason a danbooru post (dict) is a student cast or minor character, else None."""
    chars = set((post.get("tag_string_character") or "").split())
    series = set((post.get("tag_string_copyright") or "").split())
    if not chars and not series:
        tags = set((post.get("tag_string") or "").split())
        chars, series = tags, tags
    if hit := chars & MINOR_CHARACTERS:
        return f"character {sorted(hit)[0]}"
    if hit := series & MINOR_SERIES:
        return f"series {sorted(hit)[0]}"
    return None


def school_blocked(tags, rating) -> str | None:
    """School-coded tags are only allowed in general-rated images."""
    if rating in ("g", "general"):
        return None
    hit = {_norm(t) for t in tags} & SCHOOL_CODED
    return f"school {sorted(hit)[0]}" if hit else None
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


# names that contain a blocked word without meaning it (the series tag lands in the caption)
CAPTION_NAMES = re.compile(r"teenage[\s_]+mutant[\s_]+ninja[\s_]+turtles", re.I)


def caption_blocked(text: str) -> str | None:
    m = CAPTION_BLOCK.search(CAPTION_NAMES.sub("tmnt", text or ""))
    return f"caption word '{m.group(0)}'" if m else None
