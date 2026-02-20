import re

# стоп-слова
DEFAULT_STOP_WORDS = {"the", "a", "an", "and", "or", "to", "in", "of"}


def normalize_query(text: str, *, stop_words: set[str] | None = None) -> list[str]:

    t = text.strip()
    if t == "":
        return []

    used_stop_words = set(stop_words) if stop_words else set()

    t_clean = re.sub(r"[,.:;!?()\'\"]", "", t.lower())
    t_words = [w for w in t_clean.split() if w and w not in used_stop_words]

    return t_words
