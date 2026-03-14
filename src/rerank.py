from src.query_normalize import DEFAULT_STOP_WORDS, normalize_query
from src.vector_search import Chunk


def phrase_bonus(query_tokens: list[str], text: str) -> float:
    if not query_tokens:
        return 0.0

    return 2.0 if " ".join(query_tokens) in text else 0.0


def token_overlap_score(query_tokens: list[str], text_tokens: list[str]) -> float:
    q_tokens = set(query_tokens)
    q_tokens_length = len(q_tokens)

    if q_tokens_length == 0:
        return 0.0

    t_count: float = 0.0
    for t in q_tokens:
        t_count += 1 if t in text_tokens else 0

    return t_count / q_tokens_length


def proximity_bonus(query_tokens: list[str], text_tokens: list[str], window: int = 5) -> float:

    pairs_count = len(query_tokens) - 1

    if pairs_count < 1:
        return 0.0

    proximity_pairs_count = 0
    i = 0
    while i < pairs_count:
        w_1 = query_tokens[i]
        w_2 = query_tokens[i + 1]

        w_1_poses = [j for j, w in enumerate(text_tokens) if w_1 == w]
        w_2_poses = [j for j, w in enumerate(text_tokens) if w_2 == w]

        window_found = False
        for p_1 in w_1_poses:
            for p_2 in w_2_poses:
                if (p_1 < p_2) and ((p_2 - p_1) <= window):
                    proximity_pairs_count += 1
                    window_found = True
                    break

            if window_found:
                break

        i += 1

    return proximity_pairs_count / pairs_count


def rerank_hits(
    query: str,
    hits: list[tuple[float, Chunk]],
    *,
    top_k: int = 5,
    use_stop_words: bool = True,
    proximity_window: int = 5
) -> list[tuple[float, Chunk]]:

    if top_k <= 0:
        return []

    stop_words = None
    if use_stop_words:
        stop_words = DEFAULT_STOP_WORDS

    q_tokens = normalize_query(query, stop_words=stop_words)
    new_hits: list[tuple[float, Chunk]] = []

    for h in hits:
        text_tokens = normalize_query(h[1].text, stop_words=stop_words)

        base_score = h[0]
        overlap = token_overlap_score(q_tokens, text_tokens)
        phrase = phrase_bonus(q_tokens, " ".join(text_tokens))
        proximity = proximity_bonus(q_tokens, text_tokens, proximity_window)
        score = base_score + overlap + phrase + proximity

        new_hits.append((score, h[1]))

    new_hits.sort(key=lambda x: (-x[0], x[1].source, x[1].idx))

    return new_hits[:top_k]
