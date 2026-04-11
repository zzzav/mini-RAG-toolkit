from src.vector_search import Chunk

FUSION_METHODS = {"rrf", "weighted"}


def normalize_hit_id(source: str, idx: int) -> tuple[str, int]:
    return source, idx


def merge_hits_union(
    hits_by_retrievers: list[list[tuple[float, Chunk]]],
) -> list[tuple[float, Chunk]]:
    final_hits: list[tuple[float, Chunk]] = []
    final_hits_keys: set[tuple[str, int]] = set()

    for hits in hits_by_retrievers:
        for hit in hits:
            key = normalize_hit_id(hit[1].source, hit[1].idx)

            if key in final_hits_keys:
                continue

            final_hits_keys.add(key)
            final_hits.append(hit)

    return final_hits


def build_score_map(hits: list[tuple[float, Chunk]]) -> dict[tuple[str, int], tuple[float, Chunk]]:
    out: dict[tuple[str, int], tuple[float, Chunk]] = {}
    for score, chunk in hits:
        key = normalize_hit_id(chunk.source, chunk.idx)
        out[key] = (score, chunk)
    return out


def build_rank_map(hits: list[tuple[float, Chunk]]) -> dict[tuple[str, int], int]:
    rank_map: dict[tuple[str, int], int] = {}

    for i, (_, chunk) in enumerate(hits, start=1):
        key = normalize_hit_id(chunk.source, chunk.idx)
        rank_map[key] = i

    return rank_map


def weighted_score_fusion(
    vector_hits: list[tuple[float, Chunk]],
    bm25_hits: list[tuple[float, Chunk]],
    tfidf_hits: list[tuple[float, Chunk]] | None = None,
    w_vector: float = 1.0,
    w_bm25: float = 1.0,
    w_tfidf: float = 1.0,
) -> list[tuple[float, Chunk]]:

    if tfidf_hits is None:
        tfidf_hits = []

    vector_map = build_score_map(vector_hits)
    bm25_map = build_score_map(bm25_hits)
    tfidf_map = build_score_map(tfidf_hits)

    all_keys = set(vector_map) | set(bm25_map) | set(tfidf_map)

    fused_hits: list[tuple[float, Chunk]] = []

    for key in all_keys:
        vector_score = vector_map.get(key, (0.0, None))[0]
        bm25_score = bm25_map.get(key, (0.0, None))[0]
        tfidf_score = tfidf_map.get(key, (0.0, None))[0]

        chunk = (
            vector_map.get(key, (0.0, None))[1]
            or bm25_map.get(key, (0.0, None))[1]
            or tfidf_map.get(key, (0.0, None))[1]
        )

        final_score = w_vector * vector_score + w_bm25 * bm25_score + w_tfidf * tfidf_score

        fused_hits.append((final_score, chunk))

    fused_hits.sort(key=lambda x: (-x[0], x[1].source, x[1].idx))
    return fused_hits


def rrf_fusion(
    vector_hits: list[tuple[float, Chunk]],
    bm25_hits: list[tuple[float, Chunk]],
    tfidf_hits: list[tuple[float, Chunk]] | None = None,
    w_vector: float = 1.0,
    w_bm25: float = 1.0,
    w_tfidf: float = 1.0,
    k: int = 60,
) -> list[tuple[float, Chunk]]:

    if tfidf_hits is None:
        tfidf_hits = []

    vector_map = build_score_map(vector_hits)
    bm25_map = build_score_map(bm25_hits)
    tfidf_map = build_score_map(tfidf_hits)

    vector_rank_map = build_rank_map(vector_hits)
    bm25_rank_map = build_rank_map(bm25_hits)
    tfidf_rank_map = build_rank_map(tfidf_hits)

    all_keys = set(vector_map) | set(bm25_map) | set(tfidf_map)

    fused_hits: list[tuple[float, Chunk]] = []

    for key in all_keys:
        vector_rank = vector_rank_map.get(key)
        bm25_rank = bm25_rank_map.get(key)
        tfidf_rank = tfidf_rank_map.get(key)

        rrf_score_vector = w_vector / (k + vector_rank) if vector_rank is not None else 0.0
        rrf_score_bm25 = w_bm25 / (k + bm25_rank) if bm25_rank is not None else 0.0
        rrf_score_tfidf = w_tfidf / (k + tfidf_rank) if tfidf_rank is not None else 0.0

        rrf_score = rrf_score_vector + rrf_score_bm25 + rrf_score_tfidf

        chunk = (
            vector_map.get(key, (0.0, None))[1]
            or bm25_map.get(key, (0.0, None))[1]
            or tfidf_map.get(key, (0.0, None))[1]
        )

        fused_hits.append((rrf_score, chunk))

    fused_hits.sort(key=lambda x: (-x[0], x[1].source, x[1].idx))
    return fused_hits
