import math
import pickle
from dataclasses import dataclass
from pathlib import Path

from src.query_normalize import DEFAULT_STOP_WORDS, normalize_query
from src.retrieval_types import Chunk
from src.simple_search import load_text_files
from src.synonyms import DEFAULT_SYNONYMS, expand_tokens
from src.tfidf_search import count_tf
from src.vector_search import build_chunks


@dataclass
class BM25Index:
    chunks: list[Chunk]
    dl: list[int]
    tf: list[dict[str, int]]
    df: dict[str, int]
    avgdl: float
    k1: float = 1.2
    b: float = 0.75
    stop_words: set[str] | None = None


def build_bm25_index(
    doc_dir: str, chunk_size: int, overlap: int, *, use_stop_words: bool = True
) -> BM25Index:

    docs = load_text_files(doc_dir)
    chunks = build_chunks(docs, chunk_size, overlap)
    tf: list[dict[str, int]] = []
    df: dict[str, int] = {}
    dl: list[int] = []
    stop_words = DEFAULT_STOP_WORDS if use_stop_words else None

    for chunk in chunks:
        tokens = normalize_query(chunk.text, stop_words=stop_words)

        tf.append(count_tf(tokens))
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

        dl.append(len(tokens))

    avgdl = sum(dl) / len(dl)

    return BM25Index(chunks=chunks, tf=tf, df=df, avgdl=avgdl, dl=dl, stop_words=stop_words)


def bm25_search(
    query: str, index: BM25Index, top_k: int = 5, *, use_synonyms: bool = False
) -> list[tuple[float, Chunk]]:

    q_tokens = normalize_query(query, stop_words=index.stop_words)
    if use_synonyms:
        q_tokens = expand_tokens(tokens=q_tokens, synonyms=DEFAULT_SYNONYMS)

    scored_chunks: list[tuple[float, Chunk]] = []

    if not q_tokens:
        return []

    N = len(index.chunks)

    for i, chunk in enumerate(index.chunks):
        score = 0.0
        dl = index.dl[i]

        for t in q_tokens:
            tf = index.tf[i].get(t, 0)
            if tf > 0:
                df = index.df.get(t, 0)
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                score += (
                    idf
                    * (tf * (index.k1 + 1))
                    / (tf + index.k1 * (1 - index.b + index.b * dl / index.avgdl))
                )

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: (-x[0], x[1].source, x[1].idx))

    return scored_chunks[:top_k]


def save_bm25(path: str, index: BM25Index) -> None:
    Path(path).write_bytes(pickle.dumps(index))


def load_bm25(path: str) -> BM25Index:
    return pickle.loads(Path(path).read_bytes())
