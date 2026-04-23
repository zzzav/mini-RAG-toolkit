import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.query_normalize import DEFAULT_STOP_WORDS, normalize_query
from src.retrieval_types import Chunk
from src.simple_search import build_chunks, load_text_files
from src.synonyms import DEFAULT_SYNONYMS, expand_tokens


@dataclass
class VectorIndex:
    vectorizer: TfidfVectorizer
    matrix: np.ndarray  # shape: (n_chunks, dim)
    chunks: list[Chunk]


def build_vector_index_by_chunks(chunks: list[Chunk]) -> VectorIndex:
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(lowercase=True, min_df=1)
    matrix = vectorizer.fit_transform(texts)  # sparse matrix

    return VectorIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def build_vector_index(docs_dir: str, chunk_size: int, overlap: int) -> VectorIndex:
    docs = load_text_files(docs_dir)
    chunks = build_chunks(docs, chunk_size=chunk_size, overlap=overlap)

    vector_index = build_vector_index_by_chunks(chunks)

    return vector_index


# Выполняет поиск по векторному индексу.
def search(
    query: str,
    index: VectorIndex,
    top_k: int,
    use_synonyms: bool = False,
    use_stop_words: bool = True,
    debug: bool = False,
) -> list[tuple[float, Chunk]]:
    stop_words = ()
    if use_stop_words:
        stop_words = DEFAULT_STOP_WORDS

    tokens = normalize_query(query, stop_words=stop_words)
    if use_synonyms:
        tokens = expand_tokens(tokens=tokens, synonyms=DEFAULT_SYNONYMS)

    normalized_query = " ".join(tokens)
    if normalized_query == "":
        return []
    elif debug:
        print(f"query={normalized_query}")

    q_vec = index.vectorizer.transform([normalized_query])
    sims = cosine_similarity(q_vec, index.matrix).ravel()
    # top-k indices
    top_idx = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), index.chunks[int(i)]) for i in top_idx if sims[i] > 0]


# Сохраняет векторный индекс на диск.
def save_index(path: str, index: VectorIndex) -> None:
    Path(path).write_bytes(pickle.dumps(index))


# Загружает векторный индекс с диска.
def load_index(path: str) -> VectorIndex:
    return pickle.loads(Path(path).read_bytes())


# Собирает парсер аргументов для векторного поиска.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vector search (local embeddings via TF-IDF)")
    p.add_argument("--docs", type=str, default=None)
    p.add_argument("--query", type=str, default=None)
    p.add_argument("--top-k", "--top", dest="top_k", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=80)
    p.add_argument("--index-out", "--build-index", dest="index_out_path", default=None)
    p.add_argument("--index-in", "--use-index", dest="index_in_path", default=None)
    p.add_argument("--debug", action="store_true", default=False)
    return p


# Запускает CLI для построения индекса и/или поиска.
def main() -> None:
    args = build_parser().parse_args()

    if args.index_in_path:
        index = load_index(args.index_in_path)
    else:
        if not args.docs:
            raise SystemExit("Нужно указать --docs, если индекс не загружается из файла")
        index = build_vector_index(args.docs, chunk_size=args.chunk_size, overlap=args.overlap)
        if args.index_out_path:
            save_index(args.index_out_path, index)

    if args.query:
        results = search(args.query, index, top_k=args.top_k, debug=args.debug)
        print(f"CHUNKS={len(index.chunks)} RESULTS={len(results)}")
        if results != []:
            for score, ch in results:
                print(f"[{score:.3f}] {ch.source}#{ch.idx}: {ch.text[:120]}...")
        else:
            print("NO_MATCHES")


if __name__ == "__main__":
    main()
