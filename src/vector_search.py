import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.simple_search import Chunk, build_chunks, load_text_files


@dataclass
class VectorIndex:
    vectorizer: TfidfVectorizer
    matrix: np.ndarray  # shape: (n_chunks, dim)
    chunks: list[Chunk]


def build_vector_index_by_chunks(chunks: list[Chunk]) -> VectorIndex:
    texts = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", min_df=1)
    matrix = vectorizer.fit_transform(texts)  # sparse matrix

    return VectorIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def build_vector_index(docs_dir: str, chunk_size: int, overlap: int) -> VectorIndex:
    docs = load_text_files(docs_dir)
    chunks = build_chunks(docs, chunk_size=chunk_size, overlap=overlap)

    vector_index = build_vector_index_by_chunks(chunks)

    return vector_index


def search(query: str, index: VectorIndex, top_k: int) -> list[tuple[float, Chunk]]:
    q_vec = index.vectorizer.transform([query])
    sims = cosine_similarity(q_vec, index.matrix).ravel()
    # top-k indices
    top_idx = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), index.chunks[int(i)]) for i in top_idx if sims[i] > 0]


def save_index(path: str, index: VectorIndex) -> None:
    Path(path).write_bytes(pickle.dumps(index))


def load_index(path: str) -> VectorIndex:
    return pickle.loads(Path(path).read_bytes())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vector search (local embeddings via TF-IDF)")
    p.add_argument("--docs", type=str, required=True)
    p.add_argument("--query", type=str, default=None)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=80)
    p.add_argument("--build-index", dest="build_index_path", default=None)
    p.add_argument("--use-index", dest="use_index_path", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.use_index_path:
        index = load_index(args.use_index_path)
    else:
        index = build_vector_index(args.docs, chunk_size=args.chunk_size, overlap=args.overlap)
        if args.build_index_path:
            save_index(args.build_index_path, index)

    if args.query:
        results = search(args.query, index, top_k=args.top)
        print(f"CHUNKS={len(index.chunks)} RESULTS={len(results)}")
        for score, ch in results:
            print(f"[{score:.3f}] {ch.source}#{ch.idx}: {ch.text[:120]}...")


if __name__ == "__main__":
    main()
