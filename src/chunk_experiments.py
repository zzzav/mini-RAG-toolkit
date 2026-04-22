from pathlib import Path

import src.bm25_search as bm25_search
from src.eval_retrieval import evaluate, load_eval_cases

DEFAULT_CHUNKING_CONFIG = ((200, 40), (400, 80), (800, 120))


def run_chunk_experiments(
    docs_path: Path,
    dataset_path: str,
    k: int = 5,
    chunking_cfg: tuple[tuple[int, int]] = DEFAULT_CHUNKING_CONFIG,
) -> list[dict]:

    fin_results: list[dict] = []
    cases = load_eval_cases(dataset_path)

    for chunk_size, overlap in chunking_cfg:
        index = bm25_search.build_bm25_index(docs_path, chunk_size, overlap)
        report = evaluate(cases, k, index_bm25=index, retriever="bm25")

        fin_results.append(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "k": k,
                "retriever": "bm25",
                "recall_mean": report.recall_mean,
                "mrr_mean": report.mrr_mean,
                "n": report.n,
            }
        )

    return fin_results
