import src.bm25_search as bm25_search
import src.eval_retrieval as eval_retrieval
import src.vector_search as v_search


def compare_retrievers(
    index_vector_path: str,
    index_bm25_path: str,
    dataset_path: str,
    k: int,
    *,
    rerank_top_n: int = 10,
    proximity_window: int = 5,
    fusion_top_n: int = 5,
    fusion_method: str = "rrf",
) -> dict[str, dict[str, float | int]]:

    report: dict[str, dict[str, float | int]] = {}

    index_bm25 = bm25_search.load_bm25(index_bm25_path)
    index_vector = v_search.load_index(index_vector_path)

    cases = eval_retrieval.load_eval_cases(dataset_path)

    # vector
    vector_results = eval_retrieval.evaluate(
        cases,
        k,
        index_vector=index_vector,
        retriever="vector",
    )
    vector_rerank_results = eval_retrieval.evaluate(
        cases,
        k,
        index_vector=index_vector,
        retriever="vector",
        rerank=True,
        rerank_top_n=rerank_top_n,
        proximity_window=proximity_window,
    )

    # bm25
    bm25_results = eval_retrieval.evaluate(
        cases,
        k,
        index_bm25=index_bm25,
        retriever="bm25",
    )
    bm25_rerank_results = eval_retrieval.evaluate(
        cases,
        k,
        index_bm25=index_bm25,
        retriever="bm25",
        rerank=True,
        rerank_top_n=rerank_top_n,
        proximity_window=proximity_window,
    )

    # fusion
    fusion_results = eval_retrieval.evaluate(
        cases,
        k,
        index_bm25=index_bm25,
        index_vector=index_vector,
        retriever="fusion",
        fusion_method=fusion_method,
        fusion_top_n=fusion_top_n,
    )

    def report_to_summary(res: eval_retrieval.EvalReport) -> dict[str, float | int]:
        return {
            "k": res.k,
            "recall_mean": res.recall_mean,
            "mrr_mean": res.mrr_mean,
            "n": res.n,
        }

    report["vector"] = report_to_summary(vector_results)
    report["bm25"] = report_to_summary(bm25_results)
    report["rerank_vector"] = report_to_summary(vector_rerank_results)
    report["rerank_bm25"] = report_to_summary(bm25_rerank_results)
    report["fusion"] = report_to_summary(fusion_results)

    return report
