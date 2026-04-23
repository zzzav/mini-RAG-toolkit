import src.bm25_search as bm25_search
import src.fusion_search as fusion_search
import src.rag_answer as rag_answer
import src.vector_search as v_search
from src.eval_answer import AnswerEvalReport, evaluate_answers, load_answer_eval_cases
from src.rerank import rerank_hits


def compare_pipelines(
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

    # пока только один метод поиска
    llm: str = "extract"

    index_bm25 = bm25_search.load_bm25(index_bm25_path)
    index_vector = v_search.load_index(index_vector_path)

    cases = load_answer_eval_cases(dataset_path)

    rag_cfg = rag_answer.RAGConfig(top_k=k)

    # локальные функции для получения ответа на запрос
    def get_answer_from_rag_res_vector(query: str) -> str:
        search_res = v_search.search(query, index_vector, top_k=k)
        rag_res = rag_answer.rag_answer(query, search_res, rag_cfg, llm=llm)
        return rag_res.answer

    def get_answer_from_rag_res_vector_rerank(query: str) -> str:
        search_res = v_search.search(query, index_vector, top_k=rerank_top_n)
        search_res = rerank_hits(
            query,
            search_res,
            top_k=k,
            proximity_window=proximity_window,
        )
        rag_res = rag_answer.rag_answer(query, search_res, rag_cfg, llm=llm)
        return rag_res.answer

    def get_answer_from_rag_res_bm25(query: str) -> str:
        search_res = bm25_search.bm25_search(query, index_bm25, top_k=k)
        rag_res = rag_answer.rag_answer(query, search_res, rag_cfg, llm=llm)
        return rag_res.answer

    def get_answer_from_rag_res_bm25_rerank(query: str) -> str:
        search_res = bm25_search.bm25_search(query, index_bm25, top_k=rerank_top_n)
        search_res = rerank_hits(
            query,
            search_res,
            top_k=k,
            proximity_window=proximity_window,
        )
        rag_res = rag_answer.rag_answer(query, search_res, rag_cfg, llm=llm)
        return rag_res.answer

    def get_answer_from_rag_res_fusion(query: str) -> str:
        vector_search_res = v_search.search(query, index_vector, top_k=fusion_top_n)
        bm25_search_res = bm25_search.bm25_search(query, index_bm25, top_k=fusion_top_n)
        search_res = None
        if fusion_method == "weighted":
            search_res = fusion_search.weighted_score_fusion(vector_search_res, bm25_search_res, [])
        elif fusion_method == "rrf":
            search_res = fusion_search.rrf_fusion(vector_search_res, bm25_search_res, [])
        else:
            raise ValueError(f"Ошибка: не поддерживаемый метод fusion-поиска: {fusion_method}")

        rag_res = rag_answer.rag_answer(query, search_res, rag_cfg, llm=llm)
        return rag_res.answer

    # vector
    vector_answer_rep = evaluate_answers(cases, get_answer_from_rag_res_vector)
    vector_rerank_answer_rep = evaluate_answers(cases, get_answer_from_rag_res_vector_rerank)

    # bm25
    bm25_answer_rep = evaluate_answers(cases, get_answer_from_rag_res_bm25)
    bm25_rerank_answer_rep = evaluate_answers(cases, get_answer_from_rag_res_bm25_rerank)

    # fusion
    fusion_answer_rep = evaluate_answers(cases, get_answer_from_rag_res_fusion)

    def report_to_summary(res: AnswerEvalReport) -> dict[str, float | int]:
        return {
            "n": res.n,
            "contains_rate": res.contains_rate,
            "no_info_accuracy": res.no_info_accuracy,
        }

    report["vector_extract"] = report_to_summary(vector_answer_rep)
    report["bm25_extract"] = report_to_summary(bm25_answer_rep)
    report["rerank_vector_extract"] = report_to_summary(vector_rerank_answer_rep)
    report["rerank_bm25_extract"] = report_to_summary(bm25_rerank_answer_rep)
    report["fusion_extract"] = report_to_summary(fusion_answer_rep)

    return report
