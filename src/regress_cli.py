import argparse
import json
import src.bm25_search as bm25_search
import src.eval_retrieval as eval_retrieval
import src.vector_search as v_search
from src.retrieval_types import FUSION_METHODS, OUTPUT_FORMATS, RETRIEVER_TYPES
from src.utils import check_path, die, write_output


def run_regression(
    *,
    index_vector_path: str | None = None,
    index_bm25_path: str | None = None,
    dataset_path: str,
    k: int,
    min_recall: float,
    min_mrr: float,
    for_test: bool = False,
    retriever: str = "vector",
    rerank: bool = False,
    rerank_top_n: int = 10,
    proximity_window: int = 5,
    out_format: str = "text",
    out: str | None = None,
    fusion_method: str | None = None,
    fusion_top_n: int = 5,
) -> int:
    # Запускает проверку регрессии по порогам качества.
    index_vector = None
    index_bm25 = None
    if retriever == "vector" or retriever == "fusion":
        index_vector = v_search.load_index(index_vector_path)
    if retriever == "bm25" or retriever == "fusion":
        index_bm25 = bm25_search.load_bm25(index_bm25_path)

    cases = eval_retrieval.load_eval_cases(dataset_path)

    report = eval_retrieval.evaluate(
        cases,
        k,
        index_vector=index_vector,
        index_bm25=index_bm25,
        retriever=retriever,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
        proximity_window=proximity_window,
        fusion_method=fusion_method,
        fusion_top_n=fusion_top_n,
    )

    if report.recall_mean < min_recall or report.mrr_mean < min_mrr:
        return 2

    if for_test:
        return 0

    if out_format == "json" and out:
        write_output(
            json.dumps(
                eval_retrieval.create_json_from_eval_report(report), ensure_ascii=False, indent=2
            ),
            out,
        )
    # text
    else:
        write_output(
            f"recall@{report.k}={report.recall_mean}\nmrr@{report.k}={report.mrr_mean}\nn={report.n}",
            out,
        )

    return 0


# Собирает парсер аргументов для регрессионной проверки.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Регрессионная проверка retrieval")
    p.add_argument("--index-vector", dest="index_vector_path", type=str)
    p.add_argument("--index-bm25", dest="index_bm25_path", type=str)
    p.add_argument("--dataset", dest="dataset_path", type=str, required=True)
    p.add_argument("--top-k", "--k", dest="top_k", type=int, default=5)
    p.add_argument("--min-recall", type=float, required=True)
    p.add_argument("--min-mrr", type=float, required=True)
    p.add_argument("--format", type=str, default="text", choices=OUTPUT_FORMATS)
    p.add_argument("--output", "--out", dest="output_path", type=str, default=None)

    p.add_argument("--retriever", type=str, default="vector", choices=RETRIEVER_TYPES)

    p.add_argument("--fusion-method", type=str, default="rrf", choices=FUSION_METHODS)
    p.add_argument("--fusion-top-n", type=int, default=5)

    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-top-n", type=int, default=10)
    p.add_argument("--proximity-window", type=int, default=5)

    return p


# Запускает CLI регрессионной проверки.
def main() -> None:
    args = build_parser().parse_args()

    if args.top_k < 1:
        die("top-k должен быть >= 1")
    if (args.retriever == "vector" or args.retriever == "fusion") and not args.index_vector_path:
        die("не задан путь к index-vector")
    if (args.retriever == "bm25" or args.retriever == "fusion") and not args.index_bm25_path:
        die("не задан путь к index-bm25")
    if not args.dataset_path:
        die("не задан путь к файлу с данными")

    dataset_path = check_path(args.dataset_path, entity="dataset")

    if args.retriever == "fusion":
        if args.fusion_top_n < 1:
            die("fusion-top-n должен быть >= 1")

    index_vector_path = None
    if args.retriever == "vector" or args.retriever == "fusion":
        index_vector_path = check_path(args.index_vector_path, entity="index-vector")
    index_bm25_path = None
    if args.retriever == "bm25" or args.retriever == "fusion":
        index_bm25_path = check_path(args.index_bm25_path, entity="index-bm25")

    if args.rerank:
        if args.rerank_top_n < 1:
            die("rerank-top-n должен быть >= 1")
        if args.proximity_window < 1:
            die("proximity-window должен быть >= 1")

    if (
        run_regression(
            index_vector_path=index_vector_path,
            index_bm25_path=index_bm25_path,
            dataset_path=dataset_path,
            k=args.top_k,
            min_recall=args.min_recall,
            min_mrr=args.min_mrr,
            retriever=args.retriever,
            rerank=args.rerank,
            rerank_top_n=args.rerank_top_n,
            proximity_window=args.proximity_window,
            out_format=args.format,
            out=args.output_path,
            fusion_method=args.fusion_method if args.fusion_method else None,
            fusion_top_n=args.fusion_top_n if args.fusion_top_n else None,
        )
        == 2
    ):
        die("recall или mrr ниже порогового")


if __name__ == "__main__":
    main()
