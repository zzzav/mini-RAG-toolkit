import argparse
import json

import src.bm25_search as bm25_search
import src.eval_retrieval as eval_retrieval
import src.vector_search as v_search
from src.retrieval_types import FUSION_METHODS, OUTPUT_FORMATS, RETRIEVER_TYPES, Filters
from src.utils import check_path, die, write_output


# Собирает парсер аргументов для оценки retrieval.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Оценка retrieval-качества")
    p.add_argument("--index-vector", dest="index_vector_path", type=str)
    p.add_argument("--index-bm25", dest="index_bm25_path", type=str)
    p.add_argument("--dataset", dest="dataset_path", type=str, required=True)
    p.add_argument("--top-k", "--k", dest="top_k", type=int, default=5)
    p.add_argument("--format", type=str, default="text", choices=OUTPUT_FORMATS)
    p.add_argument("--output", "--out", dest="output_path", type=str, default=None)
    p.add_argument("--retriever", type=str, default="vector", choices=RETRIEVER_TYPES)

    p.add_argument("--fusion-method", type=str, default="rrf", choices=FUSION_METHODS)
    p.add_argument("--fusion-top-n", type=int, default=5)

    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-top-n", type=int, default=10)
    p.add_argument("--proximity-window", type=int, default=5)

    p.add_argument("--filter-source", type=str, default=None)
    p.add_argument("--filter-ext", type=str, default=None)
    p.add_argument("--filter-source-contains", type=str, default=None)

    return p


# Запускает CLI оценки retrieval.
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
    if args.retriever == "vector" or args.retriever == "fusion":
        index_vector_path = check_path(args.index_vector_path, entity="index-vector")
    else:
        index_vector_path = None
    if args.retriever == "bm25" or args.retriever == "fusion":
        index_bm25_path = check_path(args.index_bm25_path, entity="index-bm25")
    else:
        index_bm25_path = None

    if args.rerank:
        if args.rerank_top_n < 1:
            die("rerank-top-n должен быть >= 1")
        if args.proximity_window < 1:
            die("proximity-window должен быть >= 1")

    # Собираем фильтры из CLI-аргументов.
    filters = Filters()
    if args.filter_source is not None:
        filters.source_items = str(args.filter_source).split()
    if args.filter_ext is not None:
        filters.ext_items = str(args.filter_ext).split()
    if args.filter_source_contains is not None:
        filters.source_contains_items = str(args.filter_source_contains).split()

    index_bm25 = None
    index_vector = None
    if args.retriever == "bm25" or args.retriever == "fusion":
        index_bm25 = bm25_search.load_bm25(index_bm25_path)
    if args.retriever == "vector" or args.retriever == "fusion":
        index_vector = v_search.load_index(index_vector_path)

    cases = eval_retrieval.load_eval_cases(dataset_path)
    eval_rep = eval_retrieval.evaluate(
        cases,
        args.top_k,
        index_vector=index_vector,
        index_bm25=index_bm25,
        retriever=args.retriever,
        rerank=args.rerank,
        rerank_top_n=args.rerank_top_n,
        proximity_window=args.proximity_window,
        fusion_top_n=args.fusion_top_n if args.retriever == "fusion" else 0,
        fusion_method=args.fusion_method if args.retriever == "fusion" else None,
        filters=filters,
    )

    if args.format == "json":
        write_output(
            json.dumps(
                eval_retrieval.create_json_from_eval_report(eval_rep), ensure_ascii=False, indent=2
            ),
            args.output_path,
        )
    else:
        write_output(
            f"recall@{eval_rep.k}={eval_rep.recall_mean}\nmrr@{eval_rep.k}={eval_rep.mrr_mean}\nn={eval_rep.n}",
            args.output_path,
        )


if __name__ == "__main__":
    main()
