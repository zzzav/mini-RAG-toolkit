import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

import src.bm25_search as bm25_search
import src.eval_retrieval as eval_retrieval
import src.fusion_search as fusion_search
import src.rag_answer as rag_answer
import src.vector_search as v_search
from src.retrieval_types import Filters

g_formats_arr = {"text", "json"}


def write_output(text: str, out_path: str | None) -> None:
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Eval CLI")
    p.add_argument("--index-vector", dest="index_vector", type=str)
    p.add_argument("--index-bm25", dest="index_bm25", type=str)
    p.add_argument("--dataset", dest="dataset_path", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--format", type=str, default="text", choices=g_formats_arr)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--retriever", type=str, default="vector", choices=rag_answer.RETRIEVER_TYPES)

    p.add_argument("--fusion-method", type=str, default="rrf", choices=fusion_search.FUSION_METHODS)
    p.add_argument("--fusion-top-n", type=int, default=5)

    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-top-n", type=int, default=10)
    p.add_argument("--proximity-window", type=int, default=5)

    p.add_argument("--filter-source", type=str, default=None)
    p.add_argument("--filter-ext", type=str, default=None)
    p.add_argument("--filter-source-contains", type=str, default=None)

    return p


def die(msg: str, code: int = 2) -> NoReturn:
    print("Ошибка: " + msg, file=sys.stderr)
    raise SystemExit(code) from None


def main() -> None:
    args = build_parser().parse_args()

    if args.format not in g_formats_arr:
        die(f"format должен принимать одно из значений: {g_formats_arr}")
    if not isinstance(args.k, int):
        die("k должен целочисленным")
    if args.k < 1:
        die("k должен быть >= 1")
    if (args.retriever == "vector" or args.retriever == "fusion") and not args.index_vector:
        die("не задан путь к index-vector")
    if (args.retriever == "bm25" or args.retriever == "fusion") and not args.index_bm25:
        die("не задан путь к index-bm25")
    if not args.dataset_path:
        die("не задан путь к файлу с данными")
    if not args.retriever:
        die("не задан тип поиска")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        die(f"dataset не найден: {args.dataset_path}")
    if not dataset_path.is_file():
        die(f"dataset должен быть файлом: {args.dataset_path}")

    if args.retriever == "fusion":
        if not args.fusion_method:
            die("включен режим смешивания, но не задан тип fusion-method")
        if not args.fusion_top_n or args.fusion_top_n < 1:
            die("включен режим смешивания, но не задан fusion-top-n")

    def check_index_path(path: str, retriever: str):
        index_path = Path(path)
        if not index_path.exists():
            die(f"index-{retriever} не найден: {path}")

    if args.retriever == "vector" or args.retriever == "fusion":
        check_index_path(args.index_vector, "vector")
    if args.retriever == "bm25" or args.retriever == "fusion":
        check_index_path(args.index_bm25, "bm25")

    if args.rerank:
        if not args.rerank_top_n or args.rerank_top_n < 1:
            die("включен режим реранкинга, но не задан rerank-top-n")
        if not args.proximity_window or args.proximity_window < 1:
            die("включен режим реранкинга, но не задан proximity-window")

    # создание фильтра
    filters = Filters()
    if args.filter_source is not None:
        filters.source_items = str(args.filter_source).split(" ")
    if args.filter_ext is not None:
        filters.ext_items = str(args.filter_ext).split(" ")
    if args.filter_source_contains is not None:
        filters.source_contains_items = str(args.filter_source_contains).split(" ")

    index_bm25 = None
    index_vector = None
    if args.retriever == "bm25" or args.retriever == "fusion":
        index_bm25 = bm25_search.load_bm25(args.index_bm25)
    if args.retriever == "vector" or args.retriever == "fusion":
        index_vector = v_search.load_index(args.index_vector)

    cases = eval_retrieval.load_eval_cases(dataset_path)
    eval_rep = eval_retrieval.evaluate(
        cases,
        args.k,
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
            args.out,
        )
    # text
    else:
        write_output(
            f"recall@{eval_rep.k}={eval_rep.recall_mean}\nmrr@{eval_rep.k}={eval_rep.mrr_mean}\nn={eval_rep.n}",
            args.out,
        )


if __name__ == "__main__":
    main()
