import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

import src.eval_retrieval as eval_retrieval
import src.vector_search as v_search

g_formats_arr = ["text", "json"]


def run_regression(
    *,
    index_path: str,
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
) -> int:

    index = v_search.load_index(index_path)
    cases = eval_retrieval.load_eval_cases(dataset_path)
    report = eval_retrieval.evaluate(
        index,
        cases,
        k,
        retriever=retriever,
        rerank=rerank,
        rerank_top_n=rerank_top_n,
        proximity_window=proximity_window,
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


def write_output(text: str, out_path: str | None) -> None:
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Eval CLI")
    p.add_argument("--index", dest="index_in", type=str, required=True)
    p.add_argument("--dataset", dest="dataset_path", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--min-recall", type=float, required=True)
    p.add_argument("--min-mrr", type=float, required=True)
    p.add_argument("--format", type=str, default="text", choices=g_formats_arr)
    p.add_argument("--out", type=str, default=None)

    p.add_argument("--retriever", type=str, default="vector", choices=["vector", "bm25"])

    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-top-n", type=int, default=10)
    p.add_argument("--proximity-window", type=int, default=5)

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
    if not args.index_in:
        die("не задан путь к index")
    if not args.dataset_path:
        die("не задан путь к файлу с данными")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        die(f"dataset не найден: {args.dataset_path}")
    if not dataset_path.is_file():
        die(f"dataset должен быть файлом: {args.dataset_path}")

    index_path = Path(args.index_in)
    if not index_path.exists():
        die(f"index не найден: {args.index_in}")
    if not index_path.is_file():
        die(f"index должен быть файлом: {args.index_in}")

    if not args.retriever:
        die("не задан тип поиска")

    if args.rerank:
        if not args.rerank_top_n or args.rerank_top_n < 1:
            die("включен режим реранкинга, но не задан rerank-top-n")
        if not args.proximity_window or args.proximity_window < 1:
            die("включен режим реранкинга, но не задан proximity-window")

    if (
        run_regression(
            index_path=args.index_in,
            dataset_path=dataset_path,
            k=args.k,
            min_recall=args.min_recall,
            min_mrr=args.min_mrr,
            retriever=args.retriever,
            rerank=args.rerank,
            rerank_top_n=args.rerank_top_n,
            proximity_window=args.proximity_window,
            out_format=args.format,
            out=args.out,
        )
        == 2
    ):
        die("recall или mrr ниже порогового")


if __name__ == "__main__":
    main()
