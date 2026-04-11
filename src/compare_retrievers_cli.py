import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

import src.fusion_search as fusion_search
from src.compare_retrievers import compare_retrievers

g_formats_arr = {"text", "json"}


def write_output(text: str, out_path: str | None) -> None:
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def create_text_from_compare_report(report: dict[str, dict[str, float | int]]) -> str:
    lines: list[str] = []
    for key, value in report.items():
        lines.append(f"{key}:")
        lines.append(f"recall@{value['k']}={value['recall_mean']}")
        lines.append(f"mrr@{value['k']}={value['mrr_mean']}")
        lines.append(f"n={value['n']}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Eval CLI")
    p.add_argument("--index-vector", dest="index_vector", type=str)
    p.add_argument("--index-bm25", dest="index_bm25", type=str)
    p.add_argument("--dataset", dest="dataset_path", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--format", type=str, default="text", choices=g_formats_arr)
    p.add_argument("--out", type=str, default=None)

    p.add_argument("--fusion-method", type=str, default="rrf", choices=fusion_search.FUSION_METHODS)
    p.add_argument("--fusion-top-n", type=int, default=5)

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
    if not args.index_vector:
        die("не задан путь к index-vector")
    if not args.index_bm25:
        die("не задан путь к index-bm25")
    if not args.dataset_path:
        die("не задан путь к файлу с данными")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        die(f"dataset не найден: {args.dataset_path}")
    if not dataset_path.is_file():
        die(f"dataset должен быть файлом: {args.dataset_path}")

    if not args.fusion_method:
        die("не задан тип fusion-method")
    if not args.fusion_top_n or args.fusion_top_n < 1:
        die("не задан fusion-top-n")

    def check_index_path(path: str, retriever: str):
        index_path = Path(path)
        if not index_path.exists():
            die(f"index-{retriever} не найден: {path}")

    check_index_path(args.index_vector, "vector")
    check_index_path(args.index_bm25, "bm25")

    if not args.rerank_top_n or args.rerank_top_n < 1:
        die("не задан rerank-top-n")
    if not args.proximity_window or args.proximity_window < 1:
        die("не задан proximity-window")

    report = compare_retrievers(
        args.index_vector,
        args.index_bm25,
        args.dataset_path,
        args.k,
        rerank_top_n=args.rerank_top_n,
        proximity_window=args.proximity_window,
        fusion_method=args.fusion_method,
        fusion_top_n=args.fusion_top_n,
    )

    if args.format == "json":
        write_output(
            json.dumps(report, ensure_ascii=False, indent=2),
            args.out,
        )
    # text
    else:
        write_output(create_text_from_compare_report(report), args.out)


if __name__ == "__main__":
    main()
