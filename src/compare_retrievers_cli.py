import argparse
import json

from src.compare_retrievers import compare_retrievers
from src.retrieval_types import FUSION_METHODS, OUTPUT_FORMATS
from src.utils import check_path, die, write_output


# Преобразует отчёт сравнения в человекочитаемый текст.
def create_text_from_compare_report(report: dict[str, dict[str, float | int]]) -> str:
    lines: list[str] = []
    for i, (key, value) in enumerate(report.items()):
        if i > 0:
            lines.append("")
        lines.append(f"{key}:")
        lines.append(f"recall@{value['k']}={value['recall_mean']}")
        lines.append(f"mrr@{value['k']}={value['mrr_mean']}")
        lines.append(f"n={value['n']}")

    return "\n".join(lines)


# Собирает парсер аргументов для сравнения ретриверов.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Сравнение retriever-ов")
    p.add_argument("--index-vector", dest="index_vector_path", type=str)
    p.add_argument("--index-bm25", dest="index_bm25_path", type=str)
    p.add_argument("--dataset", dest="dataset_path", type=str, required=True)
    p.add_argument("--top-k", "--k", dest="top_k", type=int, default=5)
    p.add_argument("--format", type=str, default="text", choices=OUTPUT_FORMATS)
    p.add_argument("--output", "--out", dest="output_path", type=str, default=None)

    p.add_argument("--fusion-method", type=str, default="rrf", choices=FUSION_METHODS)
    p.add_argument("--fusion-top-n", type=int, default=5)

    p.add_argument("--rerank-top-n", type=int, default=10)
    p.add_argument("--proximity-window", type=int, default=5)
    return p


# Запускает CLI сравнения ретриверов.
def main() -> None:
    args = build_parser().parse_args()

    if args.top_k < 1:
        die("top-k должен быть >= 1")
    if not args.index_vector_path:
        die("не задан путь к index-vector")
    if not args.index_bm25_path:
        die("не задан путь к index-bm25")
    if not args.dataset_path:
        die("не задан путь к файлу с данными")

    check_path(args.dataset_path, entity="dataset")

    if args.fusion_top_n < 1:
        die("fusion-top-n должен быть >= 1")
    check_path(args.index_vector_path, entity="index-vector")
    check_path(args.index_bm25_path, entity="index-bm25")

    if args.rerank_top_n < 1:
        die("rerank-top-n должен быть >= 1")
    if args.proximity_window < 1:
        die("proximity-window должен быть >= 1")

    report = compare_retrievers(
        args.index_vector_path,
        args.index_bm25_path,
        args.dataset_path,
        args.top_k,
        rerank_top_n=args.rerank_top_n,
        proximity_window=args.proximity_window,
        fusion_method=args.fusion_method,
        fusion_top_n=args.fusion_top_n,
    )

    if args.format == "json":
        write_output(
            json.dumps(report, ensure_ascii=False, indent=2),
            args.output_path,
        )
    else:
        write_output(create_text_from_compare_report(report), args.output_path)


if __name__ == "__main__":
    main()
