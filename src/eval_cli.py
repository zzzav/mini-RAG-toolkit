import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

import src.eval_retrieval as eval_retrieval
import src.vector_search as v_search

g_formats_arr = ["text", "json"]


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
    p.add_argument("--format", type=str, default="text", choices=g_formats_arr)
    p.add_argument("--out", type=str, default=None)
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

    index = v_search.load_index(args.index_in)
    cases = eval_retrieval.load_eval_cases(dataset_path)
    eval_rep = eval_retrieval.evaluate(index, cases, args.k)

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
