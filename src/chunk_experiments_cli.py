import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from src.chunk_experiments import DEFAULT_CHUNKING_CONFIG, run_chunk_experiments


def write_output(text: str, out_path: str | None) -> None:
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chunk_experiments_CLI")
    p.add_argument("--docs", type=str, required=True)
    p.add_argument("--dataset", dest="dataset", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--chunks-cfg", type=str, default=repr(DEFAULT_CHUNKING_CONFIG))

    return p


def die(msg: str, code: int = 2) -> NoReturn:
    print("Ошибка: " + msg, file=sys.stderr)
    raise SystemExit(code) from None


def main() -> None:
    args = build_parser().parse_args()

    docs = Path(args.docs)
    if not docs.exists():
        die(f"docs не найден: {docs}")
    if not docs.is_dir():
        die(f"docs должен быть директорией: {docs}")

    if not args.dataset:
        die("не задан путь к файлу с данными")
    dataset = Path(args.dataset)
    if not dataset.exists():
        die(f"dataset не найден: {args.dataset}")
    if not dataset.is_file():
        die(f"dataset должен быть файлом: {args.dataset}")

    if not isinstance(args.k, int):
        die("k должен целочисленным")
    if args.k < 1:
        die("k должен быть >= 1")

    if not isinstance(args.chunks_cfg, tuple):
        die(f"chunks-cfg должен быть кортежем вида ((200, 50), (400, 80)): {args.chunks_cfg}")
    for item in args.chunks_cfg:
        if not (isinstance(item, tuple) and len(item) == 2):
            die(f"chunks-cfg должен быть кортежем вида ((200, 50), (400, 80)): {args.chunks_cfg}")
        if not all(isinstance(x, int) for x in item):
            die(f"chunks-cfg должен быть кортежем со значениями типа int: {args.chunks_cfg}")

    report = run_chunk_experiments(docs, dataset, args.k, args.chunks_cfg)
    write_output(json.dumps(report, ensure_ascii=False, indent=2), None)


if __name__ == "__main__":
    main()
