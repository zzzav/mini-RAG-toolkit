import argparse
import ast
import json
import sys
from pathlib import Path
from typing import NoReturn

from src.chunk_experiments import DEFAULT_CHUNKING_CONFIG, run_chunk_experiments


def parse_chunks_cfg(value: str) -> tuple[tuple[int, int], ...]:
    try:
        cfg = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "chunks-cfg must look like ((200, 40), (400, 80))"
        ) from exc

    if not isinstance(cfg, tuple):
        raise argparse.ArgumentTypeError("chunks-cfg must be a tuple")

    for item in cfg:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise argparse.ArgumentTypeError(
                "chunks-cfg must be a tuple of 2-element tuples"
            )
        if not all(isinstance(x, int) for x in item):
            raise argparse.ArgumentTypeError("chunks-cfg values must be ints")

    return cfg


def write_output(text: str, out_path: str | None) -> None:
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chunk_experiments_CLI")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--docs", type=str, required=True)
    p.add_argument("--dataset", dest="dataset", type=str, required=True)
    p.add_argument(
        "--chunks-cfg",
        type=parse_chunks_cfg,
        default=DEFAULT_CHUNKING_CONFIG,
    )

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

    report = run_chunk_experiments(args.docs, args.dataset, args.k, args.chunks_cfg)
    write_output(json.dumps(report, ensure_ascii=False, indent=2), None)


if __name__ == "__main__":
    main()
