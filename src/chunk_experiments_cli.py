import argparse
import ast
import json

from src.chunk_experiments import DEFAULT_CHUNKING_CONFIG, run_chunk_experiments
from src.utils import check_path, die, write_output


# Собирает парсер аргументов для экспериментов с чанками.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Эксперименты с размерами чанков")
    p.add_argument("--docs", type=str, required=True)
    p.add_argument("--dataset", dest="dataset_path", type=str, required=True)
    p.add_argument("--top-k", "--k", dest="top_k", type=int, default=5)
    p.add_argument(
        "--chunking-config",
        "--chunks-cfg",
        dest="chunking_config_raw",
        type=str,
        default=repr(DEFAULT_CHUNKING_CONFIG),
    )

    return p


# Парсит конфигурацию чанков из строки CLI.
def parse_chunking_config(raw_value: str) -> tuple[tuple[int, int], ...]:
    try:
        parsed = ast.literal_eval(raw_value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            "chunking-config должен быть кортежем вида ((200, 50), (400, 80))"
        ) from exc

    if not isinstance(parsed, tuple):
        raise ValueError("chunking-config должен быть кортежем вида ((200, 50), (400, 80))")

    for item in parsed:
        if not (
            isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, int) for x in item)
        ):
            raise ValueError("chunking-config должен быть кортежем пар (int, int)")

    return parsed


# Запускает CLI для сравнения чанкинг-конфигураций.
def main() -> None:
    args = build_parser().parse_args()

    docs = check_path(args.docs, entity="docs", must_be_dir=True)

    if not args.dataset_path:
        die("не задан путь к файлу с данными")
    dataset = check_path(args.dataset_path, entity="dataset")

    if args.top_k < 1:
        die("top-k должен быть >= 1")

    try:
        chunking_config = parse_chunking_config(args.chunking_config_raw)
    except ValueError as exc:
        die(str(exc))

    report = run_chunk_experiments(docs, dataset, args.top_k, chunking_config)
    write_output(json.dumps(report, ensure_ascii=False, indent=2), None)


if __name__ == "__main__":
    main()
