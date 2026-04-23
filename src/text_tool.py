import argparse
from pathlib import Path

from src.utils import normalize_text


# Читает текст из строки или из файла.
def read_text(text: str | None, in_path: str | None) -> str:
    if text:
        return text
    if in_path:
        return Path(in_path).read_text(encoding="utf-8", errors="replace")
    raise ValueError("Нужно указать --text или --in")


# Сохраняет текст в файл или печатает его в консоль.
def write_text(out_path: str | None, content: str) -> None:
    if out_path:
        Path(out_path).write_text(content, encoding="utf-8")
    else:
        print(content)


# Считает простую статистику по исходному и нормализованному тексту.
def calc_stats(raw: str, normalized: str) -> dict[str, int]:
    chars = len(normalized)

    # выдергиваем слова из полученного контента
    stripped = normalized.strip()
    words = 0 if (stripped == "") else len(stripped.split())

    # подсчет строк
    lines = 0 if (raw == "") else len(raw.splitlines())

    # подсчет не пустых строк
    non_empty_lines = 0
    for line in raw.splitlines():
        if line.strip() != "":
            non_empty_lines += 1

    return {"chars": chars, "words": words, "lines": lines, "non_empty_lines": non_empty_lines}


# Собирает парсер аргументов для CLI по текстовой нормализации.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Нормализация текста и статистика")
    p.add_argument("--text", type=str, default=None, help="Текст прямо в командной строке")
    p.add_argument(
        "--input", "--in", dest="input_path", type=str, default=None, help="Путь к входному файлу"
    )
    p.add_argument(
        "--output",
        "--out",
        dest="output_path",
        type=str,
        default=None,
        help="Путь к выходному файлу",
    )
    p.add_argument("--lower", action="store_true", help="Привести к нижнему регистру")
    p.add_argument("--upper", action="store_true", help="Привести к верхнему регистру")
    p.add_argument(
        "--stats-only",
        "--stats_only",
        dest="stats_only",
        action="store_true",
        help="Только вывести статистику",
    )
    return p


# Применяет верхний или нижний регистр к тексту.
def upper_lower_text(text: str, lower: bool, upper: bool) -> str:
    if upper:
        return text.upper()
    elif lower:
        return text.lower()
    return text


# Запускает CLI нормализации текста.
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        raw = read_text(args.text, args.input_path)
    except ValueError as e:
        print(f"ERROR: {e}")
        raise SystemExit(2) from None

    normalized = normalize_text(raw)

    normalized = upper_lower_text(normalized, args.lower, args.upper)

    stats = calc_stats(raw, normalized)

    if not args.stats_only:
        write_text(args.output_path, normalized)

    print(
        "STATS: "
        f"chars={stats['chars']} "
        f"words={stats['words']} "
        f"lines={stats['lines']} "
        f"non_empty_lines={stats['non_empty_lines']}"
    )


if __name__ == "__main__":
    main()
