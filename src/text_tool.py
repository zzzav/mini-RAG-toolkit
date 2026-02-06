import argparse
from pathlib import Path

from src.utils import normalize_text


def read_text(text: str | None, in_path: str | None) -> str:
    if text:
        return text
    if in_path:
        return Path(in_path).read_text(encoding="utf-8")
    raise ValueError("Нужно указать --text или --in")


def write_text(out_path: str | None, content: str) -> None:
    if out_path:
        Path(out_path).write_text(content, encoding="utf-8")
    else:
        print(content)


def calc_stats(raw: str, normalized: str) -> dict[str, int]:
    chars = len(normalized)

    # выдергиваем слова из полученного контента
    stripped = normalized.strip()
    words = 0 if (stripped == "") else len(stripped.split())

    #
    lines = 0 if (raw == "") else len(raw.splitlines())

    return {"chars": chars, "words": words, "lines": lines}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Нормализация текста и статистика")
    p.add_argument("--text", type=str, default=None, help="Текст прямо в командной строке")
    p.add_argument("--in", dest="in_path", type=str, default=None, help="Путь к входному файлу")
    p.add_argument("--out", dest="out_path", type=str, default=None, help="Путь к выходному файлу")
    p.add_argument("--lower", action="store_true", help="Привести к нижнему регистру")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raw = read_text(args.text, args.in_path)
    normalized = normalize_text(raw)
    if args.lower:
        normalized = normalized.lower()

    stats = calc_stats(raw, normalized)

    write_text(args.out_path, normalized)
    print(f"STATS: chars={stats['chars']} words={stats['words']} lines={stats['lines']}")


if __name__ == "__main__":
    main()
