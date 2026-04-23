import sys
from pathlib import Path
from typing import NoReturn


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())


# Завершает CLI с сообщением об ошибке.
def die(msg: str, code: int = 2) -> NoReturn:
    print("Ошибка: " + msg, file=sys.stderr)
    raise SystemExit(code) from None


# Проверяет путь к файлу или директории и возвращает Path.
def check_path(path: str, *, entity: str, must_be_dir: bool = False) -> Path:
    checked_path = Path(path)
    if not checked_path.exists():
        die(f"{entity} не найден: {path}")
    if must_be_dir:
        if not checked_path.is_dir():
            die(f"{entity} должен быть директорией: {path}")
    elif not checked_path.is_file():
        die(f"{entity} должен быть файлом: {path}")

    return checked_path


# Печатает текст или сохраняет его в файл.
def write_output(text: str, output_path: str | None) -> None:
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
    else:
        print(text)
