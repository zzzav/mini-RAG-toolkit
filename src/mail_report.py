import argparse
import json
from pathlib import Path

from src.utils import normalize_text


# Загружает список писем из JSON-файла.
def load_mails(path: str) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Ожидаю JSON-массив писем")
    return data


# Нормализует поля письма перед построением отчёта.
def clean_mail(m: dict) -> dict:
    return {
        "from": normalize_text(str(m.get("from", ""))),
        "subject": normalize_text(str(m.get("subject", ""))),
        "snippet": normalize_text(str(m.get("snippet", ""))),
        "date": str(m.get("date", "")),
    }


# Собирает сводку по отправителям и темам писем.
def build_report(mails: list[dict], domain: str, skip_empty: bool) -> dict:
    empty = "(empty)"
    total = 0
    counts: dict[str, int] = {}
    themes: dict[str, int] = {}

    for m in mails:
        sender = m.get("from", "")
        if ((domain != "") and (sender.endswith("@" + domain))) or (domain == ""):
            if sender == "":
                sender = empty
            if (not skip_empty) or (skip_empty and sender != empty):
                counts[sender] = counts.get(sender, 0) + 1
                total += 1

                prefix = m.get("subject", "")
                if prefix != "":
                    prefix = prefix.split()[0]
                else:
                    prefix = "(empty)"
                themes[prefix] = themes.get(prefix, 0) + 1

    by_sender = [{"from": sender, "count": cnt} for sender, cnt in counts.items()]
    by_sender.sort(key=lambda x: (-x["count"], x["from"]))

    top_sender = by_sender[0] if by_sender else {"from": "", "count": 0}

    return {"total": total, "by_sender": by_sender, "themes": themes, "top_sender": top_sender}


# Сохраняет отчёт в JSON-файл.
def save_json(path: str, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# Собирает парсер аргументов для mail-report CLI.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mail report from JSON")
    p.add_argument("--input", "--in", dest="input_path", required=True)
    p.add_argument("--output", "--out", dest="output_path", required=True)
    p.add_argument("--domain", dest="domain", help="Отчет только по этому домену")
    p.add_argument("--skip-empty-from", dest="skip_empty_from", action="store_true")
    return p


# Запускает CLI построения отчёта по почтовому JSON.
def main() -> None:
    args = build_parser().parse_args()

    mails = load_mails(args.input_path)
    cleaned = [clean_mail(m) for m in mails]

    domain = args.domain if args.domain else ""

    report = build_report(cleaned, domain, args.skip_empty_from)
    save_json(args.output_path, report)

    top3 = report["by_sender"][:3]
    top3_lines = []
    for i, item in enumerate(top3, start=1):
        top3_lines.append(f"{i}) {item['from']} - {item['count']}")

    print("TOTAL={}\nTOP:\n{}".format(report["total"], "\n".join(top3_lines)))


if __name__ == "__main__":
    main()
