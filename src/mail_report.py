import argparse
import json
from pathlib import Path

from src.utils import normalize_text


def load_mails(path: str) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Ожидаю JSON-массив писем")
    return data


def clean_mail(m: dict) -> dict:
    return {
        "from": normalize_text(str(m.get("from", ""))),
        "subject": normalize_text(str(m.get("subject", ""))),
        "snippet": normalize_text(str(m.get("snippet", ""))),
        "date": str(m.get("date", "")),
    }


def build_report(mails: list[dict]) -> dict:
    total = len(mails)
    counts: dict[str, int] = {}

    for m in mails:
        sender = m.get("from", "")
        if sender == "":
            sender = "(empty)"
        counts[sender] = counts.get(sender, 0) + 1

    by_sender = [{"from": sender, "count": cnt} for sender, cnt in counts.items()]
    by_sender.sort(key=lambda x: x["count"], reverse=True)

    top_sender = by_sender[0] if by_sender else {"from": "", "count": 0}

    return {"total": total, "by_sender": by_sender, "top_sender": top_sender}


def save_json(path: str, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mail report from JSON")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()

    mails = load_mails(args.in_path)
    cleaned = [clean_mail(m) for m in mails]

    report = build_report(cleaned)
    save_json(args.out_path, report)

    top = report.get("top_sender", {})
    print(f"TOTAL={report.get('total', 0)} TOP={top.get('from', '')} ({top.get('count', 0)})")


if __name__ == "__main__":
    main()
