# scripts/make_eval_jsonl.py
# Комментарии на русском по твоему стандарту.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.vector_search import build_vector_index, search


def _preview(text: str, limit: int = 220) -> str:
    t = " ".join(text.split())
    return (t[:limit] + "…") if len(t) > limit else t


def _read_queries(path: Path) -> list[str]:
    # Файл queries.txt: по одному запросу на строку, пустые строки игнорируем
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


def main() -> None:
    docs_dir = Path("eval/docs/v1_rus")
    out_jsonl = Path("eval/eval_small_harder.jsonl")
    queries_file = Path("eval/queries.txt")

    if not docs_dir.exists():
        raise SystemExit(f"docs_dir не найден: {docs_dir}")

    if not queries_file.exists():
        raise SystemExit(
            "Не найден eval/queries.txt.\n" "Создай файл и добавь запросы (по одному на строку)."
        )

    # Фиксируем параметры чанкинга, чтобы idx не 'плавали'
    chunk_size = 400
    overlap = 80
    top_k = 5

    print(f"Строю индекс: {docs_dir} (chunk_size={chunk_size}, overlap={overlap})")
    index = build_vector_index(str(docs_dir), chunk_size, overlap)

    queries = _read_queries(queries_file)
    print(f"Запросов: {len(queries)}")

    results: list[dict[str, Any]] = []

    for qi, q in enumerate(queries, start=1):
        hits = search(q, index, top_k=top_k)

        print("\n" + "=" * 90)
        print(f"[{qi}/{len(queries)}] QUERY: {q}")
        if not hits:
            print("HITS: (none)")
            print("Пропускаю: без хитов релевантность руками не выбрать.")
            continue

        for i, (score, ch) in enumerate(hits, start=1):
            print(f"\n#{i} score={score:.4f} {ch.source} idx={ch.idx}")
            print(_preview(ch.text))

        raw = input(
            "\nВыбери релевантные хиты номерами через запятую "
            "(например: 1 или 2,3). Enter = пропустить: "
        ).strip()

        if not raw:
            print("Пропуск.")
            continue

        chosen: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            n = int(part)
            if n < 1 or n > len(hits):
                raise SystemExit(f"Некорректный номер: {n}")
            chosen.append(n)

        relevant: list[dict[str, Any]] = []
        for n in chosen:
            _, ch = hits[n - 1]
            relevant.append({"source": ch.source, "idx": int(ch.idx)})

        results.append({"query": q, "relevant": relevant})
        print("OK: добавлено.")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n",
        encoding="utf-8",
    )

    print("\nГотово:", out_jsonl)
    print(f"Кейсов записано: {len(results)}")
    print("Дальше прогони:")
    print("python -m src.eval_cli --index <путь_к_индексу> --dataset " + str(out_jsonl) + " --k 5")


if __name__ == "__main__":
    main()
