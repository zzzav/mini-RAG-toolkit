import argparse
import json
import re
from pathlib import Path

from src.retrieval_types import Chunk
from src.utils import normalize_text

# стоп-слова
stop_words = {"the", "a", "an", "and", "or", "to", "in", "of"}


# Собирает текстовые файлы из папки документов.
def load_text_files(docs_dir: str) -> list[tuple[str, str]]:
    base = Path(docs_dir)
    files = sorted(base.glob("*"))
    out: list[tuple[str, str]] = []
    for p in files:
        if not p.is_file():
            continue

        if p.suffix.lower() not in {".txt", ".md"}:
            continue

        out.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
    return out


# Режет текст на перекрывающиеся чанки.
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Размер чанка 0 или меньше")
    elif overlap < 0:
        raise ValueError("Перекрытие должно быть >= 0")
    elif overlap >= chunk_size:
        raise ValueError("Оверлей равен чанку")

    t = text.strip()
    if t == "":
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(t):
        end = start + chunk_size
        part = t[start:end]
        chunks.append(part)
        start += step

    return chunks


# Преобразует документы в список чанков.
def build_chunks(docs: list[tuple[str, str]], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for name, text in docs:
        parts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, part in enumerate(parts):
            chunks.append(Chunk(source=name, idx=i, text=part))
    return chunks


# Оценивает релевантность чанка по простому совпадению слов.
def score_chunk(query: str, chunk_text: str, boost_window: int = 80) -> tuple[int, str]:
    # простой скоринг: сколько “слов запроса” найдено в чанке
    q_clean = re.sub(r"[,.:;!?()]", "", query.lower())
    q_words = [w for w in q_clean.split() if w and w not in stop_words]
    t_words = set(re.sub(r"[,.:;!?()]", "", normalize_text(chunk_text).lower()).split())

    score = 0
    context = ""
    for w in q_words:
        if w in t_words:
            n_text_chunk = normalize_text(chunk_text)
            first_w_pos = n_text_chunk.find(w)
            if score == 0:
                start = max(0, first_w_pos - 60)
                end = min(len(n_text_chunk), first_w_pos + 60)
                context = n_text_chunk[start:end]

            score += 1
            if first_w_pos < boost_window:
                score += 1

    return (score, context)


# Возвращает лучшие чанки по простому поисковому скору.
def search(query: str, chunks: list[Chunk], top_k: int = 5) -> list[tuple[int, Chunk, str]]:
    scored: list[tuple[int, Chunk, str]] = []
    for ch in chunks:
        score, context = score_chunk(query, ch.text)
        if score > 0:
            scored.append((score, ch, context))

    scored.sort(key=lambda x: (-x[0], x[1].source, x[1].idx))

    return scored[:top_k]


# Формирует JSON-отчёт по результатам поиска.
def build_json_report(query: str, results: list[tuple[int, Chunk, str]]) -> dict:
    report: dict = {}
    report["query"] = query

    results_json = [
        {"score": score, "source": ch.source, "idx": ch.idx, "preview": context}
        for score, ch, context in results
    ]
    report["results"] = results_json

    return report


# Сохраняет JSON-отчёт в файл.
def save_json(path: str, obj) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# Собирает парсер аргументов для простого поиска по чанкам.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simple chunk search (mini-RAG without vectors)")
    p.add_argument("--docs", type=str, required=True, help="Папка с .txt документами")
    p.add_argument("--query", type=str, required=True, help="Поисковый запрос")
    p.add_argument(
        "--top-k", "--top", dest="top_k", type=int, default=5, help="Сколько результатов показать"
    )
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=80)
    p.add_argument("--output", "--json-out", dest="output_path", help="Итог в json")
    return p


# Запускает CLI простого поиска по чанкам.
def main() -> None:
    args = build_parser().parse_args()
    docs = load_text_files(args.docs)
    chunks = build_chunks(docs, chunk_size=args.chunk_size, overlap=args.overlap)

    results = search(args.query, chunks, top_k=args.top_k)
    print(f"CHUNKS={len(chunks)} RESULTS={len(results)}")

    for score, ch, context in results:
        # preview = normalize_text(ch.text)[:50]
        print(f"[{score}] {ch.source}#{ch.idx}: {context}...")

    if args.output_path:
        report = build_json_report(args.query, results)
        save_json(args.output_path, report)


if __name__ == "__main__":
    main()
