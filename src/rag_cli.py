import argparse
import sys
from pathlib import Path
from typing import NoReturn

import src.rag_answer as rag_answer
import src.vector_search as v_search


def get_hits_from_vector_index_search(search_results: list[tuple[float, v_search.Chunk]]) -> dict:
    hits: list[dict] = []
    for score, chunk in search_results:
        hits.append(
            {
                "source": chunk.source,
                "idx": int(chunk.idx),
                "score": float(score),
                "text": chunk.text,
            }
        )

    return hits


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG CLI")
    p.add_argument("--llm", type=str, default="mock")
    p.add_argument("--q", type=str)
    p.add_argument("--index", dest="index_in", type=str, default=None)
    p.add_argument("--index-out", dest="index_out", default=None)
    p.add_argument("--format", type=str, default=None)
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--context-only", action="store_true")
    p.add_argument("--max-context-chars", type=int, default=4000)
    p.add_argument("--per-chunk-chars", type=int, default=800)
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=80)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--docs", type=str)
    return p


def die(msg: str, code: int = 2) -> NoReturn:
    print("Ошибка: " + msg, file=sys.stderr)
    raise SystemExit(code) from None


def main() -> None:
    args = build_parser().parse_args()
    index: v_search.VectorIndex

    # проверка базовых аргументов
    if args.top_k < 1:
        die("top должен быть >= 1")

    # режим билда
    if args.docs:
        docs = Path(args.docs)
        if not docs.exists():
            die(f"docs не найден: {docs}")
        if not docs.is_dir():
            die(f"docs должен быть директорией: {docs}")
        if args.chunk_size < 1:
            die("chunk-size должен быть >= 1")
        if args.overlap < 0 or args.overlap >= args.chunk_size:
            die("overlap должен быть >= 0 и меньше chunk-size")
        if not args.index_out:
            die("не задан index-out")
        if args.index_in:
            die("index нельзя передавать вместе с docs")
        if args.q:
            die("поисковой запрос q нельзя передавать вместе с docs")

        index = v_search.build_vector_index(
            args.docs, chunk_size=args.chunk_size, overlap=args.overlap
        )
        v_search.save_index(args.index_out, index)

    # режим поиска
    else:
        if not args.index_in:
            die("не задан путь к index")
        if not args.q:
            die("не задан поисковой запрос q")

        index = v_search.load_index(args.index_in)
        results = v_search.search(args.q, index, top_k=args.top_k)
        hits = get_hits_from_vector_index_search(results)

        rag_cfg = rag_answer.RAGConfig(
            max_context_chars=args.max_context_chars,
            per_chunk_chars=args.per_chunk_chars,
            top_k=args.top_k,
        )

        context = rag_answer.build_context(hits, rag_cfg)
        prompt = rag_answer.build_prompt(args.q, context)

        mock_llm = rag_answer.MockLLM()

        answer = mock_llm.generate(prompt) if args.llm == "mock" else "Ручной ответ"

        print(answer)


if __name__ == "__main__":
    main()
