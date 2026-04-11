import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

import src.bm25_search as bm25_search
import src.fusion_search as fusion_search
import src.rag_answer as rag_answer
import src.vector_search as v_search
from src.rerank import rerank_hits


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG CLI")
    p.add_argument("--llm", type=str, default="mock", choices=rag_answer.ALLOWED_LLM)
    p.add_argument("--q", type=str)
    p.add_argument("--index-vector", type=str, default=None)
    p.add_argument("--index-bm25", type=str, default=None)
    p.add_argument("--index-out", dest="index_out", default=None)

    p.add_argument("--format", type=str, default="text", choices=["text", "json"])
    p.add_argument("--out", type=str, default=None)

    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--context-only", action="store_true")
    p.add_argument("--max-context-chars", type=int, default=4000)
    p.add_argument("--per-chunk-chars", type=int, default=800)
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap", type=int, default=80)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--docs", type=str)
    p.add_argument("--use-synonyms", action="store_true")
    p.add_argument("--no-stop-words", action="store_true")
    p.add_argument("--inline-citations", action="store_true")
    p.add_argument("--for-eval-jsonl-out", action="store_true")

    p.add_argument("--retriever", type=str, default="vector", choices=rag_answer.RETRIEVER_TYPES)
    p.add_argument("--index-type", type=str, default="vector", choices=rag_answer.INDEX_TYPES)

    p.add_argument("--rerank", action="store_true")
    p.add_argument("--rerank-top-n", type=int, default=10)
    p.add_argument("--proximity-window", type=int, default=5)

    p.add_argument("--fusion-method", type=str, default="rrf", choices=fusion_search.FUSION_METHODS)
    p.add_argument("--fusion-top-n", type=int, default=5)

    return p


def die(msg: str, code: int = 2) -> NoReturn:
    print("Ошибка: " + msg, file=sys.stderr)
    raise SystemExit(code) from None


def render_text(
    res: rag_answer.RAGResult,
    show_prompt: bool,
    context_only: bool,
    inline_citations: bool,
    for_eval_jsonl_out: bool,
) -> str:
    # Текстовый режим: читабельно в консоли
    if context_only:
        return "CONTEXT:\n" + (res.context or "")

    lines: list[str] = []
    if res.answer:
        lines.append(str(res.answer))
    else:
        lines.append("ANSWER:\n(no answer)")

    # Источники
    if res.chunks:
        lines.append("SOURCES:")
        for c in res.chunks:
            lines.append(f'- {c["source"]}#{c["idx"]} (score={c["score"]:.4f})')
    else:
        lines.append("SOURCES:\n(none)")

    if show_prompt:
        lines.append("PROMPT:\n" + (res.prompt or ""))

    if res.citations:
        lines.append("CITATIONS:")
        for c in res.citations:
            lines.append(f'- {c["source"]}#{c["idx"]}')
    else:
        lines.append("CITATIONS:\nnone")

    if inline_citations:
        citations_line: str = "Источники: "
        if res.citations:
            for i, c in enumerate(res.citations):
                citations_line += ", " if i > 0 else ""
                citations_line += f'{c["source"]}#{c["idx"]}'
        else:
            citations_line += "нет данных"

        lines.append(citations_line)

    if for_eval_jsonl_out:
        payload = {
            "query": res.query,
            "relevant": [{"source": c["source"], "idx": c["idx"]} for c in (res.citations or [])],
        }
        lines.append("Для тестов:")
        lines.append(json.dumps(payload, ensure_ascii=False))

    return "\n".join(lines)


def write_output(text: str, out_path: str | None) -> None:
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def main() -> None:
    args = build_parser().parse_args()

    if args.top_k < 1:
        die("top-k должен быть >= 1")

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
        if args.index_vector:
            die("index-vector нельзя передавать вместе с docs")
        if args.index_bm25:
            die("index-bm25 нельзя передавать вместе с docs")
        if args.q:
            die("поисковой запрос q нельзя передавать вместе с docs")
        if not args.index_type:
            die("не указан тип индекса для сохранения")

        if args.index_type == "bm25":
            index = bm25_search.build_bm25_index(
                args.docs,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                use_stop_words=not args.no_stop_words,
            )
            bm25_search.save_bm25(args.index_out, index)
        else:
            index = v_search.build_vector_index(
                args.docs, chunk_size=args.chunk_size, overlap=args.overlap
            )
            v_search.save_index(args.index_out, index)

        return

    # режим поиска
    if not args.retriever:
        die("не задан тип поиска")
    if (args.retriever == "vector" or args.retriever == "fusion") and not args.index_vector:
        die("не задан путь к index-vector")
    if (args.retriever == "bm25" or args.retriever == "fusion") and not args.index_bm25:
        die("не задан путь к index-bm25")
    if not args.q:
        die("не задан поисковой запрос q")
    if args.rerank:
        if not args.rerank_top_n or args.rerank_top_n < 1:
            die("включен режим реранкинга, но не задан rerank-top-n")
        if not args.proximity_window or args.proximity_window < 1:
            die("включен режим реранкинга, но не задан proximity-window")

    # режим смешивания результатов поиска
    if args.retriever == "fusion":
        if not args.fusion_method:
            die("включен режим смешивания, но не задан тип fusion-method")
        if not args.fusion_top_n or args.fusion_top_n < 1:
            die("включен режим смешивания, но не задан fusion-top-n")

    results = None
    initial_top_k = args.rerank_top_n if args.rerank else args.top_k
    fusion_retrieval_top_n = args.fusion_top_n

    if args.retriever == "bm25":
        index = bm25_search.load_bm25(args.index_bm25)
        results = bm25_search.bm25_search(
            args.q, index, top_k=initial_top_k, use_synonyms=args.use_synonyms
        )
    elif args.retriever == "vector":
        index = v_search.load_index(args.index_vector)
        results = v_search.search(
            args.q,
            index,
            top_k=initial_top_k,
            use_synonyms=args.use_synonyms,
            use_stop_words=not args.no_stop_words,
        )
    elif args.retriever == "fusion":
        index = bm25_search.load_bm25(args.index_bm25)
        results_bm25 = bm25_search.bm25_search(
            args.q, index, top_k=fusion_retrieval_top_n, use_synonyms=args.use_synonyms
        )

        index = v_search.load_index(args.index_vector)
        results_vector = v_search.search(
            args.q,
            index,
            top_k=fusion_retrieval_top_n,
            use_synonyms=args.use_synonyms,
            use_stop_words=not args.no_stop_words,
        )

        if args.fusion_method == "weighted":
            results = fusion_search.weighted_score_fusion(results_vector, results_bm25, [])
        elif args.fusion_method == "rrf":
            results = fusion_search.rrf_fusion(results_vector, results_bm25, [])

    # режим реранкинга результатов
    if args.rerank:
        results = rerank_hits(
            args.q,
            results,
            top_k=args.top_k,
            use_stop_words=not args.no_stop_words,
            proximity_window=args.proximity_window,
        )

    rag_cfg = rag_answer.RAGConfig(
        max_context_chars=args.max_context_chars,
        per_chunk_chars=args.per_chunk_chars,
        top_k=args.top_k,
    )

    res = rag_answer.rag_answer(args.q, results, rag_cfg, llm=args.llm)
    report = asdict(res)

    if args.format == "json":
        write_output(json.dumps(report, ensure_ascii=False, indent=2), args.out)
    else:
        write_output(
            render_text(
                res,
                args.show_prompt,
                args.context_only,
                args.inline_citations,
                args.for_eval_jsonl_out,
            ),
            args.out,
        )


if __name__ == "__main__":
    main()
