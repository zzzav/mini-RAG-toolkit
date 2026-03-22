import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

import src.bm25_search as bm25_search
import src.rag_answer as rag_answer
import src.vector_search as v_search
from src.eval_answer import AnswerEvalReport, evaluate_answers, load_answer_eval_cases

g_formats_arr = {"text", "json"}


def write_output(text: str, out_path: str | None) -> None:
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
    else:
        print(text)


def create_json_from_eval_answer_report(eval_answer_rep: AnswerEvalReport) -> dict:
    return {
        "n": eval_answer_rep.n,
        "contains_rate": eval_answer_rep.contains_rate,
        "no_info_accuracy": eval_answer_rep.no_info_accuracy,
        "per_case": eval_answer_rep.per_case,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Eval CLI")
    p.add_argument("--index", dest="index_in", type=str, required=True)
    p.add_argument("--k", type=int, default=5)

    p.add_argument("--format", type=str, default="text", choices=g_formats_arr)
    p.add_argument("--out", type=str, default=None)

    p.add_argument("--retriever", type=str, default="bm25", choices=rag_answer.RETRIEVER_TYPES)
    p.add_argument("--dataset", dest="dataset_path", type=str, required=True)
    p.add_argument("--llm", type=str, default="extract", choices=rag_answer.ALLOWED_LLM)

    return p


def die(msg: str, code: int = 2) -> NoReturn:
    print("Ошибка: " + msg, file=sys.stderr)
    raise SystemExit(code) from None


def main() -> None:
    args = build_parser().parse_args()

    if args.k < 1:
        die("k должен быть >= 1")
    if args.format not in g_formats_arr:
        die(f"format должен принимать одно из значений: {g_formats_arr}")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        die(f"dataset не найден: {args.dataset_path}")
    if not dataset_path.is_file():
        die(f"dataset должен быть файлом: {args.dataset_path}")

    index_path = Path(args.index_in)
    if not index_path.exists():
        die(f"index не найден: {args.index_in}")
    if not index_path.is_file():
        die(f"index должен быть файлом: {args.index_in}")

    index = None
    if args.retriever == "bm25":
        index = bm25_search.load_bm25(args.index_in)
    elif args.retriever == "vector":
        index = v_search.load_index(args.index_in)
    else:
        die(f"не поддерживаемый retriever: {args.retriever}")

    # локальная функция для получения ответа на запрос
    def get_answer_from_rag_res(query: str) -> str:
        search_res = None
        if args.retriever == "bm25":
            search_res = bm25_search.bm25_search(query, index, top_k=args.k)
        elif args.retriever == "vector":
            search_res = v_search.search(query, index, top_k=args.k)

        rag_cfg = rag_answer.RAGConfig(top_k=args.k)
        rag_res = rag_answer.rag_answer(query, search_res, rag_cfg, llm=args.llm)

        return rag_res.answer

    cases = load_answer_eval_cases(dataset_path)
    eval_answer_report = evaluate_answers(cases, get_answer_from_rag_res)

    if args.format == "json":
        write_output(
            json.dumps(
                create_json_from_eval_answer_report(eval_answer_report),
                ensure_ascii=False,
                indent=2,
            ),
            args.out,
        )
    # text
    else:
        write_output(
            f"format={args.format}, retriever={args.retriever}, llm={args.llm}\n"
            + f"n={eval_answer_report.n}\n"
            + f"contains_rate={eval_answer_report.contains_rate}\n"
            + f"no_info_accuracy={eval_answer_report.no_info_accuracy}",
            args.out,
        )


if __name__ == "__main__":
    main()
