import argparse
import json

import src.bm25_search as bm25_search
import src.rag_answer as rag_answer
import src.vector_search as v_search
from src.eval_answer import AnswerEvalReport, evaluate_answers, load_answer_eval_cases
from src.retrieval_types import ALLOWED_LLM, OUTPUT_FORMATS, RETRIEVER_TYPES
from src.utils import check_path, die, write_output


# Преобразует отчёт оценки ответов в JSON-словарь.
def create_json_from_eval_answer_report(eval_answer_rep: AnswerEvalReport) -> dict:
    return {
        "n": eval_answer_rep.n,
        "contains_rate": eval_answer_rep.contains_rate,
        "no_info_accuracy": eval_answer_rep.no_info_accuracy,
        "per_case": eval_answer_rep.per_case,
    }


# Собирает парсер аргументов для оценки ответов.
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Оценка ответов RAG")
    p.add_argument("--index", "--index-path", dest="index_path", type=str, required=True)
    p.add_argument("--top-k", "--k", dest="top_k", type=int, default=5)

    p.add_argument("--format", type=str, default="text", choices=OUTPUT_FORMATS)
    p.add_argument("--output", "--out", dest="output_path", type=str, default=None)

    p.add_argument("--retriever", type=str, default="bm25", choices=RETRIEVER_TYPES)
    p.add_argument("--dataset", dest="dataset_path", type=str, required=True)
    p.add_argument("--llm", type=str, default="extract", choices=ALLOWED_LLM)

    return p


# Запускает CLI оценки ответов.
def main() -> None:
    args = build_parser().parse_args()

    if args.top_k < 1:
        die("top-k должен быть >= 1")
    dataset_path = check_path(args.dataset_path, entity="dataset")
    index_path = check_path(args.index_path, entity="index")

    index = None
    if args.retriever == "bm25":
        index = bm25_search.load_bm25(index_path)
    elif args.retriever == "vector":
        index = v_search.load_index(index_path)
    else:
        die(f"не поддерживаемый retriever: {args.retriever}")

    # Локальная функция удерживает всю логику вызова RAG в одном месте.
    def get_answer_from_rag_res(query: str) -> str:
        search_res = None
        if args.retriever == "bm25":
            search_res = bm25_search.bm25_search(query, index, top_k=args.top_k)
        elif args.retriever == "vector":
            search_res = v_search.search(query, index, top_k=args.top_k)

        rag_cfg = rag_answer.RAGConfig(top_k=args.top_k)
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
            args.output_path,
        )
    else:
        write_output(
            f"format={args.format}, retriever={args.retriever}, llm={args.llm}\n"
            + f"n={eval_answer_report.n}\n"
            + f"contains_rate={eval_answer_report.contains_rate}\n"
            + f"no_info_accuracy={eval_answer_report.no_info_accuracy}",
            args.output_path,
        )


if __name__ == "__main__":
    main()
