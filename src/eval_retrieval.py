import json
from dataclasses import dataclass
from pathlib import Path

import src.vector_search as v_search


@dataclass
class EvalCase:
    query: str
    relevant: set[tuple[str, int]]  # (source, idx)


@dataclass
class EvalReport:
    k: int
    n: int
    recall_mean: float
    mrr_mean: float
    per_case: list[dict]


def create_json_from_eval_report(eval_rep: EvalReport) -> dict:
    return {
        "k": eval_rep.k,
        "n": eval_rep.n,
        "recall_mean": eval_rep.recall_mean,
        "mrr_mean": eval_rep.mrr_mean,
        "par_case": eval_rep.per_case,
    }


def load_eval_cases(path: str) -> list[EvalCase]:
    eval_cases: list[EvalCase] = []

    raw = Path(path).read_text(encoding="utf-8")
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка: получение JSON: path={path}, line={i}, err={e}") from e

        query = obj.get("query")
        relevant_raw = obj.get("relevant")

        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Ошибка: получение query из JSON: path={path}, line={i}")
        if not isinstance(relevant_raw, list) or not relevant_raw:
            raise ValueError(f"Ошибка: получение relevant_raw из JSON: path={path}, line={i}")

        relevant: list[tuple[str, int]] = []
        for iRelevant in relevant_raw:

            if not isinstance(iRelevant, dict):
                raise ValueError(f"Ошибка: relevant не dict: path={path}, line={i}")

            source = iRelevant.get("source", "")
            idx = iRelevant.get("idx", -1)

            if not source:
                raise ValueError(f"Ошибка: получение source из relevant_raw: path={path}, line={i}")
            if not isinstance(idx, int):
                raise ValueError(f"Ошибка: idx из relevant_raw не int: path={path}, line={i}")
            if idx < 0:
                raise ValueError(f"Ошибка: idx из relevant_raw < 0: path={path}, line={i}")

            relevant.append((source, idx))

        eval_cases.append(EvalCase(query=query, relevant=set(relevant)))

    return eval_cases


def recall_at_k(
    hits: list[tuple[float, v_search.Chunk]], relevant: set[tuple[str, int]], k: int
) -> float:
    for _score, chunk in hits[:k]:
        if (chunk.source, int(chunk.idx)) in relevant:
            return 1.0
    return 0.0


def mrr_at_k(
    hits: list[tuple[float, v_search.Chunk]], relevant: set[tuple[str, int]], k: int
) -> float:
    for i, (_score, chunk) in enumerate(hits[:k]):
        if (chunk.source, int(chunk.idx)) in relevant:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(index: v_search.VectorIndex, cases: list[EvalCase], k: int) -> EvalReport:
    recall_list: list[float] = []
    mrr_list: list[float] = []
    per_case_list: list[dict] = []

    for case in cases:
        hits = v_search.search(case.query, index, top_k=k)

        r = recall_at_k(hits, case.relevant, k)
        m = mrr_at_k(hits, case.relevant, k)

        recall_list.append(r)
        mrr_list.append(m)
        per_case_list.append({"query": case.query, "recall": r, "mrr": m})

    recall_mean = sum(recall_list) / len(recall_list) if recall_list else 0.0
    mrr_mean = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0

    return EvalReport(
        k=k, n=len(cases), recall_mean=recall_mean, mrr_mean=mrr_mean, per_case=per_case_list
    )
