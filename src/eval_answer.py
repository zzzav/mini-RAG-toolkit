import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.rag_answer import NO_INFO_IN_CONTEXT


@dataclass
class AnswerEvalCase:
    query: str
    expected_contains: list[str]
    expected_mode: str  # answer | no_info


@dataclass
class AnswerEvalReport:
    n: int  # cases count
    contains_rate: float  # percentage of answer cases
    no_info_accuracy: float  # percentage of no info cases
    per_case: list[dict]  # details per case


def load_answer_eval_cases(path: str) -> list[AnswerEvalCase]:
    # examples
    # {"query":"...", "expected_contains":["..."], "expected_mode":"answer"}
    # {"query":"...", "expected_contains":[], "expected_mode":"no_info"}

    mode_types = {"answer", "no_info"}
    eval_cases: list[AnswerEvalCase] = []

    raw = Path(path).read_text(encoding="utf-8")
    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка: получение JSON: path={path}, line={i}, err={e}") from e

        expected_mode = obj.get("expected_mode", "")
        if expected_mode not in mode_types:
            raise ValueError(f"Ошибка: получение expected_mode из JSON: path={path}, line={i}")

        query = obj.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Ошибка: получение query из JSON: path={path}, line={i}")

        expected_contains = obj.get("expected_contains")
        if not isinstance(expected_contains, list) or (
            not expected_contains and expected_mode == "answer"
        ):
            raise ValueError(f"Ошибка: получение expected_contains из JSON: path={path}, line={i}")

        for j, s in enumerate(expected_contains):
            if not isinstance(s, str) or not s.strip():
                raise ValueError(
                    f"Ошибка: получение expected_contains[{j}] из JSON: path={path}, line={i}"
                )

        eval_cases.append(
            AnswerEvalCase(
                query=query, expected_contains=expected_contains, expected_mode=expected_mode
            )
        )

    return eval_cases


def contains_score(answer: str, expected_contains: list[str]) -> tuple[float, list[str], list[str]]:

    if len(expected_contains) == 0:
        return (0.0, [], [])

    answer_normalized = answer.lower()
    matched_contains: list[str] = []
    missing_contains: list[str] = []

    contains_count = 0
    for phrase in expected_contains:
        if phrase.lower() in answer_normalized:
            contains_count += 1
            matched_contains.append(phrase)
        else:
            missing_contains.append(phrase)

    return (contains_count / len(expected_contains), matched_contains, missing_contains)


def is_no_info_answer(answer: str) -> bool:
    return NO_INFO_IN_CONTEXT in answer


def evaluate_answers(
    eval_cases: list[AnswerEvalCase], answer_fn: Callable[[str], str]
) -> AnswerEvalReport:

    n: int = len(eval_cases)

    if n == 0:
        return AnswerEvalReport(0, 0.0, 0.0, [])

    per_case: list[dict] = []
    contains_n: int = 0
    no_info_n: int = 0
    contains_count: int = 0
    no_info_count: int = 0

    for case in eval_cases:
        answer = answer_fn(case.query)
        score_value, matched_contains, missing_contains = contains_score(
            answer, case.expected_contains
        )
        is_no_info = is_no_info_answer(answer)
        res = False

        if case.expected_mode == "answer":
            contains_n += 1
            if score_value == 1.0:
                res = True
                contains_count += 1
        elif case.expected_mode == "no_info":
            no_info_n += 1

            if is_no_info:
                res = True
                no_info_count += 1

        per_case.append(
            {
                "query": case.query,
                "expected_mode": case.expected_mode,
                "expected_contains": case.expected_contains,
                "answer": answer,
                "contains_score": score_value,
                "matched_contains": matched_contains,
                "missing_contains": missing_contains,
                "is_no_info": is_no_info,
                "ok": res,
            }
        )

    contains_rate = 0.0 if contains_n == 0 else contains_count / contains_n
    no_info_accuracy = 0.0 if no_info_n == 0 else no_info_count / no_info_n

    return AnswerEvalReport(
        n=n,
        contains_rate=contains_rate,
        no_info_accuracy=no_info_accuracy,
        per_case=per_case,
    )
