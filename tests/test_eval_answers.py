from src.eval_answer import AnswerEvalCase, contains_score, evaluate_answers, is_no_info_answer
from src.rag_answer import NO_INFO_IN_CONTEXT


def test_contains_score_full_match() -> None:
    answer = "Invoice payment is in the second sentence."
    expected_contains = ["invoice payment", "second sentence"]

    score_value, matched, missing = contains_score(answer, expected_contains)

    assert score_value == 1.0
    assert matched == ["invoice payment", "second sentence"]
    assert missing == []


def test_contains_score_partial_match() -> None:
    answer = "Invoice payment is in the second sentence."
    expected_contains = ["invoice payment", "additional info"]

    score_value, matched, missing = contains_score(answer, expected_contains)

    assert score_value == 0.5
    assert matched == ["invoice payment"]
    assert missing == ["additional info"]


def test_is_no_info_answer() -> None:
    assert is_no_info_answer(NO_INFO_IN_CONTEXT) is True
    assert is_no_info_answer("Some useful answer.") is False


def test_evaluate_answers_integration() -> None:
    cases = [
        AnswerEvalCase(
            query="q1",
            expected_contains=["python", "rag"],
            expected_mode="answer",
        ),
        AnswerEvalCase(
            query="q2",
            expected_contains=["bm25", "ranking"],
            expected_mode="answer",
        ),
        AnswerEvalCase(
            query="q3",
            expected_contains=[],
            expected_mode="no_info",
        ),
    ]

    answers = {
        "q1": "Python is often used in RAG systems.",
        "q2": "BM25 is used here.",
        "q3": NO_INFO_IN_CONTEXT,
    }

    def answer_fn(query: str) -> str:
        return answers[query]

    report = evaluate_answers(cases, answer_fn)

    assert report.n == 3
    assert report.contains_rate == 0.5
    assert report.no_info_accuracy == 1.0
    assert len(report.per_case) == 3

    assert report.per_case[0]["ok"] is True
    assert report.per_case[0]["contains_score"] == 1.0
    assert report.per_case[0]["matched_contains"] == ["python", "rag"]
    assert report.per_case[0]["missing_contains"] == []

    assert report.per_case[1]["ok"] is False
    assert report.per_case[1]["contains_score"] == 0.5
    assert report.per_case[1]["matched_contains"] == ["bm25"]
    assert report.per_case[1]["missing_contains"] == ["ranking"]

    assert report.per_case[2]["ok"] is True
    assert report.per_case[2]["is_no_info"] is True
