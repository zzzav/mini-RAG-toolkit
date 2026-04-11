import json
from pathlib import Path

from src.regress_cli import run_regression
from src.vector_search import build_vector_index, save_index


def test_regression_pass(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("invoice payment due", encoding="utf-8")
    (docs / "b.txt").write_text("weather nice", encoding="utf-8")

    index = build_vector_index(str(docs), 200, 50)
    index_path = tmp_path / "index.pkl"
    save_index(str(index_path), index)

    dataset_path = tmp_path / "eval.jsonl"
    lines = [
        {"query": "invoice payment", "relevant": [{"source": "a.txt", "idx": 0}]},
        {"query": "weather", "relevant": [{"source": "b.txt", "idx": 0}]},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8"
    )

    code = run_regression(
        index_vector_path=str(index_path),
        dataset_path=str(dataset_path),
        k=3,
        min_recall=0.5,
        min_mrr=0.2,
        for_test=True,
    )
    assert code == 0


def test_regression_fail(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("invoice payment due", encoding="utf-8")
    (docs / "b.txt").write_text("weather nice", encoding="utf-8")

    index = build_vector_index(str(docs), 200, 50)
    index_path = tmp_path / "index.pkl"
    save_index(str(index_path), index)

    dataset_path = tmp_path / "eval.jsonl"
    lines = [
        {"query": "invoice payment", "relevant": [{"source": "a.txt", "idx": 0}]},
        {"query": "weather", "relevant": [{"source": "b.txt", "idx": 0}]},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lines), encoding="utf-8"
    )

    code = run_regression(
        index_vector_path=str(index_path),
        dataset_path=str(dataset_path),
        k=3,
        min_recall=1.1,
        min_mrr=1.1,
    )
    assert code == 2
