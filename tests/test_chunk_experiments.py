from src.chunk_experiments import DEFAULT_CHUNKING_CONFIG, run_chunk_experiments
from src.text_tool import write_text


def test_run_chunk_experiments(tmp_path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text(
        (
            "Invoice #123 was issued to the client."
            "Payment due in 7 days.\nIf payment is late, "
            "send a reminder email."
        ),
        encoding="utf-8",
    )
    (docs / "b.txt").write_text(
        (
            "Bank alerts:\nCard payment 10 EUR. Another payment 5 EUR."
            "\nIf suspicious activity happens, freeze the card."
        ),
        encoding="utf-8",
    )
    (docs / "c.txt").write_text(
        (
            "Project notes:\nWe generate images for the content factory "
            "using prompt templates.\nThe pipeline is: "
            "prompt -> image -> asset store -> publish."
        ),
        encoding="utf-8",
    )

    relevant_text = """
    {"query": "payment invoice","relevant": [{"source": "a.txt", "idx": 0}]}
    {"query": "bank alerts","relevant": [{"source": "b.txt", "idx": 0}]}
    {"query": "using prompt","relevant": [{"source": "c.txt", "idx": 0}]}
    """
    write_text(eval_path, relevant_text)

    chunking_cfg = DEFAULT_CHUNKING_CONFIG
    k: int = 2
    result = run_chunk_experiments(docs, eval_path, k, chunking_cfg)

    assert len(result) == len(chunking_cfg)
    assert result[0]["chunk_size"] == chunking_cfg[0][0]
    assert result[0]["overlap"] == chunking_cfg[0][1]

    assert "k" in result[0] and isinstance(result[0]["k"], int) and result[0]["k"] == k
    assert "retriever" in result[0] and isinstance(result[0]["retriever"], str)
    assert "recall_mean" in result[0] and isinstance(result[0]["recall_mean"], float)
    assert "mrr_mean" in result[0] and isinstance(result[0]["mrr_mean"], float)
    assert "n" in result[0] and isinstance(result[0]["n"], int)
