import src.bm25_search as bm25_search
import src.vector_search as v_search
from src.compare_pipelines import compare_pipelines
from src.text_tool import write_text


def test_integration(tmp_path) -> None:
    vector_path = tmp_path / "vector_index.pkl"
    bm25_path = tmp_path / "bm25_index.pkl"
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

    vector_index = v_search.build_vector_index(docs, chunk_size=400, overlap=80)
    v_search.save_index(vector_path, vector_index)

    bm25_index = bm25_search.build_bm25_index(docs, chunk_size=400, overlap=80)
    bm25_search.save_bm25(bm25_path, bm25_index)

    answers_text = """   
    {"query":"payment invoice", "expected_contains":["client", "issued"], "expected_mode":"answer"}
    {"query":"bank alerts","expected_contains":["freeze","card"],"expected_mode":"answer"}
    {"query":"totally unrelated query","expected_contains":[],"expected_mode":"no_info"}
    """
    write_text(eval_path, answers_text)

    report = compare_pipelines(vector_path, bm25_path, eval_path, 2)

    assert report.get("vector_extract") is not None
    assert report.get("bm25_extract") is not None
    assert report.get("rerank_vector_extract") is not None
    assert report.get("rerank_bm25_extract") is not None
    assert report.get("fusion_extract") is not None
    for _, value in report.items():
        assert isinstance(value.get("n"), int)
        assert isinstance(value.get("contains_rate"), float)
        assert isinstance(value.get("no_info_accuracy"), float)
