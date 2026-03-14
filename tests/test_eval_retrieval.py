import src.eval_retrieval as eval_retrieval
from src.text_tool import write_text
from src.vector_search import (
    Chunk,
    build_chunks,
    build_vector_index_by_chunks,
    load_index,
    save_index,
)


def test_fake_hits():
    relevant = {("a", 0)}
    hits = [(0, Chunk("x", 0, "text from x")), (0, Chunk("a", 0, "text from a"))]

    recall = eval_retrieval.recall_at_k(hits, relevant, 2)
    mrr = eval_retrieval.mrr_at_k(hits, relevant, 2)

    assert recall == 1.0
    assert mrr == 0.5


def test_no_relevant():
    relevant = {("b", 0)}
    hits = [(0, Chunk("x", 0, "text from x")), (0, Chunk("a", 0, "text from a"))]

    recall = eval_retrieval.recall_at_k(hits, relevant, 2)
    mrr = eval_retrieval.mrr_at_k(hits, relevant, 2)

    assert recall == 0
    assert mrr == 0


def test_evaluate(tmp_path):
    vector_path = tmp_path / "pytest.pkl"
    eval_path = tmp_path / "eval.jsonl"
    docs = [
        (
            "a.txt",
            (
                "Invoice #123 was issued to the client."
                "Payment due in 7 days.\nIf payment is late, "
                "send a reminder email."
            ),
        ),
        (
            "b.txt",
            (
                "Bank alerts:\nCard payment 10 EUR. Another payment 5 EUR."
                "\nIf suspicious activity happens, freeze the card."
            ),
        ),
        (
            "c.txt",
            (
                "Project notes:\nWe generate images for the content factory "
                "using prompt templates.\nThe pipeline is: "
                "prompt -> image -> asset store -> publish."
            ),
        ),
    ]

    # 4-го кейса умышленно нет в исходных документах, проверяем, что recall и mrr не всегда 1,
    # а значит среднее будет < 1
    # relevant в датасете зависит от параметров chunking (chunk_size/overlap),
    # поэтому для eval нужно держать их стабильными
    # (иначе idx уедут и метрики станут бессмысленными).
    relevant_text = """
    {"query": "payment invoice","relevant": [{"source": "a.txt", "idx": 0}]}
    {"query": "bank alerts","relevant": [{"source": "b.txt", "idx": 0}]}
    {"query": "using prompt","relevant": [{"source": "c.txt", "idx": 0}]}
    {"query": "totally unrelated query","relevant": [{"source": "a.txt", "idx": 0}]}
    """
    write_text(eval_path, relevant_text)

    cases = eval_retrieval.load_eval_cases(eval_path)

    chunks = build_chunks(docs, chunk_size=400, overlap=80)
    index = build_vector_index_by_chunks(chunks)
    save_index(vector_path, index)
    index_loaded = load_index(vector_path)

    eval_rep = eval_retrieval.evaluate(index_loaded, cases, 3)

    assert eval_rep.n == 4
    assert eval_rep.recall_mean < 1.0 and eval_rep.recall_mean >= 0.0
    assert eval_rep.mrr_mean < 1.0 and eval_rep.mrr_mean >= 0.0
