from src.bm25_search import bm25_search, build_bm25_index, load_bm25, save_bm25


def test_bm25_basic_top1(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("invoice payment due", encoding="utf-8")
    (docs / "b.txt").write_text("weather nice", encoding="utf-8")

    index = build_bm25_index(docs, 100, 20)
    results = bm25_search("invoice payment", index, 1)

    assert results[0][1].source == "a.txt"


def test_bm25_empty_query_returns_empty(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("invoice payment due", encoding="utf-8")
    (docs / "b.txt").write_text("weather nice", encoding="utf-8")

    index = build_bm25_index(docs, 100, 20)
    results = bm25_search(" !!!", index, 1)

    assert results == []


def test_bm25_save_load(tmp_path):
    index_path = tmp_path / "pytest.pkl"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("invoice payment due", encoding="utf-8")
    (docs / "b.txt").write_text("weather nice", encoding="utf-8")

    index = build_bm25_index(docs, 100, 20)
    results = bm25_search("invoice payment", index, 1)

    save_bm25(index_path, index)
    loaded_index = load_bm25(index_path)
    loaded_results = bm25_search("invoice payment", loaded_index, 1)

    assert results[0][1].source == loaded_results[0][1].source
    assert results[0][0] == loaded_results[0][0]
