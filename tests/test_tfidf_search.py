from src.simple_search import Chunk
from src.tfidf_search import build_index, tfidf_search, tokenize


def test_tokenize_basic():
    assert tokenize("Invoice, invoice!") == ["invoice", "invoice"]


def test_idf():
    chunks: list[Chunk] = []
    chunks.append(Chunk(source="a", idx=1, text="apple apple banana"))
    chunks.append(Chunk(source="b", idx=1, text="apple orange"))
    index = build_index(chunks)

    assert index["df"]["banana"] == 1
    assert index["df"]["orange"] == 1
    assert index["df"]["apple"] == 2
    assert index["idf"]["banana"] > index["idf"]["apple"]
    assert index["idf"]["orange"] > index["idf"]["apple"]


def test_tfidf_search_accumulates():
    chunks = [
        Chunk(source="a", idx=0, text="apple banana"),
        Chunk(source="b", idx=0, text="apple"),
    ]
    index = build_index(chunks)
    res = tfidf_search("apple banana", chunks, index, top_k=2)
    assert res[0][1].source == "a"
