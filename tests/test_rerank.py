from src.rerank import rerank_hits
from src.simple_search import Chunk


def test_phrase_bonus():
    q = "invoice payment"
    hits = [
        (0.9, Chunk("b.txt", 0, "invoice records and payment details")),
        (0.8, Chunk("a.txt", 0, "invoice payment due tomorrow")),
    ]

    rerank_results = rerank_hits(q, hits, top_k=2)

    assert rerank_results[0][1].source == "a.txt"


def test_proximity_bonus():
    q = "invoice payment"
    hits = [
        (0.9, Chunk("a.txt", 0, "invoice records plus added info and payment details")),
        (0.8, Chunk("b.txt", 0, "invoice and test payment due tomorrow")),
    ]

    rerank_results = rerank_hits(q, hits, top_k=2, proximity_window=3)

    assert rerank_results[0][1].source == "b.txt"


def test_empty_query():
    hits = [(0.7, Chunk("a", 0, "text from a")), (0.7, Chunk("b", 0, "text from b"))]
    results = rerank_hits("", hits)

    assert results == hits


def test_sorting():
    hits = [(0.7, Chunk("b", 0, "text from a")), (0.7, Chunk("a", 0, "text from b"))]
    results = rerank_hits("text", hits)

    hits_1 = [(0.7, Chunk("a", 1, "text from a second")), (0.7, Chunk("a", 0, "text from a"))]
    results_1 = rerank_hits("text", hits_1)

    assert results[0][1].source == "a"
    assert results_1[0][1].idx == 0
