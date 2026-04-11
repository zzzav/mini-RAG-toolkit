from src.fusion_search import merge_hits_union, rrf_fusion, weighted_score_fusion
from src.vector_search import Chunk


def test_merge_hits_union_removes_duplicates() -> None:
    hits_1 = [(0.9, Chunk("a", 0, "text a0")), (0.8, Chunk("b", 0, "text b0"))]
    hits_2 = [(0.7, Chunk("a", 0, "text a0")), (0.6, Chunk("c", 0, "text c0"))]

    merged = merge_hits_union([hits_1, hits_2])

    assert len(merged) == 3
    assert merged[0][1].source == "a"
    assert merged[0][1].idx == 0
    assert merged[1][1].source == "b"
    assert merged[2][1].source == "c"


def test_weighted_score_fusion() -> None:
    hits_1 = [(0.2, Chunk("x", 0, "text from x")), (0, Chunk("a", 0, "text from a"))]
    hits_2 = [(0, Chunk("x", 0, "text from x")), (0.1, Chunk("a", 0, "text from a"))]
    hits_3 = [(0, Chunk("x", 0, "text from x")), (0, Chunk("a", 0, "text from a"))]

    hits = weighted_score_fusion(hits_1, hits_2, hits_3)

    assert len(hits) == 2
    assert hits[0][0] == 0.2 and hits[1][0] == 0.1


def test_rff_score_fusion() -> None:
    hits_1 = [(0.2, Chunk("x", 0, "text from x")), (0, Chunk("a", 0, "text from a"))]
    hits_2 = [(0, Chunk("x", 0, "text from x")), (0.1, Chunk("a", 0, "text from a"))]
    hits_3 = [(0, Chunk("x", 0, "text from x")), (0.1, Chunk("a", 0, "text from a"))]

    hits = rrf_fusion(hits_1, hits_2, hits_3)

    assert len(hits) == 2
    assert hits[0][1].source == "x" and hits[1][1].source == "a"


def test_order() -> None:
    hits_1 = [(0, Chunk("b", 0, "text from b")), (0, Chunk("a", 1, "text from a1"))]
    hits_2 = [(0, Chunk("b", 0, "text from b")), (0, Chunk("a", 0, "text from a"))]
    hits_3 = [(0, Chunk("b", 0, "text from b")), (0, Chunk("a", 0, "text from a"))]

    hits = weighted_score_fusion(hits_1, hits_2, hits_3)

    assert len(hits) == 3
    assert hits[0][1].source == "a"
    assert hits[0][1].idx == 0
    assert hits[1][1].source == "a"
    assert hits[1][1].idx == 1


def test_rrf_fusion_promotes_document_found_by_multiple_retrievers() -> None:
    target = Chunk("target", 0, "text target")

    hits_1 = [
        (1.0, Chunk("a", 0, "text a")),
        (0.9, target),
    ]
    hits_2 = [
        (1.0, Chunk("d", 0, "text d")),
        (0.9, target),
    ]

    hits = rrf_fusion(hits_1, hits_2, [])

    assert len(hits) == 3
    assert hits[0][1].source == "target"
