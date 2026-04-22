from src.retrieval_filters import filter_hits
from src.retrieval_types import Chunk, Filters


def test_filter_hits_by_source() -> None:
    hits = [(0.2, Chunk("x", 0, "text from x")), (0, Chunk("a", 0, "text from a"))]

    fin_hits = filter_hits(hits, Filters(source_items=["x"]))

    assert len(fin_hits) == 1
    assert fin_hits[0][1].source == "x"


def test_filter_hits_by_ext() -> None:
    hits = [(0.2, Chunk("x.txt", 0, "text from x")), (0, Chunk("a.md", 0, "text from a"))]

    fin_hits = filter_hits(hits, Filters(ext_items=[".txt"]))

    assert len(fin_hits) == 1
    assert fin_hits[0][1].source == "x.txt"


def test_filter_hits_by_source_contains() -> None:
    hits = [
        (0.2, Chunk("x_lesson_1.txt", 0, "text from x")),
        (0, Chunk("a_lesson_2.md", 0, "text from a")),
    ]

    fin_hits = filter_hits(hits, Filters(source_contains_items=["lesson_2"]))

    assert len(fin_hits) == 1
    assert fin_hits[0][1].source == "a_lesson_2.md"


def test_filter_hits_combines_conditions() -> None:
    hits = [
        (0.2, Chunk("x_lesson_1.txt", 0, "text from x")),
        (0.2, Chunk("y_lesson_1.md", 0, "text from y")),
        (0, Chunk("a_lesson_2.md", 0, "text from a")),
        (0, Chunk("b_lesson_2.md", 0, "text from b")),
    ]

    fin_hits = filter_hits(
        hits, Filters(source_contains_items=["lesson_2", "lesson_1"], ext_items=[".md"])
    )

    assert len(fin_hits) == 3
    assert fin_hits[0][1].source == "y_lesson_1.md"
    assert fin_hits[1][1].source == "a_lesson_2.md"
    assert fin_hits[2][1].source == "b_lesson_2.md"
