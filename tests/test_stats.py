from src.text_tool import calc_stats


def test_calc_stats_basic():
    raw = "Hello World"
    normalized = "hello world"
    stats = calc_stats(raw, normalized)
    assert stats["words"] == 2
    assert stats["lines"] == 1


def test_calc_stats_multilines():
    raw = "one\ntwo\nthree"
    normalized = "one two three"
    stats = calc_stats(raw, normalized)
    assert stats["lines"] == 3
    assert stats["words"] == 3
