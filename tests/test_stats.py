from src.text_tool import calc_stats, upper_lower_text


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


def test_upper_lower_text():
    assert upper_lower_text("a b", True, True) == "A B"
    assert upper_lower_text("A b", True, False) == "a b"
    assert upper_lower_text("A B", False, False) == "A B"


def test_calc_stats_non_empty_lines():
    raw = "a\n\n \nb"
    normalized = "a b"
    stats = calc_stats(raw, normalized)
    assert stats["non_empty_lines"] == 2
