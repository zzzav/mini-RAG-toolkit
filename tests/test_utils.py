from src.utils import normalize_text


def test_normalize_text():
    assert normalize_text("  hello   world  ") == "hello world"
