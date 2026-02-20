from src.query_normalize import normalize_query


def test_query_normalize():
    assert normalize_query("Invoice, payment  ! ") == ["invoice", "payment"]
    assert normalize_query("Invoice, payment, test!", stop_words={"test"}) == ["invoice", "payment"]
    assert normalize_query("test,  test!", stop_words={"test"}) == []
