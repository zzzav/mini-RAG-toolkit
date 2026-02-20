from src.synonyms import expand_tokens


def test_synonims_adding():
    in_tokens = ["invoice", "payment", "pay"]
    synonyms = {"invoice": ["bill", "bill_1"], "payment": ["pay"]}

    out_tokens = expand_tokens(tokens=in_tokens, synonyms=synonyms, max_expansions=1)

    assert out_tokens == ["invoice", "bill", "payment", "pay"]
    assert out_tokens[0] == "invoice" and out_tokens[1] == "bill"
    assert "bill_1" not in out_tokens
    assert len(out_tokens) == 4
