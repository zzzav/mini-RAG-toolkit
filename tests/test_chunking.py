import pytest

from src.simple_search import chunk_text, score_chunk


def test_chunk_text():
    chunks = chunk_text("hello", 10, 3)
    assert len(chunks) == 1


def test_chunk_text_overlay():
    in_text = "abcdefghijklmnopqrstuvwxyz"
    chunk_size = 10
    overlap = 2
    chunks = chunk_text(in_text, chunk_size, overlap)
    assert len(chunks) == 4
    assert chunks[1][0] == in_text[chunk_size - overlap]
    # проверка оверлапа
    assert chunks[0][-overlap:] == chunks[1][:overlap]


def test_chunk_text_bad_params():
    with pytest.raises(ValueError):
        chunk_text("x", 0, 0)
    with pytest.raises(ValueError):
        chunk_text("x", 10, -1)
    with pytest.raises(ValueError):
        chunk_text("x", 10, 10)


def test_scoring():
    query = "the one"
    chunk1 = "the just one the the"
    score1 = score_chunk(query, chunk1, boost_window=10)

    chunk2 = "the just text text one the the"
    score2 = score_chunk(query, chunk2, boost_window=10)

    assert score1[0] == 2
    assert score2[0] == 1
