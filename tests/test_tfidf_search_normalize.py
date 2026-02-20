from src.tfidf_search import tfidf_search, Chunk, build_index


def test_using_stop_words():
    chunks = [
        Chunk(
            source="a", idx=0, text="Invoice #123 was issued to the client. Payment due in 7 days."
        ),
        Chunk(source="b", idx=0, text="apple banana"),
    ]
    index = build_index(chunks)

    results_1 = tfidf_search(
        query="Invoice, payment!", chunks=chunks, index=index, top_k=1, use_stop_words=True
    )
    results_2 = tfidf_search(
        query="Invoice payment", chunks=chunks, index=index, top_k=1, use_stop_words=True
    )

    assert results_1 == results_2
