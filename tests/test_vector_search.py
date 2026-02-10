from src.vector_search import (
    build_chunks,
    build_vector_index_by_chunks,
    load_index,
    save_index,
    search,
)


def test_top_1():
    vector_path = ".\\tmp\\pytest.pk1"
    docs = [
        (
            "a.txt",
            (
                "Invoice #123 was issued to the client."
                "Payment due in 7 days.\nIf payment is late, "
                "send a reminder email."
            ),
        ),
        (
            "b.txt",
            (
                "Bank alerts:\nCard payment 10 EUR. Another payment 5 EUR."
                "\nIf suspicious activity happens, freeze the card."
            ),
        ),
        (
            "c.txt",
            (
                "Project notes:\nWe generate images for the content factory "
                "using prompt templates.\nThe pipeline is: "
                "prompt -> image -> asset store -> publish."
            ),
        ),
    ]

    chunks = build_chunks(docs, chunk_size=400, overlap=80)
    index = build_vector_index_by_chunks(chunks)
    results = search("payment invoice", index, top_k=1)

    save_index(vector_path, index)
    index_loaded = load_index(vector_path)
    results_by_loaded = search("payment invoice", index_loaded, top_k=1)

    assert results[0][1].source == "a.txt"
    assert results_by_loaded[0][1].source == "a.txt"
