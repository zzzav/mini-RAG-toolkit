from src.vector_search import (
    build_vector_index_by_chunks,
    build_chunks,
    search,
)


def test_top_1():
    docs = [
        ("a.txt", "This invoice is overdue"),
        ("b.txt", "The weather is nice"),
    ]

    chunks = build_chunks(docs, chunk_size=400, overlap=80)
    index = build_vector_index_by_chunks(chunks)

    results = search("bill", index, top_k=1, use_synonyms=False)
    results_with_synonyms = search("bill", index, top_k=1, use_synonyms=True)

    assert len(results) == 0
    assert len(results_with_synonyms) == 1
    assert results_with_synonyms[0][1].source == "a.txt"
