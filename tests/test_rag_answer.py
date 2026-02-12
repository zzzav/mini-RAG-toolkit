from src.rag_answer import MockLLM, RAGConfig, build_context, build_prompt, g_base_prompt
from src.rag_cli import get_hits_from_vector_index_search
from src.vector_search import build_vector_index, search


def test_build_context_chunks_limits():
    cfg = RAGConfig(max_context_chars=200, per_chunk_chars=5)
    hits = [
        {"source": "a", "idx": 0, "score": 1, "text": "1234567890"},
        {"source": "b", "idx": 0, "score": 1, "text": "0123456789"},
        {"source": "c", "idx": 0, "score": 1, "text": "abcdefghij"},
    ]

    context = build_context(hits, cfg)

    # 1) Проверка: нигде не встречается кусок длиной N+1
    for hit in hits:
        assert hit["text"][: cfg.per_chunk_chars] in context
        assert hit["text"][: cfg.per_chunk_chars + 1] not in context


def test_build_context_max_limits():
    cfg = RAGConfig(max_context_chars=20, per_chunk_chars=5)
    hits = [
        {"source": "a", "idx": 0, "score": 1, "text": "1234567890"},
        {"source": "b", "idx": 0, "score": 1, "text": "0123456789"},
        {"source": "c", "idx": 0, "score": 1, "text": "abcdefghij"},
    ]

    context = build_context(hits, cfg)

    # 2) max_context_chars
    assert len(context) <= cfg.max_context_chars

    # 3) заголовок первого чанка
    base_first_header = f'SOURCE={hits[0]["source"]} IDX={hits[0]["idx"]}'
    first_header = context.split(cfg.separator)[0].splitlines()[0]
    assert base_first_header == first_header


def test_build_prompt():
    context = "test_context"
    question = "test_q"

    prompt = build_prompt(question, context)

    assert context in prompt
    assert question in prompt


def test_mock_llm():
    mock_llm = MockLLM()

    request_1 = mock_llm.generate(g_base_prompt)
    request_2 = mock_llm.generate(g_base_prompt)

    assert request_1 == request_2


def test_end_to_end():
    cfg = RAGConfig()
    q = "invoice payment"
    vector_index = build_vector_index("./docs", 400, 80)
    results = search(q, vector_index, 2)
    hits = get_hits_from_vector_index_search(results)
    context = build_context(hits, cfg)
    prompt = build_prompt(q, context)
    mock_llm = MockLLM()
    answer = mock_llm.generate(prompt)

    assert len(hits) >= 1
    assert "invoice" in answer.lower()


def test_no_results():
    cfg = RAGConfig()
    q = ""
    vector_index = build_vector_index("./docs", 400, 80)
    results = search(q, vector_index, 2)
    hits = get_hits_from_vector_index_search(results)
    context = build_context(hits, cfg)
    prompt = build_prompt(q, context)
    mock_llm = MockLLM()
    answer = mock_llm.generate(prompt)

    assert hits == []
    assert context == ""
    assert answer == "В контексте нет информации."
