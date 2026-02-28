from src.rag_answer import ExtractLLM, g_base_prompt


def test_extract_empty_context():
    prompt = g_base_prompt.replace("{context}", "")
    assert ExtractLLM().generate(prompt) == "В контексте нет информации."


def test_extract_picks_relevant_sentences():
    q = "invoice payment"
    context = "First sentence!\n'Invoice payment' is in second sentence. Sentence number three."

    prompt = g_base_prompt.replace("{context}", context).replace("{question}", q)

    assert "'Invoice payment' is in second sentence" in ExtractLLM().generate(prompt)


def test_extract_no_overlap():
    q = "invoice payment"
    context = "First sentence!\nIt is second sentence. Sentence number three."

    prompt = g_base_prompt.replace("{context}", context).replace("{question}", q)

    assert ExtractLLM().generate(prompt) == "В контексте нет информации."
