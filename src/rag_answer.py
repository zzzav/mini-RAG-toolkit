import re
from dataclasses import dataclass
from typing import Literal

from src.query_normalize import DEFAULT_STOP_WORDS, normalize_query

NO_INFO_IN_CONTEXT = "В контексте нет информации."

g_base_prompt: str = f"""
SYSTEM:
Ты помощник. Отвечай только на основе CONTEXT. Если в CONTEXT нет ответа — скажи:
"{NO_INFO_IN_CONTEXT}"

USER:
Вопрос: {{question}}

CONTEXT:
{{context}}

Требования к ответу:
- Коротко и по делу (3–8 предложений).
- Если есть численные данные/сроки/факты — приведи их.
- Если ответа нет в контексте — явно так и напиши."""

ALLOWED_LLM = {"mock", "extract", "none"}

RETRIEVER_TYPES = {"vector", "bm25"}


@dataclass
class Chunk:
    source: str
    idx: int
    text_preview: str
    score: float


@dataclass
class RAGConfig:
    top_k: int = 5
    max_context_chars: int = 4000
    per_chunk_chars: int = 800
    separator: str = "\n---\n"
    min_score: float | None = None


@dataclass
class RAGResult:
    query: str
    chunks: list[dict]
    context: str
    prompt: str
    answer: str | None
    citations: list[dict]


###############################################################
# hit = {
#   "source": str,
#   "idx": int,
#   "score": float,
#   "text": str
# }
###############################################################
def build_context(hits: list[dict], cfg: RAGConfig) -> str:
    context: str = ""

    for hit in hits:
        if cfg.min_score is not None and hit["score"] < cfg.min_score:
            continue

        if context != "":
            context += cfg.separator

        header = "SOURCE=" + hit["source"] + " IDX=" + str(hit["idx"]) + "\n"
        text = hit["text"][: cfg.per_chunk_chars]

        if len(context + header + text) <= cfg.max_context_chars:
            context += header + text
        else:
            break

    context = context[: cfg.max_context_chars]

    return context


def build_prompt(question: str, context: str) -> str:
    prompt = g_base_prompt.replace("{question}", question).replace("{context}", context)

    return prompt


def get_hits_from_vector_index_search(
    search_results: list[tuple[float, Chunk]],
) -> list[dict]:
    hits: list[dict] = []
    for score, chunk in search_results:
        hits.append(
            {
                "source": chunk.source,
                "idx": int(chunk.idx),
                "score": float(score),
                "text": chunk.text,
            }
        )
    return hits


def rag_answer(
    question: str,
    results: list[tuple[float, Chunk]],
    cfg: RAGConfig,
    *,
    llm: Literal["mock", "extract", "none"] = "mock",
) -> RAGResult:

    hits = get_hits_from_vector_index_search(results)
    context = build_context(hits, cfg)

    if llm not in ALLOWED_LLM:
        raise ValueError("Недопоустимое значение llm")

    if llm == "none":
        prompt = ""
        answer = None
    elif llm == "mock":
        prompt = build_prompt(question, context)
        answer = MockLLM().generate(prompt)
    elif llm == "extract":
        prompt = build_prompt(question, context)
        answer = ExtractLLM().generate(prompt)

    citations = collect_citations(hits, max_items=cfg.top_k)
    chunks = [
        {
            "source": h["source"],
            "idx": int(h["idx"]),
            "score": float(h["score"]),
            "text_preview": h["text"][:200],
        }
        for h in hits
    ]

    return RAGResult(
        query=question,
        chunks=chunks,
        context=context,
        prompt=prompt,
        answer=answer,
        citations=citations,
    )


def collect_citations(hits: list[dict], *, max_items: int | None = None) -> list[dict]:

    citations: list[dict] = []

    for hit in hits:
        if not any(
            c.get("source") == hit["source"] and c.get("idx") == hit["idx"] for c in citations
        ):
            citations.append({"source": hit["source"], "idx": hit["idx"]})
            if max_items:
                if len(citations) == max_items:
                    break

    return citations


@dataclass
class MockLLM:
    def generate(self, prompt: str) -> str:
        start_str = "CONTEXT:\n"
        end_str = "\nТребования к ответу:"
        context: str = ""

        if start_str in prompt:
            context = prompt.split(start_str, 1)[1].split(end_str, 1)[0]

        answer = ""
        if context.strip() == "":
            answer = NO_INFO_IN_CONTEXT
        else:
            answer = "Ответ:" + context[:200]

        return answer


@dataclass
class ExtractLLM:
    def generate(self, prompt: str) -> str:
        no_context_str = NO_INFO_IN_CONTEXT
        if prompt == "":
            return no_context_str

        # обрабатываем запрос
        start_str = "Вопрос: "
        end_str = "\nCONTEXT:"
        q: str = ""

        if start_str in prompt:
            q = prompt.split(start_str, 1)[1].split(end_str, 1)[0]

        if q == "":
            return "Отсутсвует запрос."

        q_normalized = normalize_query(q, stop_words=DEFAULT_STOP_WORDS)

        # обрабатываем контекст - делим на предложения
        start_str = "CONTEXT:\n"
        end_str = "\nТребования к ответу:"
        context: str = ""

        if start_str in prompt:
            context = prompt.split(start_str, 1)[1].split(end_str, 1)[0]

        context_lines = [line.strip() for line in re.split(r"[.?!\n]+", context) if line.strip()]

        scored_context_lines: list[tuple[int, int, str]] = []
        seen_lines: set[str] = set()
        score_sum: int = 0
        for pos, line in enumerate(context_lines):

            line_key = " ".join(normalize_query(line, stop_words=DEFAULT_STOP_WORDS))
            if line_key in seen_lines:
                continue
            seen_lines.add(line_key)

            line_tokens = set(normalize_query(line, stop_words=DEFAULT_STOP_WORDS))
            line_score = sum(1 for w in q_normalized if w in line_tokens)
            score_sum += line_score

            scored_context_lines.append((line_score, pos, line))

        if score_sum == 0:
            return no_context_str

        scored_context_lines.sort(key=lambda x: (-x[0], x[1]))

        return ". ".join(line for _, _, line in scored_context_lines[:3])
