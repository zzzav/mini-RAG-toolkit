from dataclasses import dataclass
from typing import Literal

g_base_prompt: str = """
SYSTEM:
Ты помощник. Отвечай только на основе CONTEXT. Если в CONTEXT нет ответа — скажи:
"В контексте нет информации."

USER:
Вопрос: {question}

CONTEXT:
{context}

Требования к ответу:
- Коротко и по делу (3–8 предложений).
- Если есть численные данные/сроки/факты — приведи их.
- Если ответа нет в контексте — явно так и напиши."""


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


def rag_answer(
    question: str,
    hits: list[dict],
    cfg: RAGConfig,
    *,
    llm: Literal["mock", "none"] = "mock",
) -> RAGResult:

    context = build_context(hits, cfg)
    prompt = "" if llm == "none" else build_prompt(question, context)
    answer = None if llm == "none" else MockLLM().generate(prompt)
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
            answer = "В контексте нет информации."
        else:
            answer = "Ответ:" + context[:200]

        return answer
