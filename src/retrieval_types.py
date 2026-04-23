from dataclasses import dataclass

OUTPUT_FORMATS = ("text", "json")
ALLOWED_LLM = ("mock", "extract", "none")
RETRIEVER_TYPES = ("vector", "bm25", "fusion")
INDEX_TYPES = ("vector", "bm25")
FUSION_METHODS = ("rrf", "weighted")


@dataclass
class Chunk:
    source: str
    idx: int
    text: str


@dataclass
class Filters:
    source_items: list[str] | None = None
    ext_items: list[str] | None = None
    source_contains_items: list[str] | None = None
