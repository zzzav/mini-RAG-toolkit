from dataclasses import dataclass


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
