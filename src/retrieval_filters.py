from pathlib import Path

from src.retrieval_types import Chunk, Filters


def filter_hits(hits: list[tuple[float, Chunk]], filters: Filters) -> list[tuple[float, Chunk]]:
    fin_hits: list[tuple[float, Chunk]] = []

    def filter_by_item(items: list[str] | None, item: str, contain: bool = False) -> bool:
        res = True
        if items is not None:
            res = False
            for i in items:
                if (not contain and item.lower() != i.lower()) or (
                    contain and i.lower() not in item.lower()
                ):
                    continue
                res = True
                break
        return res

    for hit in hits:
        ext = Path(hit[1].source).suffix

        if not filter_by_item(filters.source_items, hit[1].source):
            continue

        if not filter_by_item(filters.ext_items, ext):
            continue

        if not filter_by_item(filters.source_contains_items, hit[1].source, True):
            continue

        fin_hits.append(hit)

    return fin_hits
