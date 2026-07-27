from __future__ import annotations
from memory_core import MemoryProvider, MemoryRecord, RetrievedItem


class NoMemoryProvider(MemoryProvider):
    name = "no_memory"

    def ingest(self, records: list[MemoryRecord]) -> None:
        return None

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[RetrievedItem], float]:
        return [], 0.0
