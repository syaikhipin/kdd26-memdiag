from __future__ import annotations
import time
from memory_core import MemoryProvider, MemoryRecord, RetrievedItem
from .verbatim import VerbatimProvider
from .episodic import EpisodicProvider

_RRF_K = 60


def _rrf_fuse(lists: list[list[RetrievedItem]], top_k: int) -> list[RetrievedItem]:
    bucket: dict[str, dict] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked):
            key = item.record_id
            entry = bucket.setdefault(key, {"item": item, "score": 0.0})
            entry["score"] += 1.0 / (_RRF_K + rank + 1)
            for rid in item.record_ids:
                if rid not in entry["item"].record_ids:
                    entry["item"].record_ids.append(rid)
    fused = sorted(bucket.values(), key=lambda e: -e["score"])[:top_k]
    for entry in fused:
        entry["item"].score = float(entry["score"])
        entry["item"].tier = "hybrid"
    return [e["item"] for e in fused]


class HierarchicalProvider(MemoryProvider):
    name = "hybrid"

    def __init__(self, embedder, llm, l1_capacity: int = 8, **opts) -> None:
        super().__init__(embedder, llm, **opts)
        self.l1_capacity = l1_capacity
        self._verbatim = VerbatimProvider(embedder, llm)
        self._episodic = EpisodicProvider(embedder, llm)
        self._l1: list[str] = []

    def ingest(self, records: list[MemoryRecord]) -> None:
        self._verbatim.ingest(records)
        self._episodic.ingest(records)

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[RetrievedItem], float]:
        t0 = time.perf_counter()
        vh, _ = self._verbatim.retrieve(query, top_k=top_k * 2)
        eh, _ = self._episodic.retrieve(query, top_k=top_k * 2)
        fused = _rrf_fuse([vh, eh], top_k)
        for item in fused:
            if item.record_id not in self._l1:
                self._l1.append(item.record_id)
        self._l1 = self._l1[-self.l1_capacity:]
        return fused, (time.perf_counter() - t0) * 1000.0

    def size(self) -> int:
        return self._verbatim.size() + self._episodic.size()
