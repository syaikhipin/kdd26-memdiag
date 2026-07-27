from __future__ import annotations
import time
import numpy as np
from memory_core import MemoryProvider, MemoryRecord, RetrievedItem


class VerbatimProvider(MemoryProvider):
    name = "verbatim"

    def __init__(self, embedder, llm, recent_buffer: int = 64, **opts) -> None:
        super().__init__(embedder, llm, **opts)
        self.recent_buffer = recent_buffer
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._contents: list[str] = []
        self._meta: list[dict] = []

    def ingest(self, records: list[MemoryRecord]) -> None:
        for r in records:
            if not r.content.strip():
                continue
            self._ids.append(r.id)
            self._vecs.append(self.embedder.embed(r.content))
            self._contents.append(r.content)
            self._meta.append(dict(r.metadata))

    def _matrix(self) -> np.ndarray | None:
        if not self._vecs:
            return None
        return np.vstack(self._vecs)

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[RetrievedItem], float]:
        t0 = time.perf_counter()
        matrix = self._matrix()
        if matrix is None:
            return [], 0.0
        q = self.embedder.embed(query)
        scores = matrix @ q
        order = np.argsort(-scores)[:top_k]
        out = [
            RetrievedItem(
                record_id=self._ids[int(i)],
                record_ids=[self._ids[int(i)]],
                content=self._contents[int(i)],
                score=float(scores[int(i)]),
            )
            for i in order
        ]
        return out, (time.perf_counter() - t0) * 1000.0

    def size(self) -> int:
        return len(self._ids)
