from __future__ import annotations
import time
from memory_core import MemoryProvider, MemoryRecord, RetrievedItem


class ExtractedFactsProvider(MemoryProvider):
    name = "extracted_facts"
    sim_threshold = 0.85

    def __init__(self, embedder, llm, max_facts_per_record: int = 6, **opts) -> None:
        super().__init__(embedder, llm, **opts)
        self.max_facts_per_record = max_facts_per_record
        self._facts: list[dict] = []

    def ingest(self, records: list[MemoryRecord]) -> None:
        for r in records:
            if not r.content.strip():
                continue
            facts = self.llm.extract_facts(r.content)[: self.max_facts_per_record]
            for fact_text in facts:
                if not fact_text.strip():
                    continue
                vec = self.embedder.embed(fact_text)
                merged = False
                for f in self._facts:
                    if f["source_id"] == r.id and self.embedder.cosine(vec, f["vec"]) >= self.sim_threshold:
                        f["mention"] += 1
                        f["confidence"] = min(1.0, f["confidence"] + 0.05)
                        merged = True
                        break
                if not merged:
                    self._facts.append({
                        "content": fact_text.strip(), "vec": vec,
                        "source_id": r.id, "mention": 1, "confidence": 0.8,
                    })

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[RetrievedItem], float]:
        t0 = time.perf_counter()
        if not self._facts:
            return [], 0.0
        q = self.embedder.embed(query)
        scored = sorted(
            ((self.embedder.cosine(q, f["vec"]), f) for f in self._facts),
            key=lambda x: -x[0],
        )[:top_k]
        out = [
            RetrievedItem(
                record_id=f["source_id"], record_ids=[f["source_id"]],
                content=f["content"], score=float(s),
                debug={"mention": f["mention"], "confidence": f["confidence"]},
            )
            for s, f in scored
        ]
        return out, (time.perf_counter() - t0) * 1000.0

    def size(self) -> int:
        return len(self._facts)
