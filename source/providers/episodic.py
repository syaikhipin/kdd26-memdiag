from __future__ import annotations
import time
from memory_core import MemoryProvider, MemoryRecord, RetrievedItem


class EpisodicProvider(MemoryProvider):
    name = "episodic"
    sem_weight = 0.7
    rec_weight = 0.3

    def __init__(self, embedder, llm, **opts) -> None:
        super().__init__(embedder, llm, **opts)
        self._episodes: list[dict] = []

    def ingest(self, records: list[MemoryRecord]) -> None:
        groups: dict[str, list[MemoryRecord]] = {}
        order: list[str] = []
        for r in records:
            if not r.content.strip():
                continue
            key = str(
                r.metadata.get("session_id")
                or r.metadata.get("source_task")
                or r.metadata.get("source_id")
                or r.id
            )
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(r)
        for rank, key in enumerate(order):
            recs = groups[key]
            text = "\n".join(r.content for r in recs)
            summary = self.llm.summarize(text) or text[:400]
            vec = self.embedder.embed(summary)
            tags = self.llm.extract_topic_tags(text)
            self._episodes.append({
                "episode_id": key, "source_ids": [r.id for r in recs],
                "summary": summary, "vec": vec, "tags": tags, "rank": rank,
            })

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[RetrievedItem], float]:
        t0 = time.perf_counter()
        if not self._episodes:
            return [], 0.0
        q = self.embedder.embed(query)
        n = len(self._episodes)
        denom = max(1, n - 1)
        scored = []
        for ep in self._episodes:
            sem = self.embedder.cosine(q, ep["vec"])
            recency = ep["rank"] / denom
            scored.append((self.sem_weight * sem + self.rec_weight * recency, ep))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, ep in scored[:top_k]:
            src = ep["source_ids"]
            primary = src[0] if src else ep["episode_id"]
            out.append(RetrievedItem(
                record_id=primary, record_ids=list(src),
                content=ep["summary"], score=float(score),
                tier="episodic", debug={"tags": ep["tags"]},
            ))
        return out, (time.perf_counter() - t0) * 1000.0

    def size(self) -> int:
        return len(self._episodes)
