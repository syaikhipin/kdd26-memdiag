import math
import re
import time
import uuid
from collections import Counter
from typing import Any

from models import MemoryEntry

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_overlap(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[token] * b.get(token, 0) for token in a)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class MemoryStore:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.entries: list[MemoryEntry] = []
        self._vectors: dict[str, Counter[str]] = {}

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        source_episode: str | None = None,
        source_task: str | None = None,
        entry_type: str = "memory",
    ) -> str:
        entry_id = str(uuid.uuid4())[:8]
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata or {},
            source_episode=source_episode,
            source_task=source_task,
            strategy=self.strategy,
            entry_type=entry_type,
        )
        self.entries.append(entry)
        self._vectors[entry_id] = Counter(tokenize(content))
        return entry_id

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[dict[str, Any]], float]:
        start = time.perf_counter()
        query_vector = Counter(tokenize(query))
        scored = []
        for entry in self.entries:
            score = cosine_overlap(query_vector, self._vectors[entry.id])
            if score > 0:
                scored.append({"entry": entry, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        latency_ms = (time.perf_counter() - start) * 1000
        return scored[:top_k], latency_ms

    def relevant_entries(self, task_id: str, action: str | None = None) -> list[MemoryEntry]:
        relevant = []
        for entry in self.entries:
            if entry.source_task != task_id:
                continue
            if action is None or entry.metadata.get("action") == action:
                relevant.append(entry)
        return relevant

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "entries": [
                {
                    "id": entry.id,
                    "content": entry.content,
                    "metadata": entry.metadata,
                    "source_episode": entry.source_episode,
                    "source_task": entry.source_task,
                    "strategy": entry.strategy,
                    "entry_type": entry.entry_type,
                }
                for entry in self.entries
            ],
        }

    def size(self) -> int:
        return len(self.entries)
