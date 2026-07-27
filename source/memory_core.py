"""Core types for the memory-provider framework."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from embedder import Embedder
    from llm_client import LLMClient


@dataclass
class MemoryRecord:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_id: str | None = None

    def resolved_evidence_id(self) -> str:
        return self.evidence_id if self.evidence_id is not None else self.id


@dataclass
class RetrievedItem:
    record_id: str
    record_ids: list[str]
    content: str
    score: float
    tier: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


class MemoryProvider(ABC):
    name: str = "base"

    def __init__(self, embedder: "Embedder", llm: "LLMClient", **opts: Any) -> None:
        self.embedder = embedder
        self.llm = llm
        self.opts = opts

    @abstractmethod
    def ingest(self, records: list[MemoryRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[RetrievedItem], float]:
        raise NotImplementedError

    def size(self) -> int:
        return 0

    def export(self) -> dict[str, Any]:
        return {"strategy": self.name, "size": self.size()}
