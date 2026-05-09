from abc import ABC, abstractmethod
from typing import Any

from memory_store import MemoryStore
from models import Episode, ResearchTask


class BaseMemoryStrategy(ABC):
    name = "base"

    def __init__(self):
        self.store = MemoryStore(self.name)

    def retrieve(self, query: str, task: ResearchTask, top_k: int) -> tuple[list[dict[str, Any]], float]:
        return self.store.retrieve(query, top_k=top_k)

    @abstractmethod
    def ingest_episode(self, task: ResearchTask, episode: Episode) -> list[str]:
        raise NotImplementedError

    def memory_size(self) -> int:
        return self.store.size()

    def export_memory(self) -> dict[str, Any]:
        return self.store.to_dict()
