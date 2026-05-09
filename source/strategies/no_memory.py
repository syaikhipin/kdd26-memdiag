from models import Episode, ResearchTask
from .base import BaseMemoryStrategy


class NoMemoryStrategy(BaseMemoryStrategy):
    name = "no_memory"

    def retrieve(self, query: str, task: ResearchTask, top_k: int):
        return [], 0.0

    def ingest_episode(self, task: ResearchTask, episode: Episode) -> list[str]:
        return []
