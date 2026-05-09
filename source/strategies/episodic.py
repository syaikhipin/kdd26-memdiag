from models import Episode, ResearchTask
from .base import BaseMemoryStrategy


class EpisodicStrategy(BaseMemoryStrategy):
    name = "episodic"

    def ingest_episode(self, task: ResearchTask, episode: Episode) -> list[str]:
        trend = "failed" if episode.failure_mode else episode.outcome_label
        content = (
            f"Research episode summary for {task.domain}: action '{episode.proposed_action}' {trend}. "
            f"Remember this trend for future {task.task_id} decisions."
        )
        return [
            self.store.add(
                content,
                metadata={
                    "action": episode.proposed_action,
                    "outcome_label": episode.outcome_label,
                    "score": episode.outcome_score,
                    "failure_mode": episode.failure_mode,
                },
                source_episode=episode.episode_id,
                source_task=task.task_id,
                entry_type="episode_summary",
            )
        ]
