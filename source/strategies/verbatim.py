from models import Episode, ResearchTask
from .base import BaseMemoryStrategy


class VerbatimStrategy(BaseMemoryStrategy):
    name = "verbatim"

    def ingest_episode(self, task: ResearchTask, episode: Episode) -> list[str]:
        content = (
            f"Task {task.task_id} ({task.domain}) episode {episode.step_idx}: "
            f"tried '{episode.proposed_action}'. Rationale: {episode.rationale}. "
            f"Outcome: {episode.outcome_label}, score={episode.outcome_score:.3f}, "
            f"failure={episode.failure_mode or 'none'}."
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
                entry_type="verbatim",
            )
        ]
