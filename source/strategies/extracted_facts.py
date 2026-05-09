from models import Episode, ResearchTask
from .base import BaseMemoryStrategy


class ExtractedFactsStrategy(BaseMemoryStrategy):
    name = "extracted_facts"

    def ingest_episode(self, task: ResearchTask, episode: Episode) -> list[str]:
        if episode.failure_mode:
            content = (
                f"Avoid '{episode.proposed_action}' for {task.domain} when constraints include "
                f"{', '.join(task.constraints)} because it caused {episode.failure_mode}."
            )
            entry_type = "failure"
        elif episode.outcome_label == "improved":
            content = (
                f"Use '{episode.proposed_action}' for {task.domain}; it improved score to "
                f"{episode.outcome_score:.3f} under {', '.join(task.constraints)}."
            )
            entry_type = "fact"
        else:
            content = (
                f"'{episode.proposed_action}' had neutral impact for {task.domain}; prefer stronger alternatives."
            )
            entry_type = "fact"
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
                entry_type=entry_type,
            )
        ]
