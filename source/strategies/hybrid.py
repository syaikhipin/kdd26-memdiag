from models import Episode, ResearchTask
from .base import BaseMemoryStrategy


class HybridStrategy(BaseMemoryStrategy):
    name = "hybrid"

    def retrieve(self, query: str, task: ResearchTask, top_k: int):
        retrieved, latency_ms = self.store.retrieve(query, top_k=top_k * 3)
        best_by_action = {}
        for item in retrieved:
            action = item["entry"].metadata.get("action")
            if action not in best_by_action:
                best_by_action[action] = item
        return list(best_by_action.values())[:top_k], latency_ms

    def ingest_episode(self, task: ResearchTask, episode: Episode) -> list[str]:
        ids = []
        exact = (
            f"Exact result: task={task.task_id}; action='{episode.proposed_action}'; "
            f"outcome={episode.outcome_label}; score={episode.outcome_score:.3f}; "
            f"failure={episode.failure_mode or 'none'}."
        )
        ids.append(self.store.add(
            exact,
            metadata={"action": episode.proposed_action, "outcome_label": episode.outcome_label, "score": episode.outcome_score, "failure_mode": episode.failure_mode},
            source_episode=episode.episode_id,
            source_task=task.task_id,
            entry_type="verbatim",
        ))
        if episode.failure_mode:
            fact = f"Avoid '{episode.proposed_action}' for {task.domain}; observed {episode.failure_mode}."
            entry_type = "failure"
        else:
            fact = f"Decision rule for {task.domain}: '{episode.proposed_action}' produced {episode.outcome_label} result."
            entry_type = "fact"
        ids.append(self.store.add(
            fact,
            metadata={"action": episode.proposed_action, "outcome_label": episode.outcome_label, "score": episode.outcome_score, "failure_mode": episode.failure_mode},
            source_episode=episode.episode_id,
            source_task=task.task_id,
            entry_type=entry_type,
        ))
        summary = f"Episode summary: {task.task_id} explored {episode.proposed_action} with {episode.outcome_label} outcome."
        ids.append(self.store.add(
            summary,
            metadata={"action": episode.proposed_action, "outcome_label": episode.outcome_label, "score": episode.outcome_score, "failure_mode": episode.failure_mode},
            source_episode=episode.episode_id,
            source_task=task.task_id,
            entry_type="episode_summary",
        ))
        return ids
