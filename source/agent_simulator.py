import random
import time

from models import Episode, ResearchTask
from synthetic_data import all_actions, action_text


class ResearchEnvironment:
    def __init__(self, seed: int):
        self.random = random.Random(seed)
        self.best_scores: dict[str, float] = {}
        self.tried_actions: dict[str, set[str]] = {}

    def evaluate(self, task: ResearchTask, action: str) -> tuple[float, str, str | None, bool]:
        tried = self.tried_actions.setdefault(task.task_id, set())
        redundant = action in tried
        tried.add(action)
        current = self.best_scores.get(task.task_id, 0.45)
        failure_mode = task.failure_actions.get(action)
        if failure_mode:
            score = max(0.1, current - 0.04)
            return score, "failed", failure_mode, redundant
        gain = task.useful_actions.get(action, 0.0)
        if redundant:
            gain *= 0.15
        jitter = self.random.uniform(-0.01, 0.01)
        score = min(0.95, max(0.1, current + gain + jitter))
        label = "improved" if score > current + 0.01 else "neutral"
        self.best_scores[task.task_id] = max(current, score)
        return score, label, None, redundant


class SimulatedResearchAgent:
    def __init__(self, seed: int):
        self.random = random.Random(seed)
        self.planned_index: dict[str, int] = {}

    def choose_action(self, task: ResearchTask, retrieved: list[dict]) -> tuple[str, str, list[str]]:
        tried = {
            item["entry"].metadata.get("action")
            for item in retrieved
            if item["entry"].metadata.get("action")
        }
        avoided = {
            item["entry"].metadata.get("action")
            for item in retrieved
            if item["entry"].metadata.get("failure_mode")
        }
        evidence_ids = [item["entry"].id for item in retrieved if item["entry"].metadata.get("action") in tried]
        candidates = [action for action in all_actions(task) if action not in avoided and action not in tried]
        if not candidates:
            candidates = [action for action in all_actions(task) if action not in avoided]
        if not candidates:
            candidates = all_actions(task)

        useful = [action for action in task.useful_actions if action in candidates]
        if retrieved and useful:
            action = useful[0]
            used = evidence_ids[:1] or [retrieved[0]["entry"].id]
            return action, f"Use retrieved memories to avoid repeats and failures; try {action}.", used

        action = self.random.choice(candidates)
        return action, f"Explore {action} under {', '.join(task.constraints)}.", []


def build_query(task: ResearchTask, step_idx: int) -> str:
    return (
        f"Task {task.task_id}: {task.prompt} Step {step_idx}. "
        f"Need prior successful actions, failed actions to avoid, and constraints: {', '.join(task.constraints)}."
    )


def relevant_memory_ids(strategy, task: ResearchTask, retrieved_action: str | None = None) -> list[str]:
    entries = strategy.store.relevant_entries(task.task_id, retrieved_action)
    if not entries:
        entries = strategy.store.relevant_entries(task.task_id)
    return [entry.id for entry in entries]


def run_episode(strategy, agent: SimulatedResearchAgent, env: ResearchEnvironment, task: ResearchTask, step_idx: int, top_k: int) -> Episode:
    start = time.perf_counter()
    query = build_query(task, step_idx)
    retrieved, retrieval_latency_ms = strategy.retrieve(query, task, top_k=top_k)
    action, rationale, referenced_ids = agent.choose_action(task, retrieved)
    outcome_score, outcome_label, failure_mode, redundant = env.evaluate(task, action)
    latency_ms = retrieval_latency_ms + (time.perf_counter() - start) * 1000
    cost_units = 0.1 + 0.02 * len(retrieved) + 0.01 * strategy.memory_size()
    return Episode(
        episode_id=f"{task.task_id}-{step_idx}",
        task_id=task.task_id,
        step_idx=step_idx,
        query=query + " " + action_text(task, action),
        proposed_action=action,
        rationale=rationale,
        referenced_memory_ids=referenced_ids,
        outcome_score=outcome_score,
        outcome_label=outcome_label,
        cost_units=cost_units,
        latency_ms=latency_ms,
        failure_mode=failure_mode,
        redundant_action=redundant,
        relevant_memory_ids=relevant_memory_ids(strategy, task, action),
        retrieved_memory_ids=[item["entry"].id for item in retrieved],
        retrieved_texts=[item["entry"].content for item in retrieved],
        retrieval_scores=[item["score"] for item in retrieved],
    )
