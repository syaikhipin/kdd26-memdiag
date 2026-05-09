from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    domain: str
    prompt: str
    constraints: list[str]
    success_criteria: str
    useful_actions: dict[str, float]
    failure_actions: dict[str, str]
    action_keywords: dict[str, list[str]]


@dataclass
class Episode:
    episode_id: str
    task_id: str
    step_idx: int
    query: str
    proposed_action: str
    rationale: str
    referenced_memory_ids: list[str]
    outcome_score: float
    outcome_label: str
    cost_units: float
    latency_ms: float
    failure_mode: str | None
    redundant_action: bool
    relevant_memory_ids: list[str] = field(default_factory=list)
    retrieved_memory_ids: list[str] = field(default_factory=list)
    retrieved_texts: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)


@dataclass
class MemoryEntry:
    id: str
    content: str
    metadata: dict[str, Any]
    source_episode: str | None
    source_task: str | None
    strategy: str
    entry_type: str


@dataclass
class RetrievalTrace:
    query: str
    retrieved_ids: list[str]
    retrieved_texts: list[str]
    scores: list[float]
    gold_relevant_ids: list[str]
    used_ids: list[str]
    latency_ms: float
