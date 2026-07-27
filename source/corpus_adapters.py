"""Corpus adapters: convert every corpus shape into the universal MemoryRecord list."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from memory_core import MemoryRecord

if TYPE_CHECKING:
    from data_loaders import Conversation, MemoryBenchmarkItem
    from models import Episode, ResearchTask


def from_loader_record(rec: dict[str, Any]) -> MemoryRecord:
    meta = dict(rec.get("metadata") or {})
    rid = str(rec.get("id") or meta.get("record_id") or meta.get("dia_id") or "")
    evidence_id = str(
        meta.get("evidence_id") or meta.get("record_id") or meta.get("dia_id") or rid
    )
    return MemoryRecord(id=rid, content=str(rec.get("content", "")), metadata=meta, evidence_id=evidence_id)


def locomo_records(conversation: "Conversation") -> list[MemoryRecord]:
    from data_loaders import locomo_memory_records
    return [from_loader_record(r) for r in locomo_memory_records(conversation)]


def benchmark_item_records(item: "MemoryBenchmarkItem") -> list[MemoryRecord]:
    return [from_loader_record(r) for r in item.memory_records]


def episode_records(task: "ResearchTask", episode: "Episode") -> list[MemoryRecord]:
    base = {
        "task_id": task.task_id, "domain": task.domain,
        "action": episode.proposed_action, "outcome_label": episode.outcome_label,
        "score": episode.outcome_score, "failure_mode": episode.failure_mode,
    }
    records: list[MemoryRecord] = []
    if episode.failure_mode:
        content = (
            f"Avoid '{episode.proposed_action}' for {task.domain} when constraints include "
            f"{', '.join(task.constraints)}: it caused {episode.failure_mode}."
        )
        records.append(MemoryRecord(
            id=f"{episode.episode_id}:failure", content=content,
            metadata={**base, "entry_type": "failure"},
            evidence_id=f"{episode.episode_id}:failure",
        ))
    else:
        label = "improved" if episode.outcome_label == "improved" else "neutral"
        content = (
            f"'{episode.proposed_action}' gave {label} result (score {episode.outcome_score:.3f}) "
            f"for {task.domain} under {', '.join(task.constraints)}."
        )
        records.append(MemoryRecord(
            id=f"{episode.episode_id}:fact", content=content,
            metadata={**base, "entry_type": "fact"},
            evidence_id=f"{episode.episode_id}:fact",
        ))
    return records


def trial_records(trial: dict[str, Any]) -> list[MemoryRecord]:
    cfg = trial.get("config", {}) or {}
    tid = str(trial.get("trial_id") or "")
    final = trial.get("final_val_accuracy", trial.get("final_val_bpb"))
    content = (
        f"Experiment {tid} on {trial.get('dataset', 'unknown')}: "
        f"config {{{', '.join(f'{k}={v}' for k, v in cfg.items())}}} -> "
        f"final_val_accuracy={final}, outcome={trial.get('outcome_label')}"
        + (f", failure={trial.get('failure_mode')}" if trial.get("failure_mode") else "")
        + "."
    )
    records = [MemoryRecord(
        id=f"{tid}:config", content=content,
        metadata={"trial_id": tid, "dataset": trial.get("dataset"), "entry_type": "experiment_config",
                  **{f"cfg_{k}": v for k, v in cfg.items()}},
        evidence_id=f"{tid}:config",
    )]
    curve = trial.get("val_accuracy_curve") or trial.get("val_bpb_curve") or []
    metric_key = "val_accuracy" if "val_accuracy_curve" in trial else "val_bpb"
    for point in curve:
        budget = point.get("budget")
        value = point.get(metric_key) or point.get("val_accuracy") or point.get("val_bpb")
        if budget is None or value is None:
            continue
        records.append(MemoryRecord(
            id=f"{tid}:budget:{budget}",
            content=f"At budget {budget}, {tid} reached {metric_key}={value}.",
            metadata={"trial_id": tid, "budget": budget, metric_key: value, "entry_type": "learning_curve"},
            evidence_id=f"{tid}:budget:{budget}",
        ))
    return records
