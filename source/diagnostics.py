from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from memory_store import tokenize


def precision_recall(retrieved_ids: list[str], relevant_ids: list[str]) -> dict[str, float | int | bool]:
    retrieved = [str(item) for item in retrieved_ids]
    relevant = {str(item) for item in relevant_ids}
    hits = sum(1 for item in retrieved if item in relevant)
    return {
        "precision": hits / len(retrieved) if retrieved else 0.0,
        "recall": hits / len(relevant) if relevant else 0.0,
        "hits": hits,
        "n_retrieved": len(retrieved),
        "n_relevant": len(relevant),
        "evidence_hit": hits > 0,
    }


def locomo_retrieval_diagnostics(retrieved: list[dict[str, Any]], evidence_ids: list[str]) -> dict[str, Any]:
    retrieved_dia_ids = [str(item["entry"].metadata.get("dia_id", "")) for item in retrieved]
    metrics = precision_recall(retrieved_dia_ids, evidence_ids)
    if not retrieved:
        failure = "no_memory_available"
    elif metrics["hits"] == 0 and evidence_ids:
        failure = "retrieval_miss"
    elif metrics["recall"] < 1.0 and evidence_ids:
        failure = "partial_evidence_retrieved"
    else:
        failure = "none"
    metrics["failure_category"] = failure
    return metrics


def locomo_utilization_category(evidence_hit: bool, answer_mode: str) -> str:
    if evidence_hit and answer_mode == "offline_evidence_heuristic":
        return "evidence_available_for_answering"
    if evidence_hit:
        return "memory_available_to_llm"
    return "retrieval_not_useful"


def novelty_score(text: str, previous_texts: list[str]) -> float:
    current = set(tokenize(text))
    if not current or not previous_texts:
        return 1.0
    overlaps = []
    for previous in previous_texts:
        prior = set(tokenize(previous))
        if prior:
            overlaps.append(len(current & prior) / len(current | prior))
    return round(1.0 - max(overlaps, default=0.0), 4)


def count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(record.get(key, "unknown")) for record in records)
    return dict(sorted(counts.items()))


def mean_field(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0)) for record in records]
    return round(mean(values), 4) if values else 0.0


def group_records(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record.get(key, "unknown"))].append(record)
    return dict(grouped)
