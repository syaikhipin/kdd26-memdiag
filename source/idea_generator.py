import json
from typing import Any

from diagnostics import novelty_score
from llm_client import OpenAICompatibleClient


OFFLINE_IDEAS = {
    "locomo": [
        {
            "title": "Evidence-first temporal memory debugger",
            "dataset": "LoCoMo",
            "hypothesis": "Separating evidence retrieval from answer generation exposes whether temporal QA failures are retrieval misses or utilization failures.",
            "method": "Store each dialogue turn with dia_id metadata, retrieve by question, and score evidence hit before asking any LLM to answer.",
            "memory_mechanism": "verbatim turn memory plus evidence-id diagnostics",
            "expected_metric_movement": "Higher evidence recall and clearer retrieval_miss vs retrieved_but_ignored attribution.",
        },
        {
            "title": "Failure-aware top-k calibration for conversational memory",
            "dataset": "LoCoMo",
            "hypothesis": "Top-k can be tuned by failure category rather than by answer F1 alone.",
            "method": "Run top-k sweeps and plot precision, recall, evidence hit, and partial evidence failures by QA category.",
            "memory_mechanism": "retrieval diagnostics over verbatim and hybrid memories",
            "expected_metric_movement": "Lower irrelevant retrieval without sacrificing temporal evidence recall.",
        },
    ],
    "autoresearch": [
        {
            "title": "Novelty-aware autoresearch experiment queue",
            "dataset": "autoresearch traces",
            "hypothesis": "A memory of failed and discarded experiments can reduce redundant train.py edits in overnight autonomous research.",
            "method": "Parse results.tsv/run.log, store action-outcome memories, and reject proposed changes that overlap prior crashes or regressions.",
            "memory_mechanism": "extracted failure facts plus episodic summaries",
            "expected_metric_movement": "Fewer repeated crashes and faster val_bpb convergence.",
        }
    ],
    "hpo": [
        {
            "title": "Memory-backed anti-redundancy HPO controller",
            "dataset": "HPOBench or LCBench",
            "hypothesis": "Persisting tried configurations and near-miss regions improves low-budget HPO exploration.",
            "method": "Store configurations, fidelity, score, and failure reason; retrieve similar prior configurations before proposing a new trial.",
            "memory_mechanism": "structured fact memory with lexical retrieval fallback",
            "expected_metric_movement": "Lower repeated configuration rate and better early regret.",
        }
    ],
    "memoryarena": [
        {
            "title": "Probe-driven episodic memory debugger for agent tasks",
            "dataset": "MemoryArena",
            "hypothesis": "Agent failures can be decomposed into missing memory, ignored memory, stale memory, and misleading retrieval.",
            "method": "Attach probe labels to each retrieved memory and visualize failure modes across sessions.",
            "memory_mechanism": "hybrid multi-tier memory",
            "expected_metric_movement": "More actionable failure attribution than aggregate task success.",
        }
    ],
    "longmemeval": [
        {
            "title": "Temporal evidence gap analysis for long memory",
            "dataset": "LongMemEval",
            "hypothesis": "Long-memory failures often come from retrieving temporally adjacent but outdated evidence.",
            "method": "Track timestamp metadata and classify stale-memory retrieval separately from generic retrieval miss.",
            "memory_mechanism": "timestamp-aware episodic memory",
            "expected_metric_movement": "Reduced temporal confusion in long-horizon QA.",
        }
    ],
    "lcbench": [
        {
            "title": "LCBench experiment-memory replay for research agents",
            "dataset": "LCBench",
            "hypothesis": "Learning-curve metadata can become memory evidence for deciding which model family to explore next.",
            "method": "Store curve summaries and retrieve similar datasets before proposing the next autonomous experiment.",
            "memory_mechanism": "episodic curve summaries plus extracted performance facts",
            "expected_metric_movement": "Better early stopping and fewer low-yield trials.",
        }
    ],
}


def generate_ideas(
    use_cases: list[str],
    ideas_per_case: int,
    dataset_registry: list[dict[str, Any]],
    autoresearch_inspection: dict[str, Any],
    client: OpenAICompatibleClient | None = None,
) -> list[dict[str, Any]]:
    ideas = []
    previous_texts = []
    registry_by_name = {item["name"].lower(): item for item in dataset_registry}
    for use_case in use_cases:
        for idx in range(ideas_per_case):
            idea = _llm_idea(use_case, idx, dataset_registry, autoresearch_inspection, client) if client else _offline_idea(use_case, idx)
            idea.setdefault("use_case", use_case)
            idea.setdefault("implementation_steps", [])
            idea.setdefault("risks", [])
            dataset_status = registry_by_name.get(str(idea.get("dataset", "")).lower(), {})
            idea["uses_real_local_data"] = bool(dataset_status.get("available_local")) or idea.get("dataset") == "autoresearch traces"
            idea["external_download_required"] = not idea["uses_real_local_data"] and idea.get("dataset") != "autoresearch traces"
            text = json.dumps(idea, sort_keys=True)
            idea["novelty_score"] = novelty_score(text, previous_texts)
            idea["redundant_idea"] = idea["novelty_score"] < 0.45
            idea["idea_id"] = f"{use_case}-{idx + 1}"
            previous_texts.append(text)
            ideas.append(idea)
    return ideas


def _offline_idea(use_case: str, idx: int) -> dict[str, Any]:
    templates = OFFLINE_IDEAS.get(use_case, OFFLINE_IDEAS["locomo"])
    base = dict(templates[idx % len(templates)])
    base.setdefault("novelty_rationale", "Deterministic tutorial idea derived from the accepted proposal use-case taxonomy.")
    base.setdefault("implementation_steps", [
        "Load available traces or dataset metadata.",
        "Store observations in the selected memory architecture.",
        "Run retrieval, utilization, and failure diagnostics.",
        "Visualize bottlenecks for tutorial participants.",
    ])
    base.setdefault("risks", ["Unavailable external dataset must be treated as future work."])
    return base


def _llm_idea(use_case: str, idx: int, dataset_registry: list[dict[str, Any]], autoresearch_inspection: dict[str, Any], client: OpenAICompatibleClient) -> dict[str, Any]:
    system = "You generate concise, implementable research ideas for a KDD tutorial on memory systems in autonomous AI research agents. Return JSON only."
    prompt = f"""
Generate one novel but practical experiment idea for use case: {use_case}.
This tutorial is already accepted, so focus on debugging memory behavior with real or explicitly listed datasets.
Dataset registry: {json.dumps(dataset_registry, indent=2)}
Autoresearch inspection: {json.dumps(autoresearch_inspection, indent=2)[:4000]}
Use local LoCoMo when real local data is needed. Mark unavailable datasets as external_download_required=true.
Do not include secrets. Return a JSON object with: title, dataset, hypothesis, method, memory_mechanism, expected_metric_movement, implementation_steps, risks, novelty_rationale.
Idea index: {idx + 1}
"""
    try:
        idea = client.chat_json(system, prompt)
    except Exception as exc:
        idea = _offline_idea(use_case, idx)
        idea["llm_error"] = str(exc)
    return idea
