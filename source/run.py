import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from collections import Counter, defaultdict
from dataclasses import asdict

from agent_simulator import ResearchEnvironment, SimulatedResearchAgent, run_episode
from autoresearch_trace import autoresearch_records_to_episodes, inspect_autoresearch_dir
from config import (
    DEFAULT_AUTORESEARCH_DIR,
    DEFAULT_BASE_URL,
    DEFAULT_LOCOMO_PATH,
    DEFAULT_MODEL,
    DEFAULT_STRATEGIES,
    DEFAULT_USE_CASES,
    ExperimentConfig,
)
from data_loaders import (
    DEFAULT_LONGMEMEVAL_DIR,
    DEFAULT_MEMORYARENA_DIR,
    dataset_registry,
    iter_longmemeval_items,
    iter_memoryarena_items,
    load_locomo,
)
from diagnostics import novelty_score, precision_recall
from evaluators import make_evaluators, redact_secret
from idea_generator import generate_ideas
from llm_client import LLMConfig, make_client
from metrics import (
    format_locomo_table,
    format_real_table,
    format_summary_table,
    summarize,
    summarize_autoresearch_trace,
    summarize_locomo,
    summarize_real_datasets,
    summarize_tutorial,
)
from probes.failure_probe import classify_failure
from probes.retrieval_probe import retrieval_precision_recall
from probes.utilization_probe import memory_utilized, utilization_category
from memory_store import MemoryStore, tokenize
from research_loop import run_locomo_memory_debug, run_memory_benchmark
from strategies import ALL_STRATEGIES
from synthetic_data import load_tasks
from visualize import create_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KDD tutorial memory benchmark for autonomous research agents")
    parser.add_argument("--mode", default="dry-run", choices=["dry-run", "synthetic", "locomo", "real", "autoresearch-agent", "inspect-autoresearch", "tutorial", "visualize"])
    parser.add_argument("--backend", default="offline", choices=["offline", "openai-compatible"])
    parser.add_argument("--runner", default="local", choices=["local", "modal"])
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES, choices=sorted(ALL_STRATEGIES))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target-score", type=float, default=0.82)
    parser.add_argument("--tasks", type=Path, default=ExperimentConfig().tasks_path)
    parser.add_argument("--out-dir", type=Path, default=ExperimentConfig().results_dir)
    parser.add_argument("--locomo-path", type=Path, default=DEFAULT_LOCOMO_PATH)
    parser.add_argument("--autoresearch-dir", type=Path, default=DEFAULT_AUTORESEARCH_DIR)
    parser.add_argument("--max-conversations", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--datasets", nargs="+", default=["locomo", "longmemeval", "memoryarena"], choices=["locomo", "longmemeval", "memoryarena"])
    parser.add_argument("--longmemeval-dir", type=Path, default=DEFAULT_LONGMEMEVAL_DIR)
    parser.add_argument("--longmemeval-files", nargs="+", default=["longmemeval_oracle.json", "longmemeval_s_cleaned.json", "longmemeval_m_cleaned.json"])
    parser.add_argument("--memoryarena-dir", type=Path, default=DEFAULT_MEMORYARENA_DIR)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--use-cases", nargs="+", default=DEFAULT_USE_CASES, choices=["locomo", "autoresearch", "hpo", "memoryarena", "longmemeval", "lcbench"])
    parser.add_argument("--ideas-per-case", type=int, default=2)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--eval-backend", default="offline", choices=["offline", "rhesis", "semantica", "all"])
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-smoke-test", action="store_true")
    parser.add_argument("--eval-fail-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rhesis-api-key-env", default="RHESIS_API_KEY")
    parser.add_argument("--rhesis-base-url", default=None)
    parser.add_argument("--rhesis-model", default=None)
    parser.add_argument("--semantica-mode", default="extract", choices=["extract", "graph", "decision"])
    parser.add_argument("--modal-app-name", default="kdd-memory-benchmark")
    parser.add_argument("--modal-timeout", type=int, default=21600)
    parser.add_argument("--modal-cpu", type=float, default=None)
    parser.add_argument("--modal-memory", type=int, default=None)
    parser.add_argument("--modal-gpu", default="T4")
    parser.add_argument("--modal-detach", action="store_true")
    parser.add_argument("--modal-call-id", default=None)
    parser.add_argument("--modal-include-figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--benchmark-metrics", type=Path, default=None)
    return parser.parse_args()


def episode_to_record(strategy_name: str, episode, global_step: int, memory_entries: int) -> dict:
    retrieval_metrics = retrieval_precision_recall(episode.retrieved_memory_ids, episode.relevant_memory_ids)
    used = memory_utilized(episode.referenced_memory_ids, episode.retrieved_memory_ids)
    failure = classify_failure(
        episode.failure_mode,
        episode.retrieved_memory_ids,
        episode.relevant_memory_ids,
        episode.referenced_memory_ids,
        episode.redundant_action,
    )
    return {
        "strategy": strategy_name,
        "global_step": global_step,
        "task_id": episode.task_id,
        "episode_id": episode.episode_id,
        "question": episode.query,
        "proposed_action": episode.proposed_action,
        "rationale": episode.rationale,
        "outcome_score": episode.outcome_score,
        "outcome_label": episode.outcome_label,
        "failure_mode": episode.failure_mode,
        "diagnosed_failure": failure,
        "redundant_action": episode.redundant_action,
        "retrieved_memory_ids": episode.retrieved_memory_ids,
        "referenced_memory_ids": episode.referenced_memory_ids,
        "relevant_memory_ids": episode.relevant_memory_ids,
        "retrieved_texts": episode.retrieved_texts,
        "retrieval_scores": episode.retrieval_scores,
        "retrieval_precision": retrieval_metrics["precision"],
        "retrieval_recall": retrieval_metrics["recall"],
        "memory_utilized": used,
        "utilization_category": utilization_category(
            episode.referenced_memory_ids,
            episode.retrieved_memory_ids,
            episode.outcome_label,
        ),
        "latency_ms": episode.latency_ms,
        "cost_units": episode.cost_units,
        "memory_entries": memory_entries,
    }


def run_strategy(strategy_name: str, config: ExperimentConfig, tasks) -> tuple[list[dict], dict]:
    strategy = ALL_STRATEGIES[strategy_name]()
    agent = SimulatedResearchAgent(config.seed)
    env = ResearchEnvironment(config.seed)
    records = []
    global_step = 0
    for step_idx in range(config.episodes):
        task = tasks[step_idx % len(tasks)]
        episode = run_episode(strategy, agent, env, task, step_idx, config.top_k)
        strategy.ingest_episode(task, episode)
        global_step += 1
        records.append(episode_to_record(strategy_name, episode, global_step, strategy.memory_size()))
    return records, strategy.export_memory()


def run_synthetic(args: argparse.Namespace) -> None:
    config = ExperimentConfig(
        mode="synthetic",
        backend=args.backend,
        episodes=args.episodes,
        seed=args.seed,
        top_k=args.top_k,
        target_score=args.target_score,
        tasks_path=args.tasks,
        results_dir=args.out_dir,
    )
    tasks = load_tasks(config.tasks_path)
    records = []
    memories = {}
    for strategy_name in args.strategies:
        strategy_records, memory = run_strategy(strategy_name, config, tasks)
        records.extend(strategy_records)
        memories[strategy_name] = memory
    summary = summarize(records, config.target_score)
    paths = _write_outputs(args.out_dir, "synthetic", {"config": _safe_config(args), "records": records, "memories": memories}, summary, format_summary_table(summary))
    print(format_summary_table(summary))
    _print_paths(paths)
    print("\nAPI key required: no for synthetic/offline mode.")


def run_locomo(args: argparse.Namespace) -> tuple[dict, dict]:
    conversations = load_locomo(args.locomo_path)
    records, memories = run_locomo_memory_debug(
        conversations=conversations,
        strategy_names=args.strategies,
        top_k=args.top_k,
        max_conversations=args.max_conversations,
        max_questions=args.max_questions,
        answer_mode="offline_evidence_heuristic" if args.backend == "offline" else "llm_with_retrieved_memory",
    )
    _apply_evaluators(records, args)
    summary = summarize_locomo(records)
    payload = {"config": _safe_config(args), "records": records, "memories": memories}
    paths = _write_outputs(args.out_dir, "locomo", payload, {"locomo": summary}, format_locomo_table(summary))
    if args.visualize:
        create_figures(paths["raw"], paths["metrics"], args.out_dir)
    print(format_locomo_table(summary))
    _print_paths(paths)
    return payload, {"locomo": summary}


def run_real(args: argparse.Namespace) -> tuple[dict, dict]:
    records = []
    memories = {}
    dataset_items = _load_real_items(args)
    for dataset_name, items in dataset_items.items():
        dataset_records, dataset_memories = run_memory_benchmark(
            items=items,
            strategy_names=args.strategies,
            top_k=args.top_k,
            answer_mode="offline_evidence_heuristic" if args.backend == "offline" else "llm_with_retrieved_memory",
        )
        records.extend(dataset_records)
        memories[dataset_name] = dataset_memories
    _apply_evaluators(records, args)
    summary = summarize_real_datasets(records)
    payload = {"config": _safe_config(args), "records": records, "memories": memories, "dataset_registry": dataset_registry(args.locomo_path)}
    table = format_real_table(summary)
    paths = _write_outputs(args.out_dir, "real", payload, {"real": summary}, table)
    if args.visualize:
        figures = create_figures(paths["raw"], paths["metrics"], args.out_dir)
        print("Figures:")
        for figure in figures:
            print(f"  {figure}")
    print(table)
    _print_paths(paths)
    return payload, {"real": summary}


def _load_real_items(args: argparse.Namespace) -> dict[str, list]:
    datasets = set(args.datasets)
    loaded = {}
    if "locomo" in datasets:
        loaded["LoCoMo"] = list(_iter_locomo_real_items(args))
    if "longmemeval" in datasets:
        items = []
        for file_name in args.longmemeval_files:
            path = args.longmemeval_dir / file_name
            if path.exists():
                items.extend(iter_longmemeval_items(path, max_items=_dataset_item_limit(args)))
        loaded["LongMemEval"] = items
    if "memoryarena" in datasets:
        items = []
        for path in sorted(args.memoryarena_dir.glob("*.jsonl")):
            items.extend(iter_memoryarena_items(path, max_items=_dataset_item_limit(args)))
        loaded["MemoryArena"] = items
    return loaded


def _dataset_item_limit(args: argparse.Namespace) -> int | None:
    if args.eval_smoke_test:
        return 1 if args.max_items is None else min(args.max_items, 1)
    return args.max_items


def _iter_locomo_real_items(args: argparse.Namespace):
    from data_loaders import iter_locomo_benchmark_items
    max_conversations = 1 if args.eval_smoke_test else args.max_conversations
    max_questions = 1 if args.eval_smoke_test else args.max_questions
    return iter_locomo_benchmark_items(args.locomo_path, max_conversations, max_questions)


def run_autoresearch_agent(args: argparse.Namespace) -> tuple[dict, dict]:
    metrics_path = args.benchmark_metrics or _latest_metrics_path(args.out_dir, "real")
    if not metrics_path or not metrics_path.exists():
        raise SystemExit("Run --mode real first or pass --benchmark-metrics pointing to real metrics JSON.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        benchmark_metrics = json.load(f)
    real_summary = benchmark_metrics.get("real", benchmark_metrics)
    registry = dataset_registry(args.locomo_path)
    client = make_client(LLMConfig(
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        api_key=args.api_key,
    ))
    seed_memories = _benchmark_memories(real_summary)
    ideas = generate_ideas(args.use_cases, args.ideas_per_case, registry, {"records": seed_memories}, client)
    records, memories = _run_autoresearch_agent_loop(ideas, seed_memories, args.top_k)
    _apply_evaluators(records, args)
    summary = _summarize_autoresearch_agent(records)
    table = _format_autoresearch_agent_table(summary)
    payload = {
        "config": _safe_config(args),
        "benchmark_metrics": str(metrics_path),
        "dataset_registry": registry,
        "seed_memories": seed_memories,
        "ideas": ideas,
        "records": records,
        "memories": memories,
    }
    paths = _write_outputs(args.out_dir, "autoresearch_agent", payload, {"autoresearch_agent": summary}, table)
    report_path = paths["raw"].with_name(paths["raw"].name.replace("_raw.json", "_report.md"))
    report_path.write_text(_autoresearch_agent_report(summary, paths), encoding="utf-8")
    print(table)
    print(f"Autoresearch agent report: {report_path}")
    _print_paths(paths)
    return payload, {"autoresearch_agent": summary}


def _latest_metrics_path(out_dir: Path, prefix: str) -> Path | None:
    paths = sorted(out_dir.glob(f"run_*_{prefix}_metrics.json"))
    return paths[-1] if paths else None


def _benchmark_memories(real_summary: dict) -> list[dict]:
    memories = []
    for dataset, dataset_summary in real_summary.get("datasets", {}).items():
        for strategy, metrics in dataset_summary.get("by_strategy", {}).items():
            if strategy == "no_memory":
                continue
            hit = float(metrics.get("evidence_hit_rate", 0.0))
            recall = float(metrics.get("retrieval_recall", 0.0))
            precision = float(metrics.get("retrieval_precision", 0.0))
            failures = metrics.get("failure_modes", {})
            decision = "keep" if hit >= 0.5 else "revise"
            content = (
                f"Experiment result: dataset={dataset}, strategy={strategy}, precision={precision:.3f}, "
                f"recall={recall:.3f}, hit_rate={hit:.3f}, decision={decision}, failures={failures}"
            )
            memories.append({
                "id": f"{dataset}:{strategy}",
                "dataset": dataset,
                "strategy": strategy,
                "content": content,
                "hit_rate": hit,
                "retrieval_recall": recall,
                "retrieval_precision": precision,
                "decision": decision,
                "failure_modes": failures,
            })
    return memories


def _run_autoresearch_agent_loop(ideas: list[dict], seed_memories: list[dict], top_k: int) -> tuple[list[dict], dict]:
    store = MemoryStore("autoresearch_agent")
    for memory in seed_memories:
        store.add(
            content=memory["content"],
            metadata={**memory, "record_id": memory["id"], "entry_type": "benchmark_result"},
            source_episode="real_benchmark",
            source_task=memory["dataset"],
            entry_type="benchmark_result",
        )
    records = []
    previous_ideas = []
    for step, idea in enumerate(ideas, start=1):
        query = _idea_query(idea)
        retrieved, latency_ms = store.retrieve(query, top_k=top_k)
        retrieved_ids = [item["entry"].metadata.get("record_id", item["entry"].id) for item in retrieved]
        relevant_ids = _relevant_memory_ids(idea, seed_memories)
        retrieval = precision_recall(retrieved_ids, relevant_ids)
        referenced_ids = _referenced_memory_ids(idea, retrieved)
        used = bool(set(referenced_ids) & set(relevant_ids))
        idea_text = json.dumps(idea, sort_keys=True)
        novelty = novelty_score(idea_text, previous_ideas)
        redundant = novelty < 0.45 or _overlaps_prior_method(idea, records)
        outcome = _agent_outcome(idea, retrieved, used, redundant)
        failure = _agent_failure(bool(retrieved), retrieval["hits"], used, redundant, outcome["decision"])
        record = {
            "step": step,
            "idea_id": idea.get("idea_id", f"idea-{step}"),
            "title": idea.get("title", "untitled"),
            "dataset": idea.get("dataset", "unknown"),
            "use_case": idea.get("use_case", "unknown"),
            "hypothesis": idea.get("hypothesis", ""),
            "memory_mechanism": idea.get("memory_mechanism", ""),
            "retrieved_memory_ids": retrieved_ids,
            "referenced_memory_ids": referenced_ids,
            "relevant_memory_ids": relevant_ids,
            "retrieved_texts": [item["entry"].content for item in retrieved],
            "retrieval_precision": retrieval["precision"],
            "retrieval_recall": retrieval["recall"],
            "memory_utilized": used,
            "novelty_score": novelty,
            "redundant_idea": redundant,
            "decision": outcome["decision"],
            "expected_metric_movement": idea.get("expected_metric_movement", ""),
            "observed_signal": outcome["observed_signal"],
            "failure_category": failure,
            "diagnosed_failure": failure,
            "latency_ms": latency_ms,
            "cost_units": 0.1 + 0.02 * len(retrieved) + 0.001 * store.size(),
            "memory_entries": store.size(),
        }
        records.append(record)
        previous_ideas.append(idea_text)
        store.add(
            content=_idea_memory_text(record),
            metadata={"record_id": record["idea_id"], "dataset": record["dataset"], "entry_type": "autoresearch_idea", "decision": record["decision"]},
            source_episode=record["idea_id"],
            source_task=record["dataset"],
            entry_type="autoresearch_idea",
        )
    return records, store.to_dict()


def _idea_query(idea: dict) -> str:
    query = " ".join(str(idea.get(key, "")) for key in ["dataset", "use_case", "title", "hypothesis", "method", "memory_mechanism"])
    return f"{query} {_canonical_dataset(query)}"


def _relevant_memory_ids(idea: dict, seed_memories: list[dict]) -> list[str]:
    query = _idea_query(idea)
    idea_tokens = set(tokenize(query))
    idea_dataset = _canonical_dataset(query)
    relevant = []
    for memory in seed_memories:
        memory_tokens = set(tokenize(memory["content"] + " " + memory["dataset"] + " " + memory["strategy"]))
        if idea_dataset and idea_dataset == _canonical_dataset(memory["dataset"]):
            relevant.append(memory["id"])
        elif idea_tokens & memory_tokens:
            relevant.append(memory["id"])
    return relevant


def _canonical_dataset(text: str) -> str:
    compact = text.lower().replace(" ", "")
    if "memoryarena" in compact:
        return "memoryarena"
    if "longmemeval" in compact:
        return "longmemeval"
    if "locomo" in compact:
        return "locomo"
    if "lcbench" in compact:
        return "lcbench"
    if "hpobench" in compact or "hpo" in compact:
        return "hpobench"
    return ""


def _referenced_memory_ids(idea: dict, retrieved: list[dict]) -> list[str]:
    text = _idea_query(idea).lower()
    idea_dataset = _canonical_dataset(text)
    referenced = []
    for item in retrieved:
        metadata = item["entry"].metadata
        dataset = str(metadata.get("dataset", "")).lower()
        strategy = str(metadata.get("strategy", "")).lower()
        if idea_dataset and idea_dataset == _canonical_dataset(dataset):
            referenced.append(metadata.get("record_id", item["entry"].id))
        elif strategy and strategy.lower() in text:
            referenced.append(metadata.get("record_id", item["entry"].id))
    return referenced


def _overlaps_prior_method(idea: dict, records: list[dict]) -> bool:
    current = set(tokenize(str(idea.get("memory_mechanism", "")) + " " + str(idea.get("dataset", ""))))
    for record in records:
        prior = set(tokenize(record.get("memory_mechanism", "") + " " + record.get("dataset", "")))
        if current and prior and len(current & prior) / len(current | prior) > 0.7:
            return True
    return False


def _agent_outcome(idea: dict, retrieved: list[dict], used: bool, redundant: bool) -> dict:
    signals = [float(item["entry"].metadata.get("hit_rate", 0.0)) for item in retrieved if item["entry"].metadata.get("hit_rate") is not None]
    best_signal = max(signals, default=0.0)
    if redundant:
        decision = "discard"
    elif used and best_signal >= 0.5:
        decision = "keep"
    elif retrieved:
        decision = "revise"
    else:
        decision = "needs_evidence"
    return {"decision": decision, "observed_signal": round(best_signal, 4)}


def _agent_failure(has_retrieved: bool, hits: int, used: bool, redundant: bool, decision: str) -> str:
    if redundant:
        return "redundant_idea"
    if not has_retrieved:
        return "no_memory_available"
    if hits == 0:
        return "retrieval_miss"
    if not used:
        return "retrieved_but_not_used"
    if decision in {"revise", "needs_evidence"}:
        return "low_signal_prior"
    return "none"


def _idea_memory_text(record: dict) -> str:
    return (
        f"Autoresearch idea {record['idea_id']}: dataset={record['dataset']}, mechanism={record['memory_mechanism']}, "
        f"decision={record['decision']}, novelty={record['novelty_score']}, failure={record['failure_category']}, "
        f"observed_signal={record['observed_signal']}"
    )


def _summarize_autoresearch_agent(records: list[dict]) -> dict:
    summary = {
        "ideas": len(records),
        "kept": sum(1 for record in records if record["decision"] == "keep"),
        "revised": sum(1 for record in records if record["decision"] == "revise"),
        "discarded": sum(1 for record in records if record["decision"] == "discard"),
        "mean_retrieval_precision": _mean_records(records, "retrieval_precision"),
        "mean_retrieval_recall": _mean_records(records, "retrieval_recall"),
        "memory_utilization_rate": round(sum(1 for record in records if record["memory_utilized"]) / len(records), 4) if records else 0.0,
        "redundant_idea_rate": round(sum(1 for record in records if record["redundant_idea"]) / len(records), 4) if records else 0.0,
        "mean_novelty_score": _mean_records(records, "novelty_score"),
        "failure_modes": dict(Counter(record["failure_category"] for record in records)),
        "decisions": dict(Counter(record["decision"] for record in records)),
    }
    evaluated = [record for record in records if "semantic_score" in record]
    if evaluated:
        summary.update({
            "semantic_coverage_rate": round(len(evaluated) / len(records), 4) if records else 0.0,
            "mean_semantic_score": _mean_records(evaluated, "semantic_score"),
            "semantic_pass_rate": round(sum(1 for record in evaluated if record.get("semantic_pass")) / len(evaluated), 4),
            "semantic_by_evaluator": dict(Counter(record.get("semantic_evaluator", "unknown") for record in evaluated)),
        })
    return summary


def _mean_records(records: list[dict], key: str) -> float:
    return round(sum(float(record.get(key, 0.0)) for record in records) / len(records), 4) if records else 0.0


def _format_autoresearch_agent_table(summary: dict) -> str:
    return "\n".join([
        "metric\tvalue",
        f"ideas\t{summary['ideas']}",
        f"kept\t{summary['kept']}",
        f"revised\t{summary['revised']}",
        f"discarded\t{summary['discarded']}",
        f"mean_retrieval_precision\t{summary['mean_retrieval_precision']}",
        f"mean_retrieval_recall\t{summary['mean_retrieval_recall']}",
        f"memory_utilization_rate\t{summary['memory_utilization_rate']}",
        f"redundant_idea_rate\t{summary['redundant_idea_rate']}",
        f"mean_novelty_score\t{summary['mean_novelty_score']}",
        f"mean_semantic_score\t{summary.get('mean_semantic_score', 0.0)}",
        f"semantic_pass_rate\t{summary.get('semantic_pass_rate', 0.0)}",
        f"failure_modes\t{json.dumps(summary['failure_modes'], sort_keys=True)}",
        f"decisions\t{json.dumps(summary['decisions'], sort_keys=True)}",
    ])


def _autoresearch_agent_report(summary: dict, paths: dict[str, Path]) -> str:
    return f"""# Autoresearch Agent Memory-Debug Report

## KDD tutorial role

This run demonstrates the proposal's autonomous-research-agent loop: generate research ideas, retrieve prior experiment memories, decide keep/revise/discard, and diagnose memory failures.

## Summary

{_format_autoresearch_agent_table(summary)}

## Outputs

- Raw trace: {paths['raw']}
- Metrics: {paths['metrics']}
- Summary TSV: {paths['summary']}
"""


def run_inspect_autoresearch(args: argparse.Namespace) -> dict:
    inspection = inspect_autoresearch_dir(args.autoresearch_dir)
    summary = summarize_autoresearch_trace(inspection)
    payload = {"config": _safe_config(args), "autoresearch": inspection}
    paths = _write_outputs(args.out_dir, "autoresearch", payload, {"autoresearch": summary}, _dict_tsv(summary))
    inspection_path = paths["raw"].with_name(paths["raw"].name.replace("_raw.json", "_autoresearch_inspection.json"))
    _write_json(inspection_path, inspection)
    print(json.dumps(summary, indent=2))
    _print_paths(paths)
    print(f"Autoresearch inspection: {inspection_path}")
    return inspection


def run_tutorial(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    registry = dataset_registry(args.locomo_path)
    inspection = inspect_autoresearch_dir(args.autoresearch_dir)
    client = make_client(LLMConfig(
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        api_key=args.api_key,
    ))
    conversations = load_locomo(args.locomo_path)
    locomo_records, memories = run_locomo_memory_debug(
        conversations=conversations,
        strategy_names=args.strategies,
        top_k=args.top_k,
        max_conversations=args.max_conversations,
        max_questions=args.max_questions,
        answer_mode="offline_evidence_heuristic" if args.backend == "offline" else "llm_with_retrieved_memory",
    )
    ideas = generate_ideas(args.use_cases, args.ideas_per_case, registry, inspection, client)
    autoresearch_episodes = autoresearch_records_to_episodes(inspection.get("records", []))
    _apply_evaluators(locomo_records, args)
    summary = summarize_tutorial(locomo_records, ideas, inspection, registry)
    payload = {
        "config": _safe_config(args),
        "dataset_registry": registry,
        "records": locomo_records,
        "memories": memories,
        "ideas": ideas,
        "autoresearch": {**inspection, "episodes": autoresearch_episodes},
    }
    table = format_locomo_table(summary["locomo"])
    paths = _write_outputs(args.out_dir, "tutorial", payload, summary, table)
    idea_path = paths["raw"].with_name(paths["raw"].name.replace("_raw.json", "_ideas.json"))
    registry_path = paths["raw"].with_name(paths["raw"].name.replace("_raw.json", "_dataset_registry.json"))
    inspection_path = paths["raw"].with_name(paths["raw"].name.replace("_raw.json", "_autoresearch_inspection.json"))
    report_path = paths["raw"].with_name(paths["raw"].name.replace("_raw.json", "_tutorial_report.md"))
    _write_json(idea_path, {"ideas": ideas})
    _write_json(registry_path, {"datasets": registry})
    _write_json(inspection_path, inspection)
    report_path.write_text(_tutorial_report(summary, paths, idea_path), encoding="utf-8")
    if args.visualize:
        figures = create_figures(paths["raw"], paths["metrics"], args.out_dir)
        print("Figures:")
        for figure in figures:
            print(f"  {figure}")
    print(table)
    print(f"Ideas: {idea_path}")
    print(f"Dataset registry: {registry_path}")
    print(f"Tutorial report: {report_path}")
    _print_paths(paths)


def run_visualize(args: argparse.Namespace) -> None:
    if not args.metrics:
        raise SystemExit("--metrics is required for --mode visualize")
    figures = create_figures(args.raw, args.metrics, args.out_dir)
    for figure in figures:
        print(figure)


def _apply_evaluators(records: list[dict], args: argparse.Namespace) -> None:
    evaluators = make_evaluators(
        args.eval_backend,
        rhesis_api_key_env=args.rhesis_api_key_env,
        rhesis_model=args.rhesis_model,
        rhesis_base_url=args.rhesis_base_url,
        semantica_mode=args.semantica_mode,
    )
    if not evaluators:
        return
    limit = args.eval_limit
    if args.eval_smoke_test:
        limit = 1 if limit is None else min(limit, 1)
    selected = _select_eval_records(records, limit)
    for record in selected:
        results = []
        for evaluator in evaluators:
            try:
                results.append(evaluator.evaluate(record))
            except Exception as exc:
                if not args.eval_fail_open:
                    raise
                results.append(type("FailedEvaluation", (), {
                    "evaluator": getattr(evaluator, "name", "unknown"),
                    "semantic_score": 0.0,
                    "faithfulness_score": 0.0,
                    "context_relevance_score": 0.0,
                    "answer_correctness_score": 0.0,
                    "passed": False,
                    "reason": "evaluation failed",
                    "details": {},
                    "error_redacted": redact_secret(exc),
                })())
        _attach_evaluation_results(record, results)


def _select_eval_records(records: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        return []
    groups = defaultdict(list)
    for record in records:
        groups[(record.get("dataset", "unknown"), record.get("strategy", record.get("use_case", "unknown")))].append(record)
    selected = []
    group_values = [items for _, items in sorted(groups.items())]
    offset = 0
    while len(selected) < limit:
        added = False
        for items in group_values:
            if offset < len(items):
                selected.append(items[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def _attach_evaluation_results(record: dict, results: list) -> None:
    if not results:
        return
    if len(results) == 1:
        result = results[0]
        if hasattr(result, "to_record_fields"):
            fields = result.to_record_fields()
        else:
            raw = vars(result)
            fields = {
                "semantic_evaluator": raw.get("evaluator", "unknown"),
                "semantic_score": raw.get("semantic_score", 0.0),
                "faithfulness_score": raw.get("faithfulness_score", 0.0),
                "context_relevance_score": raw.get("context_relevance_score", 0.0),
                "answer_correctness_score": raw.get("answer_correctness_score", 0.0),
                "semantic_pass": raw.get("passed", False),
                "semantic_reason": raw.get("reason", ""),
                "semantic_details": raw.get("details", {}),
                "semantic_error_redacted": raw.get("error_redacted"),
            }
        record.update(fields)
        _attach_evaluator_specific_fields(record, fields)
        return
    serialized = [asdict(result) if hasattr(result, "__dataclass_fields__") else vars(result) for result in results]
    successful = [result for result in serialized if not result.get("error_redacted")]
    scored = successful or serialized
    record.update({
        "semantic_evaluator": "+".join(item.get("evaluator", "unknown") for item in serialized),
        "semantic_score": _mean_eval_field(scored, "semantic_score"),
        "faithfulness_score": _mean_eval_field(scored, "faithfulness_score"),
        "context_relevance_score": _mean_eval_field(scored, "context_relevance_score"),
        "answer_correctness_score": _mean_eval_field(scored, "answer_correctness_score"),
        "semantic_pass": any(bool(item.get("passed")) for item in scored),
        "semantic_reason": " | ".join(str(item.get("reason", "")) for item in serialized if item.get("reason"))[:1000],
        "semantic_details": {"evaluations": serialized},
        "semantic_error_redacted": " | ".join(str(item.get("error_redacted", "")) for item in serialized if item.get("error_redacted")) or None,
    })
    for item in serialized:
        _attach_evaluator_specific_fields(record, {"semantic_details": item.get("details", {})})


def _mean_eval_field(results: list[dict], key: str) -> float:
    return round(sum(float(item.get(key, 0.0)) for item in results) / len(results), 4) if results else 0.0


def _attach_evaluator_specific_fields(record: dict, fields: dict) -> None:
    details = fields.get("semantic_details") or {}
    if "entities" in details:
        record["semantica_entities"] = details["entities"]
    if "triplet_count" in details:
        record["semantica_triplets"] = details["triplet_count"]
    if "rhesis_model" in details:
        record["rhesis_metric_name"] = details["rhesis_model"]


def _write_outputs(out_dir: Path, prefix: str, payload: dict, metrics: dict, table: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"run_{timestamp}_{prefix}_raw.json"
    metrics_path = out_dir / f"run_{timestamp}_{prefix}_metrics.json"
    tsv_path = out_dir / f"run_{timestamp}_{prefix}_summary.tsv"
    _write_json(raw_path, payload)
    _write_json(metrics_path, metrics)
    tsv_path.write_text(table + "\n", encoding="utf-8")
    return {"raw": raw_path, "metrics": metrics_path, "summary": tsv_path}


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _safe_config(args: argparse.Namespace) -> dict:
    return {
        "mode": args.mode,
        "backend": args.backend,
        "runner": args.runner,
        "base_url": args.base_url,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "api_key_present": bool(args.api_key or os.environ.get(args.api_key_env)),
        "top_k": args.top_k,
        "seed": args.seed,
        "locomo_path": str(args.locomo_path),
        "autoresearch_dir": str(args.autoresearch_dir),
        "max_conversations": args.max_conversations,
        "max_questions": args.max_questions,
        "datasets": getattr(args, "datasets", []),
        "max_items": getattr(args, "max_items", None),
        "longmemeval_dir": str(getattr(args, "longmemeval_dir", "")),
        "memoryarena_dir": str(getattr(args, "memoryarena_dir", "")),
        "benchmark_metrics": str(getattr(args, "benchmark_metrics", "")),
        "use_cases": args.use_cases,
        "eval": _safe_eval_config(args),
        "modal": _safe_modal_config(args),
    }


def _safe_eval_config(args: argparse.Namespace) -> dict:
    return {
        "backend": args.eval_backend,
        "limit": args.eval_limit,
        "smoke_test": args.eval_smoke_test,
        "fail_open": args.eval_fail_open,
        "rhesis_api_key_env": args.rhesis_api_key_env,
        "rhesis_api_key_present": bool(os.environ.get(args.rhesis_api_key_env)),
        "rhesis_base_url_present": bool(args.rhesis_base_url or os.environ.get("RHESIS_BASE_URL")),
        "rhesis_model": args.rhesis_model,
        "semantica_mode": args.semantica_mode,
    }


def _safe_modal_config(args: argparse.Namespace) -> dict:
    return {
        "app_name": args.modal_app_name,
        "timeout": args.modal_timeout,
        "cpu": args.modal_cpu,
        "memory": args.modal_memory,
        "gpu": args.modal_gpu,
        "detach": args.modal_detach,
        "call_id_present": bool(args.modal_call_id),
        "include_figures": args.modal_include_figures,
        "modal_token_present": bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")),
    }


def _print_paths(paths: dict[str, Path]) -> None:
    print(f"\nRaw trace: {paths['raw']}")
    print(f"Metrics: {paths['metrics']}")
    print(f"Summary TSV: {paths['summary']}")


def _dict_tsv(summary: dict) -> str:
    return "metric\tvalue\n" + "\n".join(f"{key}\t{value}" for key, value in summary.items())


def _tutorial_report(summary: dict, paths: dict[str, Path], idea_path: Path) -> str:
    return f"""# KDD Tutorial Memory-Debug Experiment Report

## Real datasets from the accepted proposal

- LoCoMo is used directly from the local PDF-listed dataset path.
- LCBench, MemoryArena, HPOBench, and LongMemEval are included in the dataset registry and treated as future/local-path use cases unless data is provided.

## LoCoMo memory diagnostics

{format_locomo_table(summary['locomo'])}

## Idea generation

- Ideas generated: {summary['ideas']['ideas']}
- Mean novelty score: {summary['ideas']['mean_novelty_score']}
- External-download ideas: {summary['ideas']['external_download_required']}

## Autoresearch traces

- results.tsv available: {summary['autoresearch']['results_tsv_available']}
- run.log available: {summary['autoresearch']['run_log_available']}
- missing: {summary['autoresearch']['missing']}

## Outputs

- Raw trace: {paths['raw']}
- Metrics: {paths['metrics']}
- Summary TSV: {paths['summary']}
- Ideas: {idea_path}
"""


def main() -> None:
    args = parse_args()
    if args.runner == "modal":
        if args.mode != "real":
            raise SystemExit("--runner modal currently supports only --mode real")
        from modal_runner import run_real_on_modal
        run_real_on_modal(args)
        return
    if args.mode in {"dry-run", "synthetic"}:
        run_synthetic(args)
    elif args.mode == "locomo":
        run_locomo(args)
    elif args.mode == "real":
        run_real(args)
    elif args.mode == "autoresearch-agent":
        run_autoresearch_agent(args)
    elif args.mode == "inspect-autoresearch":
        run_inspect_autoresearch(args)
    elif args.mode == "tutorial":
        run_tutorial(args)
    elif args.mode == "visualize":
        run_visualize(args)


if __name__ == "__main__":
    main()
