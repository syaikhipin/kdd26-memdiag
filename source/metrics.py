from collections import Counter, defaultdict
from statistics import mean


def summarize(records: list[dict], target_score: float) -> dict[str, dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["strategy"]].append(record)

    summary = {}
    for strategy, rows in grouped.items():
        best_score = max(row["outcome_score"] for row in rows) if rows else 0.0
        convergence = next((row["global_step"] for row in rows if row["outcome_score"] >= target_score), None)
        total = len(rows)
        redundant = sum(1 for row in rows if row["redundant_action"])
        improved = sum(1 for row in rows if row["outcome_label"] == "improved")
        unique_actions = len({(row["task_id"], row["proposed_action"]) for row in rows})
        summary[strategy] = {
            "episodes": total,
            "best_score": round(best_score, 4),
            "convergence_episode": convergence,
            "redundancy_rate": round(redundant / total, 4) if total else 0.0,
            "exploration_efficiency": round(unique_actions / total, 4) if total else 0.0,
            "improvement_rate": round(improved / total, 4) if total else 0.0,
            "retrieval_precision": round(mean(row["retrieval_precision"] for row in rows), 4),
            "retrieval_recall": round(mean(row["retrieval_recall"] for row in rows), 4),
            "utilization_rate": round(mean(1.0 if row["memory_utilized"] else 0.0 for row in rows), 4),
            "avg_latency_ms": round(mean(row["latency_ms"] for row in rows), 4),
            "cost_units": round(sum(row["cost_units"] for row in rows), 4),
            "memory_entries": rows[-1]["memory_entries"] if rows else 0,
            "score_auc": round(sum(row["outcome_score"] for row in rows), 4),
            "failure_modes": dict(sorted(_failure_counts(rows).items())),
        }
    return summary


def _failure_counts(rows: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        counts[row["diagnosed_failure"]] += 1
    return counts


def summarize_locomo(records: list[dict]) -> dict:
    summary = summarize_memory_benchmark(records)
    summary["dataset"] = "LoCoMo"
    return summary


def summarize_memory_benchmark(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["strategy"]].append(record)
    by_strategy = {}
    for strategy, rows in grouped.items():
        metrics = {
            "questions": len(rows),
            "retrieval_precision": _mean(rows, "retrieval_precision"),
            "retrieval_recall": _mean(rows, "retrieval_recall"),
            "evidence_hit_rate": round(mean(1.0 if row.get("evidence_hit") else 0.0 for row in rows), 4) if rows else 0.0,
            "utilization_rate": round(mean(1.0 if row.get("memory_utilized") else 0.0 for row in rows), 4) if rows else 0.0,
            "avg_latency_ms": _mean(rows, "latency_ms"),
            "cost_units": round(sum(float(row.get("cost_units", 0.0)) for row in rows), 4),
            "memory_entries": rows[-1].get("memory_entries", 0) if rows else 0,
            "failure_modes": dict(Counter(row.get("failure_category", "unknown") for row in rows)),
            "categories": _category_summary(rows),
        }
        metrics.update(_semantic_summary(rows))
        by_strategy[strategy] = metrics
    return {
        "dataset": records[0].get("dataset", "unknown") if records else "unknown",
        "records": len(records),
        "by_strategy": by_strategy,
    }


def summarize_real_datasets(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[record.get("dataset", "unknown")].append(record)
    return {
        "datasets": {dataset: summarize_memory_benchmark(rows) for dataset, rows in grouped.items()},
        "combined": summarize_memory_benchmark(records),
        "records": len(records),
    }


def summarize_ideas(ideas: list[dict]) -> dict:
    return {
        "ideas": len(ideas),
        "by_use_case": dict(Counter(idea.get("use_case", "unknown") for idea in ideas)),
        "real_local_data_ideas": sum(1 for idea in ideas if idea.get("uses_real_local_data")),
        "external_download_required": sum(1 for idea in ideas if idea.get("external_download_required")),
        "mean_novelty_score": round(mean(float(idea.get("novelty_score", 0.0)) for idea in ideas), 4) if ideas else 0.0,
        "redundancy_rate": round(mean(1.0 if idea.get("redundant_idea") else 0.0 for idea in ideas), 4) if ideas else 0.0,
    }


def summarize_autoresearch_trace(inspection: dict) -> dict:
    records = inspection.get("records", [])
    statuses = Counter(record.get("status", "unknown") for record in records)
    val_bpbs = [record.get("val_bpb", 0.0) for record in records if record.get("val_bpb", 0.0) > 0]
    memory = [record.get("memory_gb", 0.0) for record in records if record.get("memory_gb", 0.0) > 0]
    return {
        "results_tsv_available": inspection.get("results_tsv_available", False),
        "run_log_available": inspection.get("run_log_available", False),
        "missing": inspection.get("missing", []),
        "records": len(records),
        "status_counts": dict(statuses),
        "best_val_bpb": min(val_bpbs) if val_bpbs else None,
        "mean_memory_gb": round(mean(memory), 4) if memory else None,
        "run_log_summary": inspection.get("run_log_summary", {}),
    }


def summarize_tutorial(locomo_records: list[dict], ideas: list[dict], autoresearch_inspection: dict, dataset_registry: list[dict]) -> dict:
    return {
        "datasets": dataset_registry,
        "locomo": summarize_locomo(locomo_records),
        "ideas": summarize_ideas(ideas),
        "autoresearch": summarize_autoresearch_trace(autoresearch_inspection),
    }


def format_locomo_table(summary: dict) -> str:
    has_semantic = _has_semantic_metrics({"locomo": summary})
    headers = ["strategy", "questions", "retr_p", "retr_r", "hit", "util", "lat_ms", "cost", "mem"]
    if has_semantic:
        headers.extend(["sem_cov", "sem", "sem_pass", "faith", "ctx", "ans"])
    lines = ["\t".join(headers)]
    for strategy, metrics in summary.get("by_strategy", {}).items():
        row = [
            strategy,
            str(metrics["questions"]),
            f"{metrics['retrieval_precision']:.3f}",
            f"{metrics['retrieval_recall']:.3f}",
            f"{metrics['evidence_hit_rate']:.3f}",
            f"{metrics['utilization_rate']:.3f}",
            f"{metrics['avg_latency_ms']:.2f}",
            f"{metrics['cost_units']:.1f}",
            str(metrics["memory_entries"]),
        ]
        if has_semantic:
            row.extend([
                f"{metrics.get('semantic_coverage_rate', 0.0):.3f}",
                f"{metrics.get('mean_semantic_score', 0.0):.3f}",
                f"{metrics.get('semantic_pass_rate', 0.0):.3f}",
                f"{metrics.get('mean_faithfulness_score', 0.0):.3f}",
                f"{metrics.get('mean_context_relevance_score', 0.0):.3f}",
                f"{metrics.get('mean_answer_correctness_score', 0.0):.3f}",
            ])
        lines.append("\t".join(row))
    return "\n".join(lines)


def format_real_table(summary: dict) -> str:
    has_semantic = _has_semantic_metrics(summary.get("datasets", {}))
    headers = ["dataset", "strategy", "questions", "retr_p", "retr_r", "hit", "util", "lat_ms", "cost", "mem"]
    if has_semantic:
        headers.extend(["sem_cov", "sem", "sem_pass", "faith", "ctx", "ans"])
    lines = ["\t".join(headers)]
    for dataset, dataset_summary in summary.get("datasets", {}).items():
        for strategy, metrics in dataset_summary.get("by_strategy", {}).items():
            row = [
                dataset,
                strategy,
                str(metrics["questions"]),
                f"{metrics['retrieval_precision']:.3f}",
                f"{metrics['retrieval_recall']:.3f}",
                f"{metrics['evidence_hit_rate']:.3f}",
                f"{metrics['utilization_rate']:.3f}",
                f"{metrics['avg_latency_ms']:.2f}",
                f"{metrics['cost_units']:.1f}",
                str(metrics["memory_entries"]),
            ]
            if has_semantic:
                row.extend([
                    f"{metrics.get('semantic_coverage_rate', 0.0):.3f}",
                    f"{metrics.get('mean_semantic_score', 0.0):.3f}",
                    f"{metrics.get('semantic_pass_rate', 0.0):.3f}",
                    f"{metrics.get('mean_faithfulness_score', 0.0):.3f}",
                    f"{metrics.get('mean_context_relevance_score', 0.0):.3f}",
                    f"{metrics.get('mean_answer_correctness_score', 0.0):.3f}",
                ])
            lines.append("\t".join(row))
    return "\n".join(lines)


def _semantic_summary(rows: list[dict]) -> dict:
    attempted = [row for row in rows if "semantic_score" in row]
    if not attempted:
        return {}
    evaluated = [row for row in attempted if not row.get("semantic_error_redacted")]
    base = {
        "semantic_attempt_rate": round(len(attempted) / len(rows), 4) if rows else 0.0,
        "semantic_coverage_rate": round(len(evaluated) / len(rows), 4) if rows else 0.0,
        "semantic_error_count": sum(1 for row in attempted if row.get("semantic_error_redacted")),
        "semantic_by_evaluator": dict(Counter(row.get("semantic_evaluator", "unknown") for row in attempted)),
    }
    if not evaluated:
        return {
            **base,
            "mean_semantic_score": 0.0,
            "semantic_pass_rate": 0.0,
            "mean_faithfulness_score": 0.0,
            "mean_context_relevance_score": 0.0,
            "mean_answer_correctness_score": 0.0,
            "retrieval_semantic_alignment": _retrieval_semantic_alignment([]),
        }
    return {
        **base,
        "mean_semantic_score": _mean(evaluated, "semantic_score"),
        "semantic_pass_rate": round(mean(1.0 if row.get("semantic_pass") else 0.0 for row in evaluated), 4),
        "mean_faithfulness_score": _mean(evaluated, "faithfulness_score"),
        "mean_context_relevance_score": _mean(evaluated, "context_relevance_score"),
        "mean_answer_correctness_score": _mean(evaluated, "answer_correctness_score"),
        "retrieval_semantic_alignment": _retrieval_semantic_alignment(evaluated),
    }


def _retrieval_semantic_alignment(rows: list[dict]) -> dict:
    aligned = [row for row in rows if bool(row.get("evidence_hit")) == bool(row.get("semantic_pass"))]
    return {
        "agreement_rate": round(len(aligned) / len(rows), 4) if rows else 0.0,
        "mean_retrieval_recall": _mean(rows, "retrieval_recall"),
        "mean_semantic_score": _mean(rows, "semantic_score"),
    }


def _has_semantic_metrics(dataset_summaries: dict) -> bool:
    return any(
        "mean_semantic_score" in metrics
        for dataset_summary in dataset_summaries.values()
        for metrics in dataset_summary.get("by_strategy", {}).values()
    )


def format_semantic_table(summary: dict) -> str:
    datasets = summary.get("datasets", {summary.get("dataset", "unknown"): summary})
    headers = ["dataset", "strategy", "coverage", "sem", "pass", "faith", "ctx", "ans", "errors", "evaluator"]
    lines = ["\t".join(headers)]
    for dataset, dataset_summary in datasets.items():
        for strategy, metrics in dataset_summary.get("by_strategy", {}).items():
            if "mean_semantic_score" not in metrics:
                continue
            evaluators = ",".join(sorted(metrics.get("semantic_by_evaluator", {})))
            lines.append("\t".join([
                dataset,
                strategy,
                f"{metrics.get('semantic_coverage_rate', 0.0):.3f}",
                f"{metrics.get('semantic_coverage_rate', 0.0):.3f}",
                f"{metrics.get('mean_semantic_score', 0.0):.3f}",
                f"{metrics.get('semantic_pass_rate', 0.0):.3f}",
                f"{metrics.get('mean_faithfulness_score', 0.0):.3f}",
                f"{metrics.get('mean_context_relevance_score', 0.0):.3f}",
                f"{metrics.get('mean_answer_correctness_score', 0.0):.3f}",
                str(metrics.get("semantic_error_count", 0)),
                evaluators,
            ]))
    return "\n".join(lines)


def _mean(rows: list[dict], key: str) -> float:
    return round(mean(float(row.get(key, 0.0)) for row in rows), 4) if rows else 0.0


def _category_summary(rows: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("category", "unknown")].append(row)
    return {
        category: {
            "questions": len(items),
            "retrieval_recall": _mean(items, "retrieval_recall"),
            "evidence_hit_rate": round(mean(1.0 if item.get("evidence_hit") else 0.0 for item in items), 4),
        }
        for category, items in grouped.items()
    }


def format_summary_table(summary: dict[str, dict]) -> str:
    headers = [
        "strategy", "best", "conv", "redund", "explore", "retr_p", "retr_r", "util", "lat_ms", "cost", "mem"
    ]
    lines = ["\t".join(headers)]
    for strategy, metrics in summary.items():
        lines.append("\t".join([
            strategy,
            f"{metrics['best_score']:.3f}",
            str(metrics["convergence_episode"]),
            f"{metrics['redundancy_rate']:.3f}",
            f"{metrics['exploration_efficiency']:.3f}",
            f"{metrics['retrieval_precision']:.3f}",
            f"{metrics['retrieval_recall']:.3f}",
            f"{metrics['utilization_rate']:.3f}",
            f"{metrics['avg_latency_ms']:.2f}",
            f"{metrics['cost_units']:.1f}",
            str(metrics["memory_entries"]),
        ]))
    return "\n".join(lines)
