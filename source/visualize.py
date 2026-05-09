import json
from pathlib import Path
from typing import Any


def create_figures(raw_path: Path | None, metrics_path: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_json(metrics_path)
    raw = _load_json(raw_path) if raw_path and raw_path.exists() else {}
    paths = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    locomo = metrics.get("locomo", {})
    if locomo:
        paths.append(_plot_locomo(locomo, out_dir, plt))
    real = metrics.get("real", {})
    if real:
        paths.append(_plot_real_retrieval(real, out_dir, plt))
        paths.append(_plot_semantic_scores(real, out_dir, plt))
    if raw.get("records"):
        paths.append(_plot_memory_growth(raw["records"], out_dir, plt))
        paths.append(_plot_failures(raw["records"], out_dir, plt))
        paths.append(_plot_semantic_vs_retrieval(raw["records"], out_dir, plt))
        paths.append(_plot_evaluator_coverage(raw["records"], out_dir, plt))
    ideas = raw.get("ideas") or _load_optional_ideas(raw_path)
    if ideas:
        paths.append(_plot_idea_novelty(ideas, out_dir, plt))
    autoresearch = raw.get("autoresearch", {}).get("records", [])
    if autoresearch:
        paths.append(_plot_autoresearch(autoresearch, out_dir, plt))
    return [str(path) for path in paths if path]


def _plot_locomo(locomo: dict[str, Any], out_dir: Path, plt):
    strategies = list(locomo.get("by_strategy", {}))
    if not strategies:
        return None
    precision = [locomo["by_strategy"][s].get("retrieval_precision", 0.0) for s in strategies]
    recall = [locomo["by_strategy"][s].get("retrieval_recall", 0.0) for s in strategies]
    hit = [locomo["by_strategy"][s].get("evidence_hit_rate", 0.0) for s in strategies]
    x = range(len(strategies))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([i - width for i in x], precision, width, label="precision")
    ax.bar(list(x), recall, width, label="recall")
    ax.bar([i + width for i in x], hit, width, label="evidence hit")
    ax.set_xticks(list(x), strategies, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("LoCoMo memory retrieval diagnostics")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "figure_locomo_retrieval.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_real_retrieval(real: dict[str, Any], out_dir: Path, plt):
    rows = []
    for dataset, dataset_summary in real.get("datasets", {}).items():
        for strategy, metrics in dataset_summary.get("by_strategy", {}).items():
            rows.append((f"{dataset}\n{strategy}", metrics.get("evidence_hit_rate", 0.0)))
    if not rows:
        return None
    labels, values = zip(*rows)
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.6), 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("evidence/context hit rate")
    ax.set_title("Real dataset memory retrieval diagnostics")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = out_dir / "figure_real_retrieval.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_semantic_scores(real: dict[str, Any], out_dir: Path, plt):
    rows = []
    for dataset, dataset_summary in real.get("datasets", {}).items():
        for strategy, metrics in dataset_summary.get("by_strategy", {}).items():
            if "mean_semantic_score" in metrics:
                rows.append((f"{dataset}\n{strategy}", metrics.get("mean_semantic_score", 0.0)))
    if not rows:
        return None
    labels, values = zip(*rows)
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.6), 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean semantic score")
    ax.set_title("Semantic evaluator scores")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = out_dir / "figure_semantic_scores.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_semantic_vs_retrieval(records: list[dict[str, Any]], out_dir: Path, plt):
    rows = [(float(record.get("retrieval_recall", 0.0)), float(record.get("semantic_score", 0.0))) for record in records if "semantic_score" in record]
    if not rows:
        return None
    x, y = zip(*rows)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.35, s=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("retrieval recall")
    ax.set_ylabel("semantic score")
    ax.set_title("Semantic score vs retrieval recall")
    fig.tight_layout()
    path = out_dir / "figure_semantic_vs_retrieval.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_evaluator_coverage(records: list[dict[str, Any]], out_dir: Path, plt):
    counts = {}
    for record in records:
        evaluator = record.get("semantic_evaluator")
        if evaluator:
            counts[evaluator] = counts.get(evaluator, 0) + 1
    if not counts:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(counts), list(counts.values()))
    ax.set_ylabel("evaluated records")
    ax.set_title("Evaluator coverage")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = out_dir / "figure_evaluator_coverage.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_memory_growth(records: list[dict[str, Any]], out_dir: Path, plt):
    grouped = {}
    for record in records:
        if "memory_entries" not in record:
            continue
        grouped.setdefault(record.get("strategy", "unknown"), []).append(record["memory_entries"])
    if not grouped:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    for strategy, values in grouped.items():
        ax.plot(range(1, len(values) + 1), values, label=strategy)
    ax.set_xlabel("step")
    ax.set_ylabel("memory entries")
    ax.set_title("Memory growth")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "figure_memory_growth.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_failures(records: list[dict[str, Any]], out_dir: Path, plt):
    counts = {}
    for record in records:
        failure = record.get("failure_category") or record.get("diagnosed_failure") or "none"
        counts[failure] = counts.get(failure, 0) + 1
    if not counts:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(counts), list(counts.values()))
    ax.set_ylabel("count")
    ax.set_title("Diagnosed memory failure modes")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = out_dir / "figure_failures.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_idea_novelty(ideas: list[dict[str, Any]], out_dir: Path, plt):
    labels = [idea.get("idea_id", str(i + 1)) for i, idea in enumerate(ideas)]
    values = [float(idea.get("novelty_score", 0.0)) for idea in ideas]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("novelty")
    ax.set_title("Generated idea novelty")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = out_dir / "figure_idea_novelty.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_autoresearch(records: list[dict[str, Any]], out_dir: Path, plt):
    vals = [record.get("val_bpb", 0.0) for record in records]
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(vals) + 1), vals, marker="o")
    ax.set_xlabel("experiment")
    ax.set_ylabel("val_bpb")
    ax.set_title("Autoresearch trace quality")
    fig.tight_layout()
    path = out_dir / "figure_autoresearch_trace.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _load_json(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_optional_ideas(raw_path: Path | None):
    if not raw_path:
        return []
    idea_path = Path(str(raw_path).replace("_raw.json", "_ideas.json"))
    if not idea_path.exists():
        return []
    return _load_json(idea_path).get("ideas", [])
