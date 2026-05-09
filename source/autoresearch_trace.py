import csv
import re
from pathlib import Path
from typing import Any

SUMMARY_RE = re.compile(r"^([a-zA-Z_]+):\s+(.+)$")
NUMERIC_KEYS = {
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
}


def inspect_autoresearch_dir(path: Path) -> dict[str, Any]:
    results_path = path / "results.tsv"
    run_log_path = path / "run.log"
    inspection = {
        "autoresearch_dir": str(path),
        "results_tsv_available": results_path.exists(),
        "run_log_available": run_log_path.exists(),
        "missing": [],
        "records": [],
        "run_log_summary": {},
    }
    if results_path.exists():
        inspection["records"] = parse_results_tsv(results_path)
    else:
        inspection["missing"].append("results.tsv")
    if run_log_path.exists():
        inspection["run_log_summary"] = parse_run_log(run_log_path)
    else:
        inspection["missing"].append("run.log")
    return inspection


def parse_results_tsv(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            rows.append({
                "commit": row.get("commit", ""),
                "val_bpb": _to_float(row.get("val_bpb")),
                "memory_gb": _to_float(row.get("memory_gb")),
                "status": str(row.get("status", "")).strip().lower(),
                "description": row.get("description", ""),
            })
    return rows


def parse_run_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    summary = {}
    for part in re.split(r"[\r\n]+", text):
        match = SUMMARY_RE.match(part.strip())
        if not match:
            continue
        key, value = match.groups()
        if key in NUMERIC_KEYS:
            summary[key] = _to_float(value)
    if "peak_vram_mb" in summary:
        summary["peak_vram_gb"] = round(summary["peak_vram_mb"] / 1024, 4)
    if "val_bpb" not in summary:
        summary["crash_tail"] = "\n".join(text.splitlines()[-20:])
    return summary


def autoresearch_records_to_episodes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    episodes = []
    best = None
    for idx, record in enumerate(records, 1):
        val_bpb = record.get("val_bpb") or 0.0
        status = record.get("status", "")
        improved = best is None or (val_bpb and val_bpb < best)
        if val_bpb:
            best = val_bpb if best is None else min(best, val_bpb)
        episodes.append({
            "dataset": "autoresearch",
            "task_id": "autoresearch_llm_training",
            "episode_id": record.get("commit") or f"autoresearch-{idx}",
            "global_step": idx,
            "proposed_action": record.get("description", ""),
            "outcome_score": round(1.0 / val_bpb, 4) if val_bpb else 0.0,
            "outcome_label": "failed" if status == "crash" else "improved" if improved or status == "keep" else "neutral",
            "failure_mode": "training_crash" if status == "crash" else None,
            "val_bpb": val_bpb,
            "memory_gb": record.get("memory_gb", 0.0),
            "status": status,
        })
    return episodes


def _to_float(value) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0
