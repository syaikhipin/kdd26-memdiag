"""diagnostic_framework CLI — maps init/load/diagnose/analyze onto run.py."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_CONFIG_TEMPLATE = """# KDD tutorial diagnostic framework config
strategies: [no_memory, verbatim, extracted_facts, episodic, hybrid]
probes: [relevance, utilization, failure]
top_k: 5
backend: offline
datasets: [locomo, longmemeval, memoryarena]
max_questions: 40
"""


def _dispatch(run_argv: list[str]) -> int:
    sys.argv = ["run.py", *run_argv]
    import run
    run.main()
    return 0


def _latest_metrics(out_dir: Path) -> Path | None:
    paths = sorted(out_dir.glob("run_*_metrics.json"))
    return paths[-1] if paths else None


def main() -> int:
    parser = argparse.ArgumentParser(prog="diagnostic_framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--config", default="research_agent_config.yaml")

    p_load = sub.add_parser("load")
    p_load.add_argument("--input", default=None)
    p_load.add_argument("--episodes", type=int, default=20)

    p_diag = sub.add_parser("diagnose")
    p_diag.add_argument("--strategies", default="verbatim,extracted_facts,episodic,hybrid")
    p_diag.add_argument("--probes", default="relevance,utilization,failure")
    p_diag.add_argument("--top_k", type=int, default=5)
    p_diag.add_argument("--backend", default="offline")
    p_diag.add_argument("--max-questions", type=int, default=40)

    p_an = sub.add_parser("analyze")
    p_an.add_argument("--visualizations", default="all")
    p_an.add_argument("--report", default=None)
    p_an.add_argument("--metrics", default=None)

    args = parser.parse_args()

    if args.command == "init":
        Path(args.config).write_text(_CONFIG_TEMPLATE, encoding="utf-8")
        print(f"Wrote config template -> {args.config}")
        return 0

    if args.command == "load":
        return _dispatch(["--mode", "synthetic", "--backend", "offline", "--episodes", str(args.episodes)])

    if args.command == "diagnose":
        strategies = [s for s in args.strategies.split(",") if s]
        return _dispatch([
            "--mode", "locomo", "--backend", args.backend,
            "--strategies", *strategies, "--top-k", str(args.top_k),
            "--max-questions", str(args.max_questions), "--visualize",
        ])

    if args.command == "analyze":
        out_dir = Path("results")
        metrics = Path(args.metrics) if args.metrics else _latest_metrics(out_dir)
        if not metrics or not Path(metrics).exists():
            print("No metrics file found. Run `diagnose` first or pass --metrics.")
            return 1
        return _dispatch(["--mode", "visualize", "--metrics", str(metrics)])

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
