"""benchmark-cli CLI — maps configure/run/compare/serve onto run.py."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_BENCH_TEMPLATE = """# KDD tutorial benchmark config
providers: [verbatim, extracted_facts, episodic, hybrid]
benchmark: [locomo, longmemeval, memoryarena]
evaluator: offline
metrics: [accuracy, latency, cost]
sample_size: 20
"""


def _dispatch(run_argv: list[str]) -> int:
    sys.argv = ["run.py", *run_argv]
    import run
    run.main()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="benchmark-cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cfg = sub.add_parser("configure")
    p_cfg.add_argument("--providers", default="verbatim,extracted_facts,episodic,hybrid")
    p_cfg.add_argument("--benchmark", default="locomo,longmemeval,memoryarena")
    p_cfg.add_argument("--evaluator", default="offline")

    p_run = sub.add_parser("run")
    p_run.add_argument("--provider", required=True)
    p_run.add_argument("--sample_size", type=int, default=20)
    p_run.add_argument("--benchmark", default="locomo,longmemeval,memoryarena")
    p_run.add_argument("--backend", default="offline")

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--providers", default="verbatim,extracted_facts,episodic,hybrid")
    p_cmp.add_argument("--sample_size", type=int, default=10)
    p_cmp.add_argument("--metrics", default="accuracy,latency,cost")
    p_cmp.add_argument("--benchmark", default="locomo,longmemeval,memoryarena")
    p_cmp.add_argument("--backend", default="offline")

    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if args.command == "configure":
        Path("benchmark_config.yaml").write_text(_BENCH_TEMPLATE, encoding="utf-8")
        print("Wrote benchmark_config.yaml")
        return 0

    if args.command == "run":
        datasets = [d for d in args.benchmark.split(",") if d]
        return _dispatch([
            "--mode", "real", "--backend", args.backend,
            "--strategies", args.provider,
            "--datasets", *datasets,
            "--max-items", str(args.sample_size),
            "--top-k", "5", "--eval-backend", "offline", "--visualize",
        ])

    if args.command == "compare":
        providers = [p for p in args.providers.split(",") if p]
        datasets = [d for d in args.benchmark.split(",") if d]
        return _dispatch([
            "--mode", "real", "--backend", args.backend,
            "--strategies", *providers,
            "--datasets", *datasets,
            "--max-items", str(args.sample_size),
            "--top-k", "5", "--eval-backend", "offline", "--visualize",
        ])

    if args.command == "serve":
        results_dir = Path("results")
        if not results_dir.exists():
            print("No results/ directory. Run `compare` first.")
            return 1
        print(f"Serving results dashboard at http://localhost:{args.port}/ (Ctrl+C to stop)")
        import http.server
        import functools
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(results_dir))
        with __import__("socketserver").TCPServer(("", args.port), handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nStopped.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
