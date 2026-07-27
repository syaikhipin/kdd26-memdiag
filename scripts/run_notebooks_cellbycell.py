#!/usr/bin/env python3
"""Execute the phase notebooks cell-by-cell, capturing each cell's stdout + wall time + speech delay."""
import json
import os
import sys
import time
import io
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "tutorial/phase1_memory_architectures/01_short_term_memory.ipynb",
    "tutorial/phase1_memory_architectures/02_long_term_memory.ipynb",
    "tutorial/phase1_memory_architectures/03_cognitive_architectures.ipynb",
    "tutorial/phase1_memory_architectures/04_retrieval_multi_agent.ipynb",
    "tutorial/phase1_memory_architectures/05_frameworks.ipynb",
    "tutorial/phase1_memory_architectures/06_evaluation_production.ipynb",
    "tutorial/phase1_memory_architectures/07_capstone.ipynb",
    "tutorial/phase1_memory_architectures/08_exercise.ipynb",
    "tutorial/phase2_public_datasets/01_diagnostic_framework.ipynb",
    "tutorial/phase2_public_datasets/02_utilization_lab.ipynb",
    "tutorial/phase2_public_datasets/03_benchmarking.ipynb",
    "tutorial/phase2_public_datasets/04_llm_as_judge_lab.ipynb",
    "tutorial/phase2_public_datasets/05_question_type_analysis.ipynb",
    "tutorial/phase3_autoresearch/01_autoresearch_loop.ipynb",
    "tutorial/phase3_autoresearch/02_run_your_group.ipynb",
    "tutorial/phase3_autoresearch/03_aggregate_and_debug.ipynb",
    "tutorial/phase3_autoresearch/04_memory_diagnostic.ipynb",
]
LOG = ROOT / "logs" / "cellbycell_run.log"
TIMINGS = ROOT / "logs" / "cellbycell_timings.json"
DELAY = float(os.environ.get("CELL_DELAY", "20"))
MODAL_CELL_SECONDS = 14 * 60 + 37


def run():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
    all_timings = {}
    log_lines = [f"KDD tutorial — cell-by-cell run (cwd={ROOT}, delay={DELAY}s)"]
    log_lines.append("=" * 70)

    for nb_name in NOTEBOOKS:
        nb = json.loads((ROOT / nb_name).read_text())
        cells = nb["cells"]
        ns = {"__name__": "__main__"}
        per_cell = []
        log_lines.append("")
        log_lines.append(f"### {nb_name} ({sum(1 for c in cells if c['cell_type']=='code')} code cells)")
        log_lines.append("-" * 70)
        cidx = 0
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            cidx += 1
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            first = src.strip().splitlines()[0][:70] if src.strip() else "(empty)"
            is_modal = ("modal" in src.lower() and "build_autoresearch_trace" in src)
            buf = io.StringIO()
            t0 = time.perf_counter()
            if is_modal:
                log_lines.append(f"[cell {cidx}] MODAL (skipped; real = {MODAL_CELL_SECONDS}s) :: {first}")
                with contextlib.redirect_stdout(buf):
                    try:
                        exec("GROUP='A'\nprint('group', GROUP, 'done')", ns)
                    except Exception:
                        pass
                elapsed = MODAL_CELL_SECONDS
                delay_used = 0
                status = "skipped(modal)"
            else:
                with contextlib.redirect_stdout(buf):
                    try:
                        exec(compile(src, f"{nb_name}:cell{cidx}", "exec"), ns)
                        status = "ok"
                    except Exception as e:
                        status = f"ERROR: {type(e).__name__}: {e}"
                        buf.write(f"[cell {cidx} raised] {status}\n")
                elapsed = time.perf_counter() - t0
                if DELAY > 0:
                    time.sleep(DELAY)
                delay_used = DELAY
            out = buf.getvalue().rstrip()
            total = elapsed + delay_used
            per_cell.append({"cell": cidx, "first_line": first,
                             "exec_seconds": round(elapsed, 3),
                             "delay_seconds": delay_used,
                             "total_seconds": round(total, 3),
                             "modal": is_modal, "status": status,
                             "output": out[:1200]})
            log_lines.append(f"[cell {cidx}] exec={elapsed:5.2f}s +delay={delay_used:.0f}s = {total:6.2f}s  {status}  ::  {first}")
            if out:
                for line in out.splitlines()[:20]:
                    log_lines.append(f"        | {line}")
            if status.startswith("ERROR"):
                log_lines.append(f"        !! cell {cidx} errored (continuing)")
        all_timings[nb_name] = per_cell

    LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    TIMINGS.write_text(json.dumps(all_timings, indent=2), encoding="utf-8")

    print("\n=== per-cell timing summary (exec + speech delay) ===")
    ge = gd = 0
    for nb_name, cells in all_timings.items():
        te = sum(c.get("exec_seconds", 0) for c in cells)
        td = sum(c.get("delay_seconds", 0) for c in cells)
        ge += te; gd += td
        short = nb_name.split("/")[-1].replace(".ipynb", "")
        print(f"  {short:54s} exec={te:6.1f}s +delay={td:5.0f}s = {te+td:6.1f}s ({len(cells)} cells)")
    print(f"  {'TOTAL':54s} exec={ge:6.1f}s +delay={gd:5.0f}s = {ge+gd:6.1f}s")
    print(f"\nlog -> {LOG}")
    print(f"timings -> {TIMINGS}")


if __name__ == "__main__":
    os.chdir(ROOT)
    run()
