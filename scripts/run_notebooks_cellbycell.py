#!/usr/bin/env python3
"""Execute the three phase notebooks cell-by-cell, capturing each code cell's stdout + wall time.

Produces:
 - logs/cellbycell_run.log (human-readable per-cell log)
 - logs/cellbycell_timings.json (per-cell timings, for timeline tuning)

The Phase 3 Modal cell is NOT re-run live (it costs ~14.5 min on an L4 and the result is already in
leaderboard/A.json); its real measured timing is logged instead, and the downstream aggregate cells
run against the restored group-A data. Everything else executes for real, in cell order, with a
shared namespace per notebook.
"""
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

DELAY = float(__import__("os").environ.get("CELL_DELAY", "3")) # seconds between cells (speech pacing)
LOG = ROOT / "logs" / "cellbycell_run.log"
TIMINGS = ROOT / "logs" / "cellbycell_timings.json"

# Real Phase-3 group-A Modal timing from the rehearsal run (start 19:08:39 -> end 19:23:16).
MODAL_CELL_SECONDS = 14 * 60 + 37


def run() -> None:
 LOG.parent.mkdir(parents=True, exist_ok=True)
 os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com/v1")
 # API key presence is checked by the Phase 1 setup cell; leave it to the env.

 all_timings: dict[str, list[dict]] = {}
 log_lines: list[str] = []
 log_lines.append(f"KDD tutorial — cell-by-cell run (cwd={ROOT})")
 log_lines.append("=" * 70)

 for nb_name in NOTEBOOKS:
 nb = json.loads((ROOT / nb_name).read_text())
 cells = nb["cells"]
 ns: dict = {"__name__": "__main__"}
 per_cell: list[dict] = []
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
 log_lines.append(f"[cell {cidx}] MODAL (skipped live re-run; real = {MODAL_CELL_SECONDS}s) :: {first}")
 # the cell sets GROUP and runs modal; emulate the side effect note
 with contextlib.redirect_stdout(buf):
 try:
 exec("GROUP='A'\nprint('group', GROUP, 'done -> leaderboard/A.json')", ns)
 except Exception as e:
 buf.write(f"[modal-cell emulate error] {e}\n")
 elapsed = MODAL_CELL_SECONDS
 status = "skipped(modal:real-timing)"
 else:
 with contextlib.redirect_stdout(buf):
 try:
 exec(compile(src, f"{nb_name}:cell{cidx}", "exec"), ns)
 status = "ok"
 except Exception as e:
 status = f"ERROR: {type(e).__name__}: {e}"
 buf.write(f"[cell {cidx} raised] {status}\n")
 elapsed = time.perf_counter() - t0
 out = buf.getvalue().rstrip()
 # Speech-pacing delay (simulates presenter explaining the cell)
 if DELAY > 0 and not is_modal:
 time.sleep(DELAY)
 per_cell.append({"cell": cidx, "first_line": first, "exec_seconds": round(elapsed, 3),
 "delay_seconds": DELAY if not is_modal else 0,
 "total_seconds": round(elapsed + (DELAY if not is_modal else 0), 3),
 "modal": is_modal, "status": status, "output": out[:1200]})
 log_lines.append(f"[cell {cidx}] exec={elapsed:5.2f}s +delay={DELAY if not is_modal else 0:.0f}s = {elapsed+DELAY if not is_modal else elapsed:5.2f}s {status} :: {first}")
 if out:
 for line in out.splitlines()[:25]:
 log_lines.append(f" | {line}")
 if len(out.splitlines()) > 25:
 log_lines.append(f" | ... ({len(out.splitlines())-25} more lines)")
 if status.startswith("ERROR"):
 log_lines.append(f" !! cell {cidx} errored (continuing to next cell)")
 all_timings[nb_name] = per_cell

 LOG.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
 TIMINGS.write_text(json.dumps(all_timings, indent=2), encoding="utf-8")

 # summary
 print("\n=== per-cell timing summary (exec + speech delay) ===")
 grand_exec = grand_delay = 0.0
 for nb_name, cells in all_timings.items():
 tot_exec = sum(c.get("exec_seconds", 0) for c in cells)
 tot_delay = sum(c.get("delay_seconds", 0) for c in cells)
 grand_exec += tot_exec; grand_delay += tot_delay
 print(f"{nb_name:52s} exec={tot_exec:6.1f}s +delay={tot_delay:5.1f}s = {tot_exec+tot_delay:6.1f}s ({len(cells)} cells)")
 for c in cells:
 tag = " (modal)" if c["modal"] else ""
 ex = c.get("exec_seconds", 0); dl = c.get("delay_seconds", 0)
 print(f" cell {c['cell']:>2}: exec={ex:5.2f}s +delay={dl:.0f}s = {ex+dl:5.2f}s{tag} [{c['status']}] {c['first_line'][:45]}")
 print(f"{'TOTAL':52s} exec={grand_exec:6.1f}s +delay={grand_delay:5.1f}s = {grand_exec+grand_delay:6.1f}s")
 print(f"\nlog -> {LOG}")
 print(f"timings -> {TIMINGS}")


if __name__ == "__main__":
 os.chdir(ROOT)
 run()
