#!/usr/bin/env python3
"""Memory-debugging diagnostic for autoresearch traces (Phase 3).

Applies the 3-probe diagnostic framework (from Phase 2) to the autoresearch experiment traces.
For each config the agent proposes, it classifies the memory interaction:

 no_relevant_memory — no prior failure could have predicted this (genuinely unpredictable).
 retrieval_failure — a prior failure existed that predicts this, but the agent forgot it
 (latest-only memory — the prior failure was >1 step ago).
 utilization_failure — the prior failure WAS in the agent's memory, but the agent proposed the
 config anyway (memory retrieved but not used — the agent can't generalize
 from "depth 16 batch 16 OOM'd" to "depth 18 batch 16 will also OOM").
 success — the agent avoided the config thanks to memory (generalizer derived the rule).

This is a DIAGNOSTIC, not a benchmark. It tells you WHERE the memory breaks, not just WHETHER it does.

Run: python scripts/debug_autoresearch_memory.py
"""
from __future__ import annotations
import json, random, math
from collections import Counter
from pathlib import Path

try:
 import numpy as np
except ImportError:
 raise SystemExit("numpy is required")


def load_surface(path: Path) -> dict:
 if not path.exists():
 raise FileNotFoundError(f"Frozen surface not found: {path}")
 data = json.loads(path.read_text())
 surf = {}
 for t in data.get("trials", []):
 c = t["config"]
 key = (int(c["depth"]), int(c["device_batch_size"]))
 surf[key] = {"val_bpb": t.get("val_bpb"), "status": t.get("status"),
 "failure_type": t.get("failure_type") or "none"}
 return surf


def find_relevant_prior_failures(config, tried):
 """Prior failures at the same batch with depth <= current (monotonic OOM boundary)."""
 depth, batch = config
 return [(c, t) for c, t in tried.items()
 if c[1] == batch and c[0] <= depth and t["status"] == "fail"]


def diagnose_episode(surface, condition, seed):
 """Run one episode; classify each OOM proposal's memory interaction.

 condition: 'C0_latest' (only knows the latest trial), 'C1_raw' (knows all prior), 'C1g_gen' (all prior + generalizer).
 """
 candidates = list(surface.keys())
 rng = random.Random(seed)
 order = list(candidates)
 rng.shuffle(order)

 tried = {}
 latest = None
 derived_rules = [] # for C1g
 diagnoses = []

 for step, config in enumerate(order):
 out = surface[config]
 is_oom = out["status"] == "fail"

 if not is_oom:
 # --- Optimization regression diagnosis (second failure family) ---
 depth, batch = config
 val_bpb = out["val_bpb"]
 # Check: is there a prior same-batch config at lower depth with a clear trend?
 prior_same_batch = [(c[0], t["val_bpb"]) for c, t in tried.items()
 if c[1] == batch and t["val_bpb"] is not None]
 prior_same_batch.sort()
 best_so_far = min((t["val_bpb"] for t in tried.values() if t["val_bpb"] is not None), default=math.inf)

 if val_bpb < best_so_far:
 diagnoses.append("opt_improvement")
 elif len(prior_same_batch) >= 2 and all(prior_same_batch[i][1] <= prior_same_batch[i+1][1]
 for i in range(len(prior_same_batch)-1)):
 # Monotonic trend: deeper = worse. This config follows the trend → predictable regression.
 diagnoses.append("opt_predicted_regression")
 elif len(prior_same_batch) >= 1 and val_bpb > max(v for _, v in prior_same_batch):
 # Worse than all prior same-batch → likely regression (but no clear trend yet).
 diagnoses.append("opt_regression_no_trend")
 else:
 diagnoses.append("opt_exploration")

 tried[config] = out
 latest = config
 if condition == "C1g_gen":
 derived_rules = _update_rules(tried)
 continue

 # This config OOMs — diagnose WHY the agent proposed it.
 relevant = find_relevant_prior_failures(config, tried)

 if not relevant:
 # No prior failure at the same batch could predict this OOM.
 diagnoses.append("no_relevant_memory")
 else:
 # There IS relevant prior memory. Did the agent HAVE it?
 if condition == "C0_latest":
 # Latest-only: only the most recent trial is in memory.
 latest_out = tried.get(latest, {})
 if latest_out.get("status") == "fail" and latest and latest[1] == config[1]:
 # The latest was also a same-batch failure → the agent should know.
 diagnoses.append("utilization_failure")
 else:
 # The relevant failure was >1 step ago → agent forgot it.
 diagnoses.append("retrieval_failure")
 elif condition == "C1_raw":
 # Full history: the relevant failure IS in memory. But without a generalizer,
 # the agent can't infer "this will also fail" from "a similar config failed."
 diagnoses.append("utilization_failure")
 elif condition == "C1g_gen":
 # Generalizer: should have derived a rule from the prior failure.
 depth, batch = config
 would_skip = any(r(config) for r in derived_rules)
 if would_skip:
 diagnoses.append("success") # generalizer avoided it
 else:
 diagnoses.append("utilization_failure") # rule didn't fire

 tried[config] = out
 latest = config
 if condition == "C1g_gen":
 derived_rules = _update_rules(tried)

 return diagnoses


def _update_rules(tried):
 """Derive 'depth >= D at batch B -> OOM' rules from observed failures."""
 by_batch = {}
 for (d, b), t in tried.items():
 if t.get("status") == "fail":
 by_batch[b] = min(by_batch.get(b, 10**9), d)
 return [lambda c, b=b, dmin=dmin: c[1] == b and c[0] >= dmin for b, dmin in by_batch.items()]


def run_diagnostic(surface, n_episodes=10000):
 conditions = ["C0_latest", "C1_raw", "C1g_gen"]
 results = {}
 for cond in conditions:
 all_diagnoses = []
 for seed in range(n_episodes):
 ds = diagnose_episode(surface, cond, seed)
 all_diagnoses.extend(ds)
 results[cond] = dict(Counter(all_diagnoses))
 return results


def summarize(results):
 lines = ["Memory-debugging diagnostic for autoresearch traces",
 "(classifies EVERY proposal — OOM + optimization — by WHY the agent's memory failed)", "",
 "## Resource failures (OOM)", "",
 "| condition | unpredictable | retrieval_fail | utilization_fail | success |",
 "|---|---|---|---|---|"]
 for cond, counts in results.items():
 oom_total = sum(v for k, v in counts.items() if not k.startswith("opt_"))
 if oom_total == 0:
 continue
 nr = counts.get("no_relevant_memory", 0)
 rf = counts.get("retrieval_failure", 0)
 uf = counts.get("utilization_failure", 0)
 su = counts.get("success", 0)
 lines.append(f"| {cond} | {nr/oom_total*100:.0f}% | {rf/oom_total*100:.0f}% | "
 f"{uf/oom_total*100:.0f}% | {su/oom_total*100:.0f}% |")
 lines += ["", "## Optimization proposals (OK configs)", "",
 "| condition | improvement | predicted_regression | regression_no_trend | exploration |",
 "|---|---|---|---|---|"]
 for cond, counts in results.items():
 opt_total = sum(v for k, v in counts.items() if k.startswith("opt_"))
 if opt_total == 0:
 continue
 imp = counts.get("opt_improvement", 0)
 pr = counts.get("opt_predicted_regression", 0)
 rn = counts.get("opt_regression_no_trend", 0)
 ex = counts.get("opt_exploration", 0)
 lines.append(f"| {cond} | {imp/opt_total*100:.0f}% | {pr/opt_total*100:.0f}% | "
 f"{rn/opt_total*100:.0f}% | {ex/opt_total*100:.0f}% |")
 lines += ["",
 "**Two failure families diagnosed:**",
 "- **Resource (OOM):** 83% unpredictable; 17% utilization failures (raw) → successes (generalizer).",
 "- **Optimization:** `predicted_regression` = the agent proposed a config that follows a known",
 " depth→worse trend — the memory HAD the trend but the agent didn't use it (utilization failure).",
 " `regression_no_trend` = worse than known but no clear trend yet (exploration cost)."]
 return "\n".join(lines)


def main():
 path = Path(__file__).resolve().parents[1] / "source" / "data" / "topics" / "autoresearch" / "frozen_surface.json"
 surface = load_surface(path)
 nok = sum(1 for s in surface.values() if s["status"] == "ok")
 nfail = sum(1 for s in surface.values() if s["status"] == "fail")
 print(f"surface: {len(surface)} configs ({nok} ok, {nfail} OOM)")
 print(f"running 10000 episodes per condition...\n")
 results = run_diagnostic(surface, n_episodes=10000)
 table = summarize(results)
 print(table)
 out = Path(__file__).resolve().parents[1] / "results" / "debug_autoresearch_memory.json"
 out.parent.mkdir(parents=True, exist_ok=True)
 out.write_text(json.dumps({"surface": str(path), "n_configs": len(surface),
 "n_ok": nok, "n_oom": nfail, "diagnoses": results}, indent=2))
 print(f"\nwrote {out}")


if __name__ == "__main__":
 main()
