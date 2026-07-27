#!/usr/bin/env python3
"""Experiment-history-memory benchmark (auto-review-loop Round 4 — final).

Three conditions isolate the failure-generalizer effect:
 C0 latest-only : only knows the latest trial -> CAN re-propose earlier (incl. known failures).
 C1 raw-no-gen : knows all tried (no repeats), no failure generalization.
 C1g raw+generalizer : knows all tried + derives "depth>=D at batch B -> OOM" rules.

NOTE: structured representation (C2) was dropped — C1g and C2 were implementation-identical
(not a null result). Representation benefit remains UNTESTED; only the generalizer effect is measured.

Primary metric: **failed_evaluations_avoided** (OOM configs NOT proposed — honest, no imputed constants).
Secondary: coverage (fraction of feasible configs found), conditional regret (for episodes that found
at least one feasible config), OOM count, duplicates.

All JSON output is strict (NaN -> null). 10k paired episodes; paired bootstrap on differences.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
try:
 import numpy as np
except ImportError:
 raise SystemExit("numpy is required")


def synthetic_surface():
 surf = {}
 for d in (2, 4, 6, 8, 10, 12, 14, 16):
 for b in (32, 16):
 oom = d >= 12 and b == 32
 if oom:
 surf[(d, b)] = {"val_bpb": None, "training_seconds": 120.0, "status": "fail", "failure_type": "oom"}
 else:
 base = {2: 1.35, 4: 1.259, 6: 1.303, 8: 1.443, 10: 1.670, 12: 1.78, 14: 1.85, 16: 1.95}[d]
 surf[(d, b)] = {"val_bpb": base + (0.02 if b == 16 else 0.0), "training_seconds": 300.0,
 "status": "ok", "failure_type": "none"}
 return surf


def load_surface(path: Path) -> dict:
 if not path.exists():
 print(f"(surface not found at {path}; using synthetic)")
 return synthetic_surface()
 data = json.loads(path.read_text())
 surf = {}
 for t in data.get("trials", []):
 c = t["config"]
 key = (int(c["depth"]), int(c["device_batch_size"]))
 secs = t.get("training_seconds")
 if secs is None and t.get("status") == "fail":
 secs = 120.0 # NOTE: imputed — not measured. See limitations.
 surf[key] = {"val_bpb": t.get("val_bpb"), "training_seconds": float(secs or 0.0),
 "status": t.get("status"), "failure_type": t.get("failure_type") or "none"}
 return surf or synthetic_surface()


def propose_order(candidates, seed):
 r = random.Random(seed)
 order = list(candidates)
 r.shuffle(order)
 return order


def c0_latest_only(order, tried, rules, latest=None):
 if latest is None:
 return list(order)
 return [c for c in order if c != latest]


def c1_raw_no_gen(order, tried, rules, latest=None):
 return [c for c in order if c not in tried]


def c1g_raw_with_gen(order, tried, rules, latest=None):
 return [c for c in order if c not in tried and not any(r(c) for r in rules)]


USES_RULES = {c1g_raw_with_gen}


def derive_rules(trials):
 by_batch_oom_depth = {}
 for (d, b), t in trials.items():
 if t.get("failure_type") == "oom" or t.get("status") == "fail":
 by_batch_oom_depth[b] = min(by_batch_oom_depth.get(b, 10**9), d)
 return [lambda c, b=b, dmin=dmin: c[1] == b and c[0] >= dmin for b, dmin in by_batch_oom_depth.items()]


def episode(surface, condition_fn, n_proposals, seed):
 candidates = list(surface.keys())
 order = propose_order(candidates, seed)
 ok_vals = [s["val_bpb"] for s in surface.values() if s["val_bpb"] is not None]
 global_best = min(ok_vals) if ok_vals else math.inf
 global_worst = max(ok_vals) if ok_vals else math.inf
 total_feasible = sum(1 for s in surface.values() if s["status"] == "ok")
 total_fail = sum(1 for s in surface.values() if s["status"] == "fail")

 tried = {}
 rules = []
 latest = None
 oom_count = 0
 duplicates = 0
 feasible_found = set()
 best_found = math.inf

 for step in range(n_proposals):
 pool = condition_fn(order, tried, rules if condition_fn in USES_RULES else [], latest)
 if not pool:
 pool = [c for c in order if c not in tried]
 if not pool:
 break
 pick = pool[0]
 out = surface[pick]
 if pick in tried:
 duplicates += 1
 if out["status"] == "fail":
 oom_count += 1
 else:
 feasible_found.add(pick)
 if out["val_bpb"] < best_found:
 best_found = out["val_bpb"]
 tried[pick] = out
 latest = pick # explicit chronological latest
 if condition_fn in USES_RULES:
 rules = derive_rules(tried)

 failed_avoided = max(0, total_fail - oom_count)
 coverage = len(feasible_found) / max(1, total_feasible)
 if best_found != math.inf:
 regret = best_found - global_best
 else:
 regret = None # no feasible config found — reported separately
 return {"failed_avoided": failed_avoided, "oom_count": oom_count,
 "duplicates": duplicates, "coverage": round(coverage, 4),
 "regret": regret, "wasted_min_imputed": (oom_count * 120.0) / 60.0}


CONDS = {"C0_latest_only": c0_latest_only, "C1_raw_no_gen": c1_raw_no_gen, "C1g_raw_with_gen": c1g_raw_with_gen}


def paired_bootstrap_diff(a, b, n_boot=5000, seed=0):
 a = [x for x in a if x is not None]
 b = [x for x in b if x is not None]
 n = min(len(a), len(b))
 if n == 0:
 return (None, None)
 diff = np.array(a[:n]) - np.array(b[:n])
 rng = np.random.default_rng(seed)
 means = [diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
 return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_perm_p(a, b, n_perm=10000, seed=0):
 a = [x for x in a if x is not None]
 b = [x for x in b if x is not None]
 n = min(len(a), len(b))
 if n == 0:
 return None
 diff = np.array(a[:n]) - np.array(b[:n])
 obs = abs(diff.mean())
 rng = np.random.default_rng(seed)
 flips = rng.integers(0, 2, (n_perm, n)) * 2 - 1
 return float((abs((diff * flips).mean(axis=1)) >= obs).mean())


def run(surface, n_episodes=10000, n_proposals=8, seed0=0):
 results = {}
 for name, fn in CONDS.items():
 eps = [episode(surface, fn, n_proposals, seed0 + i) for i in range(n_episodes)]
 results[name] = {m: [e[m] for e in eps] for m in eps[0]}
 return results


def _sanitize(obj):
 if isinstance(obj, float) and math.isnan(obj):
 return None
 if isinstance(obj, dict):
 return {k: _sanitize(v) for k, v in obj.items()}
 if isinstance(obj, list):
 return [_sanitize(v) for v in obj]
 return obj


def summarize(results):
 lines = ["condition\tfailed_avoided\tcoverage\tregret(cond.)\toom\tdup\twasted_min(imputed)"]
 for name, m in results.items():
 n = len(m["failed_avoided"])
 reg = [r for r in m["regret"] if r is not None]
 lines.append(f"{name}\t{sum(m['failed_avoided'])/n:.2f}\t{sum(m['coverage'])/n:.3f}\t"
 f"{sum(reg)/max(1,len(reg)):.4f} (n={len(reg)})\t{sum(m['oom_count'])/n:.1f}\t"
 f"{sum(m['duplicates'])/n:.1f}\t{sum(m['wasted_min_imputed'])/n:.2f}")
 for ref, cmp, label in [("C1_raw_no_gen", "C1g_raw_with_gen", "GENERALIZER EFFECT"),
 ("C0_latest_only", "C1_raw_no_gen", "HISTORY EFFECT")]:
 for metric in ["failed_avoided", "coverage", "regret"]:
 a, b = results[ref][metric], results[cmp][metric]
 p = paired_perm_p(a, b)
 dlo, dhi = paired_bootstrap_diff(b, a) if p is not None else (None, None)
 lines.append(f"# {label} [{metric}]: {cmp} vs {ref} "
 f"p={'%.4f' % p if p is not None else 'N/A'} "
 f"CI=[{'%.4f' % dlo if dlo else 'N/A'},{'%.4f' % dhi if dhi else 'N/A'}]")
 return "\n".join(lines)


def main():
 ap = argparse.ArgumentParser()
 default = Path(__file__).resolve().parents[1] / "source" / "data" / "topics" / "autoresearch" / "frozen_surface.json"
 ap.add_argument("--surface", default=str(default))
 ap.add_argument("--episodes", type=int, default=10000)
 ap.add_argument("--proposals", type=int, default=8)
 ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "experiment_history_memory.json"))
 args = ap.parse_args()
 surface = load_surface(Path(args.surface))
 nok = sum(1 for s in surface.values() if s["status"] == "ok")
 nfail = sum(1 for s in surface.values() if s["status"] == "fail")
 print(f"surface: {len(surface)} configs ({nok} ok, {nfail} fail)")
 res = run(surface, n_episodes=args.episodes, n_proposals=args.proposals)
 table = summarize(res)
 print("\n" + table)
 Path(args.out).parent.mkdir(parents=True, exist_ok=True)
 payload = _sanitize({"surface_path": args.surface, "n_configs": len(surface),
 "n_episodes": args.episodes, "n_proposals": args.proposals, "results": res,
 "note": "wasted_min_imputed uses 120s/fail (NOT measured). "
 "regret is conditional (episodes with no feasible config excluded). "
 "C2 dropped (representation not genuinely tested)."})
 Path(args.out).write_text(json.dumps(payload, indent=2, allow_nan=False))
 print(f"\nwrote {args.out}")


if __name__ == "__main__":
 main()
