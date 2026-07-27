#!/usr/bin/env python3
"""Phase 3 memory diagnostic v3 — fundamental methodology redesign.

Unit of analysis: a DECISION POINT (frozen pre-decision snapshot), not an episode.

For each OOM config C and each possible pre-decision history H:
 - FREEZE the state: history H + candidates (all untried configs including C).
 - Run 5 memory conditions × K=5 repeats, all from the SAME frozen state.
 - Metric: proposal rate for C (fraction of K calls that proposed the OOM config).

Memory conditions (genuine interventions, not labels):
 M0 no-memory — empty history (placebo baseline).
 M1 raw-history — full trial log (what upstream autoresearch provides).
 M2 retrieved — only same-batch trials (simulates batch-keyed retrieval).
 M3 structured-rule — raw history + inferred constraint ("depth >= D at batch B -> OOM").
 M4 oracle — raw history + actual outcome ("this config WILL OOM").

Key comparisons:
 M0 vs M1: does ANY history help? (information effect)
 M1 vs M2: does retrieval filtering help? (retrieval effect)
 M1 vs M3: does structured representation help? (representation effect)
 M3 vs M4: does the constraint match the oracle? (constraint quality)

VRAM model: bilinear (depth + batch + interaction), fitted from pre-decision OK configs only.
Evaluated: false-positive rate (predicts OOM for OK) + false-negative (predicts OK for OOM).
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
from collections import defaultdict

SURFACE_PATH = Path(__file__).resolve().parents[1] / "source" / "data" / "topics" / "autoresearch" / "frozen_surface.json"
GPU_LIMIT_MB = 22 * 1024
K_REPEATS = 5


def load_surface():
 data = json.loads(SURFACE_PATH.read_text())
 surf = {}
 for t in data.get("trials", []):
 c = t["config"]
 key = (int(c["depth"]), int(c["device_batch_size"]))
 surf[key] = {"val_bpb": t.get("val_bpb"), "status": t.get("status"),
 "failure_type": t.get("failure_type") or "none",
 "peak_vram_mb": t.get("peak_vram_mb")}
 return surf


def generate_decision_points(surface):
 """For each OOM config, generate decision points with varying pre-decision histories.
 Includes prior same-batch OOMs in the history (enables genuine M3 structured-rule intervention)."""
 oom_configs = sorted([c for c, t in surface.items() if t["status"] == "fail"])
 ok_configs = sorted([c for c, t in surface.items() if t["status"] == "ok"])
 seen_states = set()
 points = []
 for oom in oom_configs:
 batch = oom[1]
 same_batch_oks = sorted([c for c in ok_configs if c[1] == batch and c[0] < oom[0]])
 prior_ooms = sorted([c for c in oom_configs if c[1] == batch and c[0] < oom[0]])
 for n in range(1, len(same_batch_oks) + 1):
 history_keys = list(same_batch_oks[:n])
 if prior_ooms:
 history_keys += prior_ooms # include prior OOMs → M3 can derive constraints
 cross_batch = [c for c in ok_configs if c[1] != batch][:2]
 full_history = sorted(set(history_keys + cross_batch))
 candidates = sorted([c for c in surface if c not in full_history])
 # Deduplicate by (history, candidates) signature
 sig = (tuple(full_history), tuple(candidates))
 if sig in seen_states:
 continue
 seen_states.add(sig)
 points.append({"target": oom, "history_keys": full_history, "candidates": candidates,
 "n_same_batch_oks": n, "has_prior_oom": bool(prior_ooms)})
 return points


def fit_vram_bilinear(history, surface):
 """VRAM ~ a*depth + b*batch + c*depth*batch + d, from OK configs in history."""
 try:
 import numpy as np
 xs, ys = [], []
 for c in history:
 t = surface[c]
 if t["status"] == "ok" and t.get("peak_vram_mb"):
 xs.append([c[0], c[1], c[0] * c[1], 1.0])
 ys.append(t["peak_vram_mb"])
 if len(xs) < 3:
 return None, None
 X, Y = np.array(xs), np.array(ys)
 coeffs, res, _, _ = np.linalg.lstsq(X, Y, rcond=None)
 def predict(depth, batch):
 return float(coeffs[0] * depth + coeffs[1] * batch + coeffs[2] * depth * batch + coeffs[3])
 # Evaluate on the surface's OK and OOM configs
 fp = sum(1 for c, t in surface.items()
 if t["status"] == "ok" and predict(c[0], c[1]) > GPU_LIMIT_MB)
 fn = sum(1 for c, t in surface.items()
 if t["status"] == "fail" and predict(c[0], c[1]) <= GPU_LIMIT_MB)
 n_ok = sum(1 for t in surface.values() if t["status"] == "ok")
 n_fail = sum(1 for t in surface.values() if t["status"] == "fail")
 eval_metrics = {"false_positive_rate": fp / max(1, n_ok),
 "false_negative_rate": fn / max(1, n_fail)}
 return predict, eval_metrics
 except Exception:
 return None, None


def make_client():
 from openai import OpenAI
 key = os.environ.get("OPENAI_API_KEY", "")
 base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
 if not key:
 raise SystemExit("Set OPENAI_API_KEY")
 return OpenAI(api_key=key, base_url=base)


SYS_PROMPT = ("You are an ML research agent optimizing a GPT pretraining loop. "
 "Goal: minimize val_bpb (lower is better). 5-min budget per config. GPU: NVIDIA L4 (22 GB).\n"
 "Reply with ONLY JSON: {\"depth\": <int>, \"batch_size\": <int>}")


def ask(client, model, candidates_str, history_str, treatment=""):
 user = f"Available configs: {candidates_str}\n\nExperiment history:\n{history_str}\n"
 if treatment:
 user += f"\n{treatment}\n"
 user += "\nPropose the next config. Reply with ONLY the JSON."
 raw = ""
 try:
 resp = client.chat.completions.create(
 model=model,
 messages=[{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user}],
 max_tokens=60, temperature=0.3)
 raw = resp.choices[0].message.content.strip()
 for pat in [r'\{[^}]*"depth"\s*:\s*(\d+)[^}]*"batch_size"\s*:\s*(\d+)[^}]*\}',
 r'"depth"\s*:\s*(\d+).*?"batch_size"\s*:\s*(\d+)', ]:
 m = re.search(pat, raw, re.DOTALL)
 if m:
 return (int(m.group(1)), int(m.group(2))), raw
 except Exception as e:
 raw = f"[error: {e}]"
 return None, raw


def build_treatments(point, surface):
 """Build the 5 condition texts from the SAME frozen pre-decision state."""
 hist = point["history_keys"]
 target = point["target"]
 candidates = point["candidates"]
 cands_str = str([list(c) for c in candidates])

 # History lines
 raw_lines = []
 for c in hist:
 t = surface[c]
 if t["status"] == "fail":
 raw_lines.append(f"- depth={c[0]}, batch={c[1]} -> FAILED (CUDA OOM)")
 else:
 raw_lines.append(f"- depth={c[0]}, batch={c[1]} -> val_bpb={t['val_bpb']:.3f}")
 raw_history = "\n".join(raw_lines) if raw_lines else "(none)"

 # Same-batch only (M2)
 same_batch = [l for l in raw_lines if f"batch={target[1]}" in l]
 retrieved_history = "\n".join(same_batch) if same_batch else "(none at this batch)"

 # Inferred constraint (M3)
 prior_fails = [c for c in hist if surface[c]["status"] == "fail" and c[1] == target[1]]
 if prior_fails:
 min_fail_depth = min(c[0] for c in prior_fails)
 constraint = (f"INFERRED CONSTRAINT: based on prior results, depth >= {min_fail_depth} "
 f"at batch {target[1]} causes CUDA OOM. Do NOT propose such configs.")
 else:
 constraint = "(no prior failures to infer a constraint from)"

 # Oracle (M4)
 oracle = f"KNOWN OUTCOME: depth={target[0]} at batch={target[1]} WILL CUDA OOM. Do NOT propose it."

 return {
 "M0_no_memory": ("(no prior trial information)", cands_str),
 "M1_raw_history": (raw_history, cands_str),
 "M2_retrieved": (retrieved_history, cands_str),
 "M3_structured": (raw_history + "\n\n" + constraint, cands_str),
 "M4_oracle": (raw_history + "\n\n" + oracle, cands_str),
 }


def main():
 surface = load_surface()
 model = os.environ.get("MODEL", "gpt-4o")
 client = make_client()
 points = generate_decision_points(surface)
 print(f"surface: {len(surface)} configs | decision points: {len(points)} | model: {model} | K={K_REPEATS}\n")

 all_results = []
 for pi, pt in enumerate(points):
 target = pt["target"]
 hist = pt["history_keys"]
 treatments = build_treatments(pt, surface)
 vram_pred, vram_eval = fit_vram_bilinear(hist, surface)
 vram_target = vram_pred(target[0], target[1]) if vram_pred else None

 print(f"[{pi+1}/{len(points)}] target={target}, history={len(hist)} configs, "
 f"VRAM_pred={vram_target:.0f}MB" if vram_target else
 f"[{pi+1}/{len(points)}] target={target}, history={len(hist)}, no VRAM model")

 for cond_name, (history_text, cands_str) in treatments.items():
 proposals = []
 raws = []
 for k in range(K_REPEATS):
 prop, raw = ask(client, model, cands_str, history_text)
 proposals.append(list(prop) if prop else None)
 raws.append(raw[:300])
 target_rate = sum(1 for p in proposals if p and tuple(p) == target) / K_REPEATS
 any_oom_rate = sum(1 for p in proposals if p and tuple(p) in surface
 and surface[tuple(p)]["status"] == "fail") / K_REPEATS
 all_results.append({
 "point_idx": pi, "target": list(target), "condition": cond_name,
 "n_history": len(hist), "vram_predicted_mb": round(vram_target) if vram_target else None,
 "vram_eval": vram_eval,
 "target_proposal_rate": round(target_rate, 2),
 "any_oom_proposal_rate": round(any_oom_rate, 2),
 "proposals": proposals,
 "raw_responses": raws,
 })
 print(f" {cond_name:20s}: target_rate={target_rate:.0%} any_oom={any_oom_rate:.0%}")

 # Aggregate by condition
 print(f"\n{'='*70}")
 print("AGGREGATE (mean target-proposal rate across decision points)")
 print(f"{'='*70}\n")
 conds = ["M0_no_memory", "M1_raw_history", "M2_retrieved", "M3_structured", "M4_oracle"]
 print(f"| condition | mean target_rate | mean any_oom_rate |")
 print(f"|---|---|---|")
 for c in conds:
 recs = [r for r in all_results if r["condition"] == c]
 if recs:
 mt = sum(r["target_proposal_rate"] for r in recs) / len(recs)
 mo = sum(r["any_oom_proposal_rate"] for r in recs) / len(recs)
 print(f"| {c} | {mt:.2f} | {mo:.2f} |")

 print(f"\nKey comparisons:")
 pairs = [("M0_no_memory", "M1_raw_history", "information effect"),
 ("M1_raw_history", "M2_retrieved", "retrieval effect"),
 ("M1_raw_history", "M3_structured", "representation effect"),
 ("M3_structured", "M4_oracle", "constraint quality gap")]
 for a, b, label in pairs:
 ra = [r for r in all_results if r["condition"] == a]
 rb = [r for r in all_results if r["condition"] == b]
 if ra and rb:
 ma = sum(r["target_proposal_rate"] for r in ra) / len(ra)
 mb = sum(r["target_proposal_rate"] for r in rb) / len(rb)
 print(f" {label}: {a} ({ma:.2f}) vs {b} ({mb:.2f}) → delta={mb-ma:+.2f}")

 # VRAM evaluation
 vram_evals = [r["vram_eval"] for r in all_results if r.get("vram_eval")]
 if vram_evals:
 avg_fp = sum(v["false_positive_rate"] for v in vram_evals) / len(vram_evals)
 avg_fn = sum(v["false_negative_rate"] for v in vram_evals) / len(vram_evals)
 print(f"\nVRAM model: avg false-positive={avg_fp:.2f}, avg false-negative={avg_fn:.2f}")

 out = Path(__file__).resolve().parents[1] / "results" / "phase3_diagnostic_v3.json"
 out.parent.mkdir(parents=True, exist_ok=True)
 out.write_text(json.dumps({"model": model, "n_points": len(points), "K": K_REPEATS,
 "results": all_results}, indent=2, default=str))
 print(f"\nwrote {out}")


if __name__ == "__main__":
 main()
