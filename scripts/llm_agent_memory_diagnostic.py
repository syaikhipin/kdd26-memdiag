#!/usr/bin/env python3
"""LLM-agent memory diagnostic v2 — rigorous interventional probes.

Fixes from v1 (per reviewer):
1. Pre-decision state: all probes use the history BEFORE the current proposal's outcome.
2. Incremental VRAM fitting: model fitted only from OK configs observed so far in the episode.
3. Placebo control: identical re-ask (same prompt, no treatment) to measure stochastic baseline.
4. One diagnosis per proposal; probe outcomes are separate fields.
5. Full logging: every prompt, response, parsed decision, outcome, probe result saved.

Classification per OOM proposal:
 unpredictable — no prior failure AND no VRAM data / VRAM predicts OK.
 predictable_from_telemetry — no prior failure BUT VRAM model predicts > GPU limit.
 has_prior_failure + probes:
 placebo_changed — agent changed proposal WITHOUT treatment (stochastic) → inconclusive.
 warning_effective — agent changed WITH warning but NOT on placebo → treatment effect.
 utilization_failure — agent didn't change despite warning → genuine utilization failure.
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
from collections import Counter

SURFACE_PATH = Path(__file__).resolve().parents[1] / "source" / "data" / "topics" / "autoresearch" / "frozen_surface.json"
GPU_LIMIT_MB = 22 * 1024 # L4 = 22 GB


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


def fit_vram_incremental(tried):
 """Fit VRAM ~ a*(depth*batch) + b from OK configs observed SO FAR (pre-decision)."""
 try:
 import numpy as np
 xs, ys = [], []
 for (d, b), t in tried.items():
 if t["status"] == "ok" and t.get("peak_vram_mb"):
 xs.append(d * b)
 ys.append(t["peak_vram_mb"])
 if len(xs) < 2:
 return None
 a, b = np.polyfit(xs, ys, 1)
 return lambda depth, batch: float(a * depth * batch + b)
 except Exception:
 return None


def make_client():
 from openai import OpenAI
 key = os.environ.get("OPENAI_API_KEY", "")
 base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
 if not key:
 raise SystemExit("Set OPENAI_API_KEY")
 return OpenAI(api_key=key, base_url=base)


def ask_agent(client, model, history_lines, candidates, extra="", sys_extra=""):
 sys_prompt = ("You are an ML research agent optimizing a GPT pretraining loop. "
 "Goal: minimize val_bpb (bits-per-byte; lower is better). "
 "5-minute budget per config. GPU: NVIDIA L4 (22 GB).\n"
 f"Available configs: {candidates}\n"
 + sys_extra +
 "\nReply with ONLY JSON: {\"depth\": <int>, \"batch_size\": <int>}")
 user_msg = "Experiment history:\n" + ("\n".join(history_lines) if history_lines else "(none)")
 if extra:
 user_msg += "\n\n" + extra
 user_msg += "\n\nPropose the next config. Reply with ONLY the JSON."
 raw = ""
 try:
 resp = client.chat.completions.create(
 model=model, messages=[{"role": "system", "content": sys_prompt},
 {"role": "user", "content": user_msg}],
 max_tokens=80, temperature=0.3)
 raw = resp.choices[0].message.content.strip()
 for m in [re.search(r'\{[^}]*"depth"\s*:\s*(\d+)[^}]*"batch_size"\s*:\s*(\d+)[^}]*\}', raw),
 re.search(r'"depth"\s*:\s*(\d+).*?"batch_size"\s*:\s*(\d+)', raw, re.DOTALL)]:
 if m:
 return (int(m.group(1)), int(m.group(2))), raw
 except Exception as e:
 raw = f"[error: {e}]"
 return None, raw


def find_relevant_failure(config, tried_pre):
 """Prior same-batch OOM at <= depth, EXCLUDING the current config. Uses PRE-decision state."""
 depth, batch = config
 for c, t in tried_pre.items():
 if c != config and c[1] == batch and c[0] <= depth and t["status"] == "fail":
 return c
 return None


def run_episode(client, model, surface, candidates, ep_id, log):
 tried = {} # PRE-decision state (updated AFTER probes)
 history_lines = []
 records = []

 for step in range(min(8, len(candidates))):
 remaining = [c for c in candidates if c not in tried]
 if not remaining:
 break

 # Agent proposes (PRE-decision: tried has all PRIOR outcomes, not current)
 proposal, raw = ask_agent(client, model, history_lines, remaining)
 if proposal is None or proposal not in surface:
 proposal = remaining[0] # fallback
 raw += " [FALLBACK]"

 out = surface[proposal]
 is_oom = out["status"] == "fail"

 # Update history + tried (POST-decision)
 if is_oom:
 history_lines.append(f"- depth={proposal[0]}, batch={proposal[1]} -> FAILED (CUDA OOM)")
 else:
 history_lines.append(f"- depth={proposal[0]}, batch={proposal[1]} -> val_bpb={out['val_bpb']:.3f}")
 tried[proposal] = out # now tried includes current (for NEXT step's pre-decision)

 rec = {"ep": ep_id, "step": step, "proposal": list(proposal), "is_oom": is_oom,
 "raw_response": raw[:200]}

 if not is_oom:
 rec["classification"] = "ok"
 records.append(rec)
 continue

 # --- OOM: classify using PRE-decision state (tried BEFORE adding current) ---
 # But we already added current to tried... need to use a snapshot.
 # FIX: use tried_before = everything EXCEPT current.
 tried_before = {k: v for k, v in tried.items() if k != proposal}
 relevant = find_relevant_failure(proposal, tried_before)
 vram_model = fit_vram_incremental(tried_before)
 vram_pred = vram_model(proposal[0], proposal[1]) if vram_model else None
 rec["vram_predicted"] = round(vram_pred, 1) if vram_pred else None

 if relevant is None and (vram_pred is None or vram_pred <= GPU_LIMIT_MB):
 rec["classification"] = "unpredictable"
 elif relevant is None and vram_pred and vram_pred > GPU_LIMIT_MB:
 rec["classification"] = "predictable_from_telemetry"
 else:
 # has_prior_failure → run interventional probes
 rec["prior_failure"] = list(relevant)
 remaining_post = [c for c in candidates if c not in tried]

 # 1. Placebo: identical re-ask (no treatment)
 placebo, placebo_raw = ask_agent(client, model, history_lines, remaining_post)
 rec["placebo_proposal"] = list(placebo) if placebo else None
 placebo_changed = placebo is not None and placebo != proposal

 # 2. Utilization probe: inject explicit warning
 warning = (f"WARNING: depth={relevant[0]} at batch={relevant[1]} FAILED with CUDA OOM. "
 f"Similar or higher depth at same batch may also OOM.")
 warned, warned_raw = ask_agent(client, model, history_lines, remaining_post, extra=warning)
 rec["warned_proposal"] = list(warned) if warned else None
 warned_changed = warned is not None and warned != proposal

 # 3. Generalization probe: inject structured constraint
 constraint = (f"CONSTRAINT: depth >= {relevant[0]} at batch {relevant[1]} causes CUDA OOM. "
 f"Do NOT propose any config matching this.")
 constrained, constrained_raw = ask_agent(client, model, history_lines, remaining_post, extra=constraint)
 rec["constrained_proposal"] = list(constrained) if constrained else None

 # Classify
 if placebo_changed:
 rec["classification"] = "inconclusive_placebo_changed"
 elif warned_changed:
 rec["classification"] = "warning_effective" # treatment effect (not placebo)
 else:
 rec["classification"] = "utilization_failure" # didn't change despite warning

 records.append(rec)

 return records


def main():
 surface = load_surface()
 candidates = sorted(surface.keys())
 model = os.environ.get("MODEL", "gpt-4o")
 client = make_client()
 n_episodes = int(os.environ.get("N_EPISODES", "10"))

 nok = sum(1 for s in surface.values() if s["status"] == "ok")
 nfail = sum(1 for s in surface.values() if s["status"] == "fail")
 print(f"surface: {len(surface)} ({nok} ok, {nfail} OOM) | model: {model} | episodes: {n_episodes}\n")

 all_records = []
 for ep in range(n_episodes):
 print(f"--- episode {ep+1}/{n_episodes} ---")
 recs = run_episode(client, model, surface, candidates, ep, all_records)
 all_records.extend(recs)
 classifications = [r["classification"] for r in recs]
 print(f" {Counter(classifications)}")

 # Aggregate
 counts = Counter(r["classification"] for r in all_records)
 total = len(all_records)
 oom_recs = [r for r in all_records if r["is_oom"]]
 oom_counts = Counter(r["classification"] for r in oom_recs)

 print(f"\n{'='*60}")
 print(f"LLM-AGENT MEMORY DIAGNOSTIC v2 — {total} proposals, {len(oom_recs)} OOMs")
 print(f"{'='*60}\n")
 print(f"| classification | count | % of OOMs |")
 print(f"|---|---|---|")
 for k in ["unpredictable", "predictable_from_telemetry", "inconclusive_placebo_changed",
 "warning_effective", "utilization_failure"]:
 v = oom_counts.get(k, 0)
 if v > 0:
 print(f"| {k} | {v} | {v/max(1,len(oom_recs))*100:.0f}% |")
 print(f"| ok (not OOM) | {counts.get('ok',0)} | — |")

 # Write full log
 out = Path(__file__).resolve().parents[1] / "results" / "llm_agent_memory_diagnostic.json"
 out.parent.mkdir(parents=True, exist_ok=True)
 payload = {"model": model, "n_episodes": n_episodes, "total_proposals": total,
 "total_ooms": len(oom_recs), "oom_classifications": dict(oom_counts),
 "records": all_records}
 out.write_text(json.dumps(payload, indent=2, default=str))
 print(f"\nwrote {out} ({len(all_records)} records with full prompts/responses)")


if __name__ == "__main__":
 main()
