#!/usr/bin/env modal
"""Authentic single-GPU autonomous-research trace builder (Modal, H100).

Runs the repo's own ``autoresearch/train.py`` (Karpathy nanochat-derived GPT pretraining; metric =
val_bpb) for a hyperparameter sweep on Modal, producing a REAL trace of >= ``--configs`` experiments.
Each experiment edits DEPTH / DEVICE_BATCH_SIZE in train.py, trains for the fixed 5-min budget, and
records val_bpb + telemetry. This is the "100+ experiments overnight on a single GPU" motivating
trace for the tutorial's autonomous-research topic.

Heavy (H100 + climbmix data + ~5 min/run). Launch detached and collect later:

 modal run scripts/build_autoresearch_trace.py --configs 6

Requires a Modal account (``modal setup``); the ``nur-arifin-akbar`` workspace is configured here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import modal


def _find_autoresearch() -> Path:
 here = Path(__file__).resolve()
 for parent in [here.parent, *here.parents]:
 if (parent / "autoresearch").is_dir():
 return parent / "autoresearch"
 raise RuntimeError("Could not locate the autoresearch/ directory from scripts/build_autoresearch_trace.py")


AUTORESEARCH_DIR = _find_autoresearch()
OUT_DIR = Path(__file__).resolve().parents[1] / "source" / "data" / "topics" / "autoresearch"

# (DEPTH, DEVICE_BATCH_SIZE) — batch kept small (32) to fit a cheaper L4 GPU; DEPTH is the sweep
# knob (varies model capacity -> distinct val_bpb). >= 6 experiments.
DEFAULT_CONFIGS = [(4, 32), (6, 32), (8, 32), (10, 32), (12, 32), (16, 32)]

# Phase-3 group slices: each slice fits one L4 so every group succeeds (smaller batch for deeper
# models avoids the depth>=12 OOM that a single full sweep hits). Aggregate all groups for the
# full depth-4..18 sweep.
GROUP_CONFIGS = {
 "A": [(4, 32), (6, 32)],
 "B": [(8, 32), (10, 32)],
 "C": [(12, 16), (14, 16)], # batch 16 recovers the configs that OOM'd at batch 32
 "D": [(16, 16), (18, 16)],
}

# Frozen outcome surface for the experiment-history-memory benchmark (auto-review-loop).
# A grid that produces BOTH successes (low depth / batch 16) and resource failures (deep + batch 32
# OOMs). ALL trials are retained (incl. failures) so the C0/C1/C2 comparison has real waste/regret.
SURFACE_CONFIGS = [
 (2, 32), (4, 32), (6, 32), (8, 32), (10, 32), (12, 32), # batch 32: depths 2-10 ok, 12+ OOM
 (12, 16), (14, 16), (16, 16), (18, 16), # batch 16: recovered deep configs
] # 10 configs — fits within 90-min L4 timeout

_FA3_SETUP = """from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface"""

_FA3_SETUP_PATCHED = "# PATCHED for non-Hopper GPUs: flash-attn-3 removed; using torch SDPA instead."

_FA3_CALL = """ y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
 y = y.contiguous().view(B, T, -1)"""

_FA3_CALL_PATCHED = """ # PATCHED: torch SDPA (works on any CUDA GPU; full causal, sliding-window ignored).
 _q = q.transpose(1, 2)
 _k = k.transpose(1, 2)
 _v = v.transpose(1, 2)
 if _k.size(1) != _q.size(1):
 _rep = _q.size(1) // _k.size(1)
 _k = _k.repeat_interleave(_rep, dim=1)
 _v = _v.repeat_interleave(_rep, dim=1)
 y = F.scaled_dot_product_attention(_q, _k, _v, is_causal=True)
 y = y.transpose(1, 2).contiguous().view(B, T, -1)"""


def _apply_sdpa_patch(src: str) -> str:
 """Replace flash-attention-3 (Hopper-only) with torch SDPA so the code runs on a cheaper GPU.

 Applied at runtime to a copy of train.py; the canonical ``autoresearch/train.py`` is untouched.
 """
 if _FA3_SETUP not in src:
 raise RuntimeError("Could not find the flash-attn-3 setup block to patch in train.py")
 if _FA3_CALL not in src:
 raise RuntimeError("Could not find the flash-attn-3 call site to patch in train.py")
 return src.replace(_FA3_SETUP, _FA3_SETUP_PATCHED).replace(_FA3_CALL, _FA3_CALL_PATCHED)

image = (
 modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.11")
 .apt_install("git", "build-essential", "libglib2.0-0", "wget")
 .pip_install("uv")
 .add_local_dir(str(AUTORESEARCH_DIR), "/root/autoresearch", copy=True)
 .run_commands("cd /root/autoresearch && uv sync --no-progress")
)

app = modal.App("kdd-autoresearch-trace", image=image)


@app.function(gpu="L4", timeout=5400)
def run_sweep(configs: list[tuple[int, int]] | None = None, shards: int = 8) -> dict:
 import pathlib
 import subprocess

 ar = pathlib.Path("/root/autoresearch")
 train_py = ar / "train.py"
 base = _apply_sdpa_patch(train_py.read_text())

 # 1. one-time data + tokenizer prep (downloads a few climbmix shards)
 prep = subprocess.run(
 ["uv", "run", "prepare.py", "--num-shards", str(shards)],
 cwd=ar, capture_output=True, text=True,
 )
 if prep.returncode != 0:
 return {"ok": False, "stage": "prepare", "stderr": prep.stderr[-4000:], "stdout": prep.stdout[-2000:]}

 configs = configs or DEFAULT_CONFIGS
 trials = []
 for i, (depth, batch) in enumerate(configs):
 trial_id = f"ar_trial_{i:03d}"
 text = re.sub(r"^DEPTH\s*=\s*\d+", f"DEPTH = {depth}", base, count=1, flags=re.MULTILINE)
 text = re.sub(r"^DEVICE_BATCH_SIZE\s*=\s*\d+", f"DEVICE_BATCH_SIZE = {batch}", text, count=1, flags=re.MULTILINE)
 train_py.write_text(text)
 proc = subprocess.run(["uv", "run", "train.py"], cwd=ar, capture_output=True, text=True)
 out = proc.stdout + "\n" + proc.stderr
 failed = proc.returncode != 0 or "FAIL" in proc.stdout
 m_bpb = re.search(r"val_bpb:\s*([0-9.]+)", out)
 m_secs = re.search(r"training_seconds:\s*([0-9.]+)", out)
 m_vram = re.search(r"peak_vram_mb:\s*([0-9.]+)", out)
 m_steps = re.search(r"num_steps:\s*([0-9.]+)", out)
 m_params = re.search(r"num_params_M:\s*([0-9.]+)", out)
 val_bpb = float(m_bpb.group(1)) if m_bpb else None
 trials.append({
 "trial_id": trial_id,
 "config": {"depth": depth, "device_batch_size": batch},
 "val_bpb": val_bpb,
 "training_seconds": float(m_secs.group(1)) if m_secs else None,
 "peak_vram_mb": float(m_vram.group(1)) if m_vram else None,
 "num_steps": int(float(m_steps.group(1))) if m_steps else None,
 "num_params_M": float(m_params.group(1)) if m_params else None,
 "status": "ok" if (not failed and val_bpb is not None) else ("fail" if failed else "no_metric"),
 "failure_type": ("oom" if (failed and "OutOfMemory" in out) else ("crash" if failed else ("no_metric" if val_bpb is None else "none"))),
 "error_tail": (proc.stderr[-1500:] if failed else ""),
 })
 # restore base for next edit
 train_py.write_text(base)
 valid = [t for t in trials if t["val_bpb"] is not None]
 best = min(valid, key=lambda t: t["val_bpb"]) if valid else None # lower val_bpb is better
 return {"ok": True, "n": len(trials), "best_trial_id": best["trial_id"] if best else None,
 "best_val_bpb": best["val_bpb"] if best else None, "trials": trials}


def _write_group(out_dir: Path, group: str, result: dict) -> Path:
 """Write one group's slice to the shared leaderboard dir."""
 lb = out_dir / "leaderboard"
 lb.mkdir(parents=True, exist_ok=True)
 payload = {
 "group": group,
 "topic": "autoresearch",
 "source": "autoresearch/train.py on Modal L4 (SDPA patch; real LLM-pretraining val_bpb)",
 "metric": "val_bpb (lower is better)",
 "n_trials": result["n"],
 "best_trial_id": result["best_trial_id"],
 "best_val_bpb": result["best_val_bpb"],
 "trials": result["trials"],
 }
 path = lb / f"{group}.json"
 path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
 return path


@app.local_entrypoint()
def main(configs: int = 6, shards: int = 8, group: str = None, surface: bool = False):
 if surface:
 cfgs = SURFACE_CONFIGS
 elif group is not None:
 if group not in GROUP_CONFIGS:
 raise SystemExit(f"Unknown group '{group}'. Known: {sorted(GROUP_CONFIGS)}")
 cfgs = GROUP_CONFIGS[group]
 else:
 cfgs = DEFAULT_CONFIGS[:configs] if configs <= len(DEFAULT_CONFIGS) else DEFAULT_CONFIGS
 result = run_sweep.remote(cfgs, shards)
 if not result.get("ok"):
 print("SWEEP FAILED at stage:", result.get("stage"))
 print(result.get("stderr", ""))
 return
 out = OUT_DIR
 out.mkdir(parents=True, exist_ok=True)
 payload = {
 "topic": "autoresearch",
 "source": "autoresearch/train.py on Modal L4 (SDPA patch; real LLM-pretraining val_bpb)",
 "metric": "val_bpb (lower is better)",
 "n_trials": result["n"],
 "best_trial_id": result["best_trial_id"],
 "best_val_bpb": result["best_val_bpb"],
 "trials": result["trials"],
 }
 if surface:
 # Frozen outcome surface for the experiment-history-memory benchmark: RETAIN ALL trials
 # (ok + fail), with failure_type, so the C0/C1/C2 comparison has real waste/regret signal.
 surface_path = out / "frozen_surface.json"
 n_ok = sum(1 for t in result["trials"] if t["status"] == "ok")
 n_fail = result["n"] - n_ok
 surface_payload = {**payload, "n_ok": n_ok, "n_fail": n_fail,
 "note": "frozen outcome surface; all trials retained (incl. OOM/crash) for the experiment-history-memory benchmark."}
 surface_path.write_text(json.dumps(surface_payload, indent=2), encoding="utf-8")
 print(f"[surface] wrote {result['n']} trials ({n_ok} ok, {n_fail} fail) -> {surface_path}")
 return
 if group is not None:
 payload["group"] = group
 gpath = _write_group(out, group, result)
 print(f"[group {group}] wrote {result['n']} trials -> {gpath}")
 print(f"[group {group}] best (lowest val_bpb): {result['best_trial_id']} = {result['best_val_bpb']}")
 print("Run the Phase 3 notebook (or read leaderboard/*.json) to aggregate all groups.")
 return
 (out / "autoresearch_trials.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
 with open(out / "results.tsv", "w", encoding="utf-8") as f:
 f.write("trial_id\tdepth\tbatch\tval_bpb\tstatus\n")
 for t in result["trials"]:
 f.write(f"{t['trial_id']}\t{t['config']['depth']}\t{t['config']['device_batch_size']}\t{t['val_bpb']}\t{t['status']}\n")
 with open(out / "run.log", "w", encoding="utf-8") as f:
 f.write(json.dumps(payload, indent=2)[:4000] + "\n")
 print(f"wrote {result['n']} autoresearch trials -> {out / 'autoresearch_trials.json'}")
 print(f"best (lowest val_bpb): {result['best_trial_id']} = {result['best_val_bpb']}")
