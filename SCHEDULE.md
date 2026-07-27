# KDD'26 Tutorial — 3-Phase Schedule (3 × 60 min)

**Tutorial:** *Systematic Diagnosis and Benchmarking of Memory Systems in Autonomous AI Research
Agents: A Low-Resource Framework* (KDD'26, Jeju).

This delivery schedule re-cuts the accepted proposal into **three equal 60-minute phases**. Each
phase is self-contained and ends with a runnable outcome.

> ⚠️ **Restructure note (for the tutorial chairs).** The *accepted* proposal specified four parts
> plus a 15-minute break (Part 1 45 min / Part 2 50 min / Part 3 45 min / Break 15 / Part 4 30 min).
> This delivery uses **3 × 60 min (180 min)** instead. The content coverage is equivalent (Phase 1 ≈
> Part 1; Phase 2 ≈ Parts 2–3; Phase 3 ≈ Part 4); only the segmentation changed. Flag this to the
> chairs if a written amendment is required.

## Prerequisites (per participant / group)
- Laptop with Python 3.10+, 8 GB RAM, no GPU required for Phases 1–2.
- **Phase 1 (LLM techniques):** a free OpenAI-compatible key — the **OpenAI-compatible endpoint**
 (`OPENAI_API_KEY=-…`, `OPENAI_BASE_URL=https://api.openai.com/v1`).
- **Phase 3 (autoresearch groups):** a Modal account (`modal setup`) for L4 GPU access
 (~$0.5–1 per group).

---

## Phase 1 — Agent Memory Techniques (60 min)
*All 30 techniques, surveyed and hands-on, using Nir Diamant's Agent Memory Techniques (Apache-2.0).*

| Time | Activity |
|---|---|
| 0:00–0:08 | Intro + the 6-family taxonomy (Short-term 01–05, Long-term 06–11, Cognitive 12–19, Retrieval/multi-agent 20–23, Frameworks 24–27, Eval/production 28–30) |
| 0:08–0:38 | Guided tour of all 30 in 6 clusters (~5 min each): lecturer runs the representative notebook — **01 Buffer** (offline), **06 Vector Store**, **09 Episodic**, **13 Hierarchical**, **20 Retrieval Patterns** (via your API); **25 Mem0 / 26 Letta** demoed |
| 0:38–0:52 | Hands-on: each participant picks **2 techniques** from {01, 06, 09, 10, 13, 20} and runs them on a sample conversation (Colab + API key). *07 Entity / 08 KG use Anthropic tool-use → lecturer demo only* |
| 0:52–0:58 | Share-out + failure-mode teaser (sets up Phase 2) |
| 0:58–1:00 | Bridge to Phase 2 |

**Driver:** `tutorial/phase1_memory_architectures/notebook.ipynb` (all-30 survey + curated hands-on,
-wired). Curated AMT notebooks vendored under `tutorial/phase1_memory_architectures/amt_curated/` (Apache-2.0 attribution).

## Phase 2 — Public datasets (60 min, offline)
*Diagnostic + benchmarking on real public memory benchmarks, using the rebuilt 4 providers.*

| Time | Activity |
|---|---|
| 0:00–0:10 | Lecture: the 3-probe diagnostic framework + INGEST→INDEX→SEARCH→ANSWER→EVALUATE→REPORT pipeline + the 4 genuine providers |
| 0:10–0:15 | Setup: clone, `pip install`, confirm an offline smoke run |
| 0:15–0:35 | **Exercise 1 — memory failure diagnosis:** `python -m diagnostic_framework diagnose …` on LoCoMo + LongMemEval; find retrieval_miss vs utilization failures; read the 3-arm utilization lab (`util ≠ hit`) |
| 0:35–0:52 | **Exercise 2 — benchmarking:** `benchmark-cli compare …` across verbatim/extracted_facts/episodic/hybrid on MemoryArena; shared leaderboard (accuracy/latency/cost/utilization) |
| 0:52–1:00 | Discussion: no single architecture dominates → bridge to Phase 3 |

**Driver:** `tutorial/phase2_public_datasets/notebook.ipynb`.

## Phase 3 — Autoresearch, group-split (60 min, Modal L4)
*The autonomous-research case study. Participants split into groups so each config slice fits the
GPU (avoiding the depth≥12 OOM seen in single-group runs).*

| Time | Activity |
|---|---|
| 0:00–0:10 | Lecture: the autoresearch loop (edit `train.py` → 5-min train → `val_bpb` → keep/discard) + memory-debug angle; recap the depth≥12 OOM → why we split |
| 0:10–0:15 | Group assignment — each group a slice that fits the L4: **A** depth 4,6 (b32) · **B** depth 8,10 (b32) · **C** depth 12,14 (b16, recovers OOM) · **D** depth 16,18 (b16) |
| 0:15–0:40 | Groups run their slice live on Modal L4 (`modal run scripts/build_autoresearch_trace.py --group X`); results stream to a shared leaderboard |
| 0:40–0:48 | Aggregate the depth-4→18 sweep; run the memory-debug/idea loop (keep/revise/discard across the group traces) |
| 0:48–0:56 | Group presentations (3 min each) |
| 0:56–1:00 | Wrap-up + resources |

**Driver:** `tutorial/phase3_autoresearch/notebook.ipynb` + `scripts/build_autoresearch_trace.py --group`.

---

## Topic set (all real data, bundled in `source/data/topics/`)
LoCoMo · LongMemEval (subset) · MemoryArena · **Autoresearch** (real L4 `val_bpb` trace).
