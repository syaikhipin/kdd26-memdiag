# Systematic Diagnosis & Benchmarking of Memory Systems in Autonomous AI Research Agents

**A hands-on KDD'26 tutorial** — diagnose, benchmark, and improve memory systems for autonomous
research agents. Everything runs on a laptop; no GPU required for Phases 1–2.

**KDD'26 · Jeju Island, Republic of Korea · August 9–13, 2026**

**Nur Arifin Akbar**¹ · **Rahool Dembani**² · **Gregorius Airlangga**³ · **Ripto Mukti Wibowo**³ ·
**Biagio Lenzitti**¹ · **Domenico Tegolo**¹

¹ Università degli Studi di Palermo, Italy · ² Singularlogic, Athens, Greece ·
³ Atma Jaya Catholic University of Indonesia, Jakarta

DOI: [10.1145/3770855.3816477](https://doi.org/10.1145/3770855.3816477) · ISBN: 979-8-4007-2259-2/2026/08

> **▶ Run it step by step:** open [`RUN.html`](RUN.html) (or serve locally with
> `python -m http.server 8000` → `http://localhost:8000/RUN.html`) for the phase-by-phase guide
> with commands, expected output, and timings. The 3-phase plan is in [`SCHEDULE.md`](SCHEDULE.md).
> The full documentation site is [`index.html`](index.html).

## Tutorial workflow

```text
┌─────────────────────────────────────────────────────────────────────┐
│                  4-PHASE HANDS-ON TUTORIAL (180 min)                │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 1 (45 min)  30 memory techniques × 6 families (8 notebooks)  │
│                    LLM demos (offline fallback)                     │
│  Phase 2 (40 min)  Diagnose + benchmark on public data (5 books)    │
│                    3-probe diag · utilization lab · LLM-as-judge    │
│  Phase 3 (35 min)  Real LLM-pretraining sweep on Modal L4 (4 books) │
│                    groups split by depth/batch · memory diagnostic  │
│  Phase 4 (50 min)  Cognitive Memory Layer — constraint-aware        │
│                    memory: CML-lite build + diagnose (1 notebook)   │
├─────────────────────────────────────────────────────────────────────┤
│  4 storage providers: verbatim | extracted_facts | episodic | hybrid│
│  + cognitive_constraint (Phase 4) — constraint-aware retrieval      │
│  Honest metrics: util ≠ hit | semantic_pass ≠ hit | 3-arm reader    │
└─────────────────────────────────────────────────────────────────────┘
```

## What you'll learn

1. **The 30 memory techniques** across 6 families (short-term, long-term, cognitive, retrieval,
   frameworks, production) — with hands-on demos using a real LLM.
2. **The 3-probe diagnostic framework** — how to measure retrieval relevance, context utilization,
   and failure root-cause independently (not circularly).
3. **How to benchmark** four genuine memory providers (verbatim, extracted facts, episodic,
   hierarchical) on real public datasets (LoCoMo, LongMemEval, MemoryArena).
4. **The autonomous-research case study** — run a real LLM-pretraining sweep on a cloud GPU,
   aggregate results across groups, and diagnose memory failures.
5. **Cognitive / constraint-aware memory** — build a pure-Python "CML-lite" provider that surfaces
   stored constraints (allergies, OOM rules) even on zero-overlap queries, then diagnose it with
   the same 3-probe harness.

## 🔬 Run in Google Colab

Every notebook below is a clickable Colab link (🔬). Click any name to open that specific notebook
in Google Colab — the setup cell auto-clones the repo and installs dependencies.

| Phase | Start here (click to open) | Total |
|---|---|---|
| `1` | [**01 Short-term Memory 🔬**](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase1_memory_architectures/01_short_term_memory.ipynb) | 8 notebooks |
| `2` | [**01 Diagnostic Framework 🔬**](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase2_public_datasets/01_diagnostic_framework.ipynb) | 5 notebooks |
| `3` | [**01 Autoresearch Loop 🔬**](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase3_autoresearch/01_autoresearch_loop.ipynb) | 4 notebooks |
| `4` | [**11 Cognitive Memory Layer 🔬**](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase4_cognitive_memory/11_cognitive_constraint_layer.ipynb) | 1 notebook |

Individual notebook links (all 18) are in the Phase sections below.

## Setup (5 minutes)

| Requirement | Details |
|---|---|
| **Python** | 3.10 or later |
| **RAM** | 8 GB+ (16 GB recommended for Phase 3) |
| **GPU** | Not required for Phases 1–2. Phase 3 uses a cloud L4 (Modal). |
| **API key (Phase 1)** | Free OpenAI-compatible endpoint: `OPENAI_API_KEY` + `OPENAI_BASE_URL=https://api.openai.com/v1` |
| **Modal (Phase 3)** | `pip install modal && modal setup` (~$0.5–1/group) |

```bash
# 1. Clone the repository
git clone -b tutorial-rebuild https://github.com/syaikhipin/kdd26-memdiag kdd-tutorial
cd kdd-tutorial

# 2. Install dependencies (offline-only, lightweight)
pip install numpy matplotlib pyyaml

# 3. Install the CLI package (optional but recommended)
pip install -e source              # core package
pip install -e "source[api]"       # + openai/tiktoken for LLM demos

# 4. Verify
python -m compileall -q source     # should print nothing (success)
```

> **For Phase 1 LLM demos:** export an OpenAI-compatible endpoint before launching Jupyter.
> Without a key, the demos fall back to offline heuristics (still runnable, just less rich).

## Quick start

Open any notebook in Jupyter or Colab — each is independent and self-contained:

```bash
# Phase 1 (offline or LLM):
jupyter notebook tutorial/phase1_memory_architectures/01_short_term_memory.ipynb

# Phase 2 (offline, no key):
jupyter notebook tutorial/phase2_public_datasets/01_diagnostic_framework.ipynb

# Or use the CLI directly:
diagnostic-framework diagnose --strategies verbatim,extracted_facts,episodic,hybrid --top_k 5
benchmark-cli compare --providers verbatim,extracted_facts,episodic,hybrid --sample_size 20
```

Reproduce the bundled sample run offline (no API key), from this folder:

```bash
python -m compileall -q source
python source/run.py --mode synthetic --backend offline --episodes 5
```

Real benchmark on the bundled topic subsets (LoCoMo, LongMemEval, MemoryArena):

```bash
python source/run.py --mode real --backend offline \
  --datasets locomo longmemeval memoryarena \
  --max-questions 40 --max-items 100 --top-k 5 --visualize
```

## `1` Phase 1 — Agent Memory Techniques (45 min)

**Goal:** tour all 30 memory techniques across 6 families, with hands-on demos. LLM-based
techniques use an OpenAI-compatible endpoint; offline fallbacks are always available.

*Adapted from **Agent Memory Techniques** by Nir Diamant (Apache-2.0).*

| Family | Techniques | What it solves |
|---|---|---|
| **Short-term** | 01–05 | Keep recent turns without filling the context window |
| **Long-term** | 06–11 | Persist knowledge across sessions, users, and time |
| **Cognitive** | 12–19 | Working, hierarchical, consolidation, reflection, routing, forgetting |
| **Retrieval** | 20–23 | Choose what to recall and when; share across agents |
| **Frameworks** | 24–27 | Production libraries: Graphiti, Mem0, Letta, Zep |
| **Evaluation** | 28–30 | Measure quality, benchmark, deploy at scale |

### Phase 1 notebooks (8)

| # | Notebook | Covers | LLM demos |
|---|---|---|---|
| 01 | [`01_short_term_memory` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase1_memory_architectures/01_short_term_memory.ipynb) | Buffer, sliding, summary, token budget | 03 summary |
| 02 | [`02_long_term_memory` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase1_memory_architectures/02_long_term_memory.ipynb) | Vector, entity, KG, episodic, semantic, procedural | 07 entity |
| 03 | [`03_cognitive_architectures` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase1_memory_architectures/03_cognitive_architectures.ipynb) | Working, hierarchical, consolidation, forgetting | via providers |
| 04 | [`04_retrieval_multi_agent` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase1_memory_architectures/04_retrieval_multi_agent.ipynb) | Retrieval patterns, cross-session, multi-agent, tools | — |
| 05 | [`05_frameworks` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase1_memory_architectures/05_frameworks.ipynb) | Graphiti, Mem0, Letta, Zep (descriptions) | — |
| 06 | [`06_evaluation_production` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase1_memory_architectures/06_evaluation_production.ipynb) | Evaluation, benchmarks, production | — |
| 07 | [`07_capstone` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase1_memory_architectures/07_capstone.ipynb) | 4 architectures compared on one conversation | via providers |
| 08 | [`08_exercise` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase1_memory_architectures/08_exercise.ipynb) | **Hands-on:** build a support-agent memory system | yes |

## `2` Phase 2 — Public Datasets (40 min, offline)

**Goal:** diagnose retrieval vs utilization failures on real benchmarks, then benchmark the 4
providers. No API key needed.

### The 3-probe diagnostic framework

| Probe | Measures | How |
|---|---|---|
| **1. Retrieval relevance** | Did the system retrieve the right evidence? | P/R/F1 vs gold evidence IDs |
| **2. Context utilization** | Did the agent *use* the memory? | 3-arm reader (no-context / oracle / provider) |
| **3. Failure root-cause** | Where did it break? | retrieval_miss / partial / retrieved-but-unused |

> **The circularity trap:** many benchmarks set "memory used" = "evidence retrieved" — making
> utilization just retrieval in disguise. Our 3-arm reader avoids this.

### Phase 2 notebooks (5)

| # | Notebook | What | Time |
|---|---|---|---|
| 01 | [`diagnostic_framework` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase2_public_datasets/01_diagnostic_framework.ipynb) | Exercise 1: diagnose failures | 1.4s |
| 02 | [`utilization_lab` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase2_public_datasets/02_utilization_lab.ipynb) | 3-arm utilization lab + inspect a failure | <1s |
| 03 | [`benchmarking` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase2_public_datasets/03_benchmarking.ipynb) | Exercise 2: benchmark providers | 4.2s |
| 04 | [`llm_as_judge_lab` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase2_public_datasets/04_llm_as_judge_lab.ipynb) | **Live LLM judge** vs offline proxy | 25s |
| 05 | [`question_type_analysis` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase2_public_datasets/05_question_type_analysis.ipynb) | Per-type: which architecture wins where | <1s |

## `3` Phase 3 — Autoresearch (35 min, Modal L4)

**Goal:** run a real LLM-pretraining sweep on a cloud GPU, aggregate across groups, and diagnose
memory failures.

### Group slices (each fits one L4)

| Group | Configs (depth, batch) | Why |
|---|---|---|
| A | (4,32), (6,32) | Shallow, safe |
| B | (8,32), (10,32) | Medium depth |
| C | (12,16), (14,16) | Deep — batch 16 recovers the OOM |
| D | (16,16), (18,16) | Deepest |

### Phase 3 notebooks (4)

| # | Notebook | What | Time |
|---|---|---|---|
| 01 | [`autoresearch_loop` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase3_autoresearch/01_autoresearch_loop.ipynb) | The loop + why we split into groups | <1s |
| 02 | [`run_your_group` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase3_autoresearch/02_run_your_group.ipynb) | Run YOUR slice on Modal L4 | ~14.5 min |
| 03 | [`aggregate_and_debug` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase3_autoresearch/03_aggregate_and_debug.ipynb) | Aggregate sweep + keep/revise/discard | 8s |
| 04 | [`memory_diagnostic` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase3_autoresearch/04_memory_diagnostic.ipynb) | 5-condition interventional diagnostic | pilot |

## `4` Phase 4 — Cognitive Memory Layer (50 min, offline)

**Goal:** go beyond *storage* memory. Build a tiny "CML-lite" provider that adds the one thing
verbatim / episodic / extracted / hybrid all lack — **constraint awareness**: surfacing a stored
rule ("I'm allergic to shellfish", "depth 16 → OOM") even when the query shares zero words with
it. Then diagnose it with the same 3-probe / LoCoMo harness from Phase 2.

*Inspired by the **Cognitive Memory Layer**
([avinash-mall/CognitiveMemoryLayer](https://github.com/avinash-mall/CognitiveMemoryLayer)), a
neuro-inspired system (Docker + Postgres + Neo4j + Redis + ~25 GB of models) we cannot run in a
Colab slot — so we capture its cognitive core in ~90 lines of pure Python, fully offline.*

| # | Notebook | What | Time |
|---|---|---|---|
| 11 | [`cognitive_constraint_layer` 🔬](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/tutorial/phase4_cognitive_memory/11_cognitive_constraint_layer.ipynb) | Constraint extraction + zero-overlap retrieval + 3-probe diagnostic | <1s |

> **The zero-overlap win:** store "I'm allergic to shellfish", then ask "recommend a restaurant".
> Plain lexical memory (verbatim) retrieves *nothing* (cosine ≈ 0); CML-lite's read path
> **injects the POLICY constraint regardless of overlap**. This is exactly where standard RAG
> fails but cognitive memory succeeds — and it turns the Phase-3 OOM rule into a first-class
> CAUSAL constraint.

## Results (honest, tutorial-scale)

> **Teaching sample** (LoCoMo 40 / LongMemEval 50 / MemoryArena 86 questions). Full provenance in
> [`results/RESULTS_MANIFEST.md`](results/RESULTS_MANIFEST.md) — read before citing.

### LoCoMo — where the 4 architectures are distinct

| Strategy | Precision | Recall | Hit | Memory size |
|---|---|---|---|---|
| no_memory | 0.000 | 0.000 | 0.000 | 0 |
| verbatim | 0.040 | 0.188 | 0.200 | 419 |
| extracted_facts | 0.040 | 0.188 | 0.200 | 1282 |
| episodic | 0.000 | 0.000 | 0.000 | 19 |
| hybrid | 0.003 | 0.125 | 0.125 | 438 |

Memory footprints differ 10x (419 / 1282 / 19 / 438) — genuinely different architectures.
No single strategy dominates.

### Autoresearch trace (real L4)

| Depth | Batch | val_bpb (lower = better) |
|---|---|---|
| 4 | 32 | **1.254** |
| 6 | 32 | 1.296 |
| 8 | 32 | 1.433 |
| 10 | 32 | 1.661 |

### 3-arm utilization lab results

| Strategy | p0 (baseline) | Oracle (ceiling) | Provider | Net gain | Beneficial | Harmful |
|---|---|---|---|---|---|---|
| no_memory | 0.25 | 0.60 | 0.25 | 0.00 | 0 | 0 |
| verbatim | 0.25 | 0.60 | **0.35** | **+0.10** | 7 | 3 |
| extracted_facts | 0.25 | 0.60 | 0.325 | +0.075 | 6 | 3 |
| episodic | 0.25 | 0.60 | **0.35** | **+0.10** | 5 | 1 |
| hybrid | 0.25 | 0.60 | **0.35** | **+0.10** | 7 | 3 |

Memory helps by **+10pp** accuracy (provider vs no-memory). The oracle ceiling (0.60) shows room
for improvement — better retrieval/representation could close the gap.

### Memory diagnostic (Phase 3D)

5 memory conditions × K=5 repeated probes, frozen pre-decision snapshots, real LLM agent:

| Condition | OOM rate | Any-OOM | Tests |
|---|---|---|---|
| M0 no-memory | **50%** | 62% | Placebo baseline |
| M1 raw-history | **12%** | 12% | Full trial log |
| M2 retrieved | 12% | **30%** | Batch-keyed retrieval |
| M3 structured-rule | **5%** | 5% | Raw + constraint |
| M4 oracle | **0%** | 12% | Raw + outcome |

> **Key finding:** any history cuts OOM proposals by 76%. **Retrieval filtering is
> counterproductive** (loses cross-task context, increases OOMs 2.5x). Structured rules approach
> oracle performance.

> **Provenance:** the committed sample was produced by the rebuilt providers (4 genuine
> architectures, non-circular metrics) at tutorial scale — not paper-scale. Pre-refactor numbers
> were removed (they used prefix-string strategies + circular utilization metrics). Full-scale
> runs require the complete datasets ([`data/README.md`](data/README.md)) and the Modal GPU runner
> (`source/modal_runner.py`):
>
> ```bash
> python source/run.py \
>   --runner modal --modal-gpu A10G --modal-detach \
>   --mode real --backend offline \
>   --datasets locomo longmemeval memoryarena \
>   --max-conversations 999 --max-questions 999 --top-k 5 --visualize
> ```
>
> Detached Modal runs print a call id; fetch artifacts later with
> `python source/run.py --runner modal --mode real --modal-call-id <call-id>`.

## 4-Phase schedule (180 min)

| Time | Phase | Activity |
|---|---|---|
| 0:00–0:45 | `1` | 30 memory techniques (8 notebooks) + hands-on exercise |
| 0:45–0:55 | — | Break (install Modal for Phase 3) |
| 0:55–1:35 | `2` | Diagnose + benchmark (5 notebooks) + LLM-as-judge lab + type analysis |
| 1:35–2:10 | `3` | Autoresearch groups (4 notebooks) + memory diagnostic |
| 2:10–3:00 | `4` | Cognitive Memory Layer — CML-lite build + diagnose (1 notebook) + discussion |

## FAQ

<details><summary>Do I need an API key?</summary>

Phase 1 LLM demos use an OpenAI-compatible endpoint. Without a key, offline heuristics.
Phase 2 is fully offline. Phase 3 uses Modal (cloud GPU).
</details>

<details><summary>Do I need a GPU?</summary>

No for Phases 1–2. Phase 3 uses a cloud L4 via Modal (~$0.5–1/group, ~15 min).
</details>

<details><summary>Are these results publishable?</summary>

No — tutorial-scale (40–86 questions). Full-scale needs complete datasets + Modal.
</details>

<details><summary>Can I use the providers in my project?</summary>

Yes — `pip install -e source` then `from providers import build_provider`. MIT-licensed.
</details>

## Repo contents

```text
source/        Python implementation (providers, probes, evaluators, CLIs)
tutorial/      18 numbered Jupyter/Colab notebooks across 4 phases
data/          Dataset availability and download guidance
results/       Sample-run metrics, summaries, reports, figures (see results/RESULTS_MANIFEST.md)
logs/          Selected provenance logs
scripts/       Real-trace builders (OpenML HPO + authentic Modal autoresearch)
index.html     Static GitHub Pages homepage (full documentation)
RUN.html       Step-by-step run guide with commands and timings
SCHEDULE.md    The session plan
```

## Citation

If you use this tutorial or its code in your work, please cite the paper:

> Nur Arifin Akbar, Rahool Dembani, Gregorius Airlangga, Ripto Mukti Wibowo, Biagio Lenzitti,
> and Domenico Tegolo. 2026. *Systematic Diagnosis & Benchmarking of Memory Systems in
> Autonomous AI Research Agents: A Low-Resource, Offline-First Framework*. In *Proceedings of
> the ACM SIGKDD Conference on Knowledge Discovery and Data Mining Tutorials (KDD '26)*.
> ACM. https://doi.org/10.1145/3770855.3816477

```bibtex
@inproceedings{akbar2026memdiag,
  author    = {Akbar, Nur Arifin and Dembani, Rahool and Airlangga, Gregorius and
               Wibowo, Ripto Mukti and Lenzitti, Biagio and Tegolo, Domenico},
  title     = {Systematic Diagnosis \& Benchmarking of Memory Systems in Autonomous AI Research
               Agents: A Low-Resource, Offline-First Framework},
  booktitle = {Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining
               Tutorials (KDD '26)},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3770855.3816477},
  url       = {https://doi.org/10.1145/3770855.3816477}
}
```

## Credits & licenses

- **Phase 1 hands-on is adapted from *Agent Memory Techniques* by Nir Diamant**
  (https://github.com/NirDiamant/Agent_Memory_Techniques), licensed under the Apache License 2.0.
  Six curated notebooks are vendored under `tutorial/phase1_memory_architectures/amt_curated/`
  (with `utils/helpers.py`); see `tutorial/phase1_memory_architectures/amt_curated/NOTICE` and
  `licenses/APACHE-2.0.txt`. The remaining 24 techniques are surveyed (with links) in
  `tutorial/phase1_memory_architectures/notebook.ipynb`.
- **Phase 4 cognitive-constraint provider ("CML-lite")** — original pure-Python re-implementation
  of the key idea of the *Cognitive Memory Layer*
  ([avinash-mall/CognitiveMemoryLayer](https://github.com/avinash-mall/CognitiveMemoryLayer)).
  MIT-licensed; the reference system is not vendored.
- **Datasets** — LoCoMo (snap-research/locomo), LongMemEval (xiaowu0162/longmemeval-cleaned),
  MemoryArena (ZexueHe/memoryarena); research use, upstream licenses (see `data/README.md`).
- **Autoresearch** — Karpathy's nanochat-derived pretraining loop (vendored in `autoresearch/`).
- All other code is original to the tutorial authors (MIT; see `source/pyproject.toml`).

Funded by the European Union's Horizon research and innovation programme under the
Marie Skłodowska-Curie grant agreement No. 101073381.

## Secret safety

This submission does not include API keys. OpenAI-compatible, Rhesis, and Modal credentials must
be supplied through environment variables only and are never written to source, notebooks, JSON,
TSV, reports, or figures.
