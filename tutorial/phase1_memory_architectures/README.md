# Phase 1 — 30 agent-memory techniques (8 independent notebooks)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/syaikhipin/kdd26-memdiag/blob/tutorial-rebuild/experiment/github_submission/phase1_memory_architectures/)

Each notebook is **self-contained** — open any one in Google Colab (or locally) and run top-to-bottom.
The setup cell auto-clones the repo + installs deps, and uses the LLM gateway when a key is set
(falls back to offline heuristics without one).

| Notebook | Techniques |
|---|---|
| [01_short_term_memory.ipynb](01_short_term_memory.ipynb) | 01–05 buffer / sliding window / summary / summary-buffer / token |
| [02_long_term_memory.ipynb](02_long_term_memory.ipynb) | 06–11 vector / entity / KG / episodic / semantic / procedural |
| [03_cognitive_architectures.ipynb](03_cognitive_architectures.ipynb) | 12–19 working / hierarchical / consolidation / compaction / self-reflection / routing / temporal / forgetting |
| [04_retrieval_multi_agent.ipynb](04_retrieval_multi_agent.ipynb) | 20–23 retrieval patterns / cross-session / multi-agent / memory-as-tools |
| [05_frameworks.ipynb](05_frameworks.ipynb) | 24–27 Graphiti / Mem0 / Letta / Zep (descriptions + pointers) |
| [06_evaluation_production.ipynb](06_evaluation_production.ipynb) | 28–30 evaluation / benchmarks / production |
| [07_capstone.ipynb](07_capstone.ipynb) | the 4 production architectures compared |
| [08_exercise.ipynb](08_exercise.ipynb) | **Hands-on:** build a memory system for a support agent using 2 techniques |

**LLM demos:** techniques 03 (summary), 07 (entity), 10 (semantic), 13 (hierarchical) now use
real LLM calls via an OpenAI-compatible endpoint (with offline fallback). Set `OPENAI_API_KEY` + `OPENAI_BASE_URL`
before launching.

Full AMT versions of curated techniques are in `amt_curated/`.
Attribution: adapted from *Agent Memory Techniques* by Nir Diamant, Apache-2.0.

**60-min plan:** 0:00–0:08 intro · 0:08–0:46 tour notebooks 01–06 (~6 min each) · 0:46–0:54 capstone (07) ·
0:54–1:00 hands-on exercise (08) + bridge to Phase 2.
