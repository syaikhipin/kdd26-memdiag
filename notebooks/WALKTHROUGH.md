# KDD Tutorial Notebook Walkthrough

This walkthrough explains how to run the Jupyter/Colab tutorial notebooks and how to reproduce the experiment results.

## 1. What this tutorial covers

The notebooks follow the accepted KDD tutorial proposal: **Systematic Diagnosis and Benchmarking of Memory Systems in Autonomous AI Research Agents: A Low-Resource Framework**.

The tutorial design is:

| Proposal section | Notebook coverage |
|---|---|
| Part 1: Memory Architectures in Autonomous Research Systems | `1_setup_and_timeline.ipynb`, `3_memory_architectures.ipynb` |
| Live demo: memory failures in action | `5_full_real_benchmark.ipynb`, `6_visualization_dashboard.ipynb` |
| Part 2: Diagnostic Framework and Evaluation Protocols | `4_three_probe_diagnostics.ipynb` |
| Hands-On Exercise 1: Memory Failure Diagnosis | `4_three_probe_diagnostics.ipynb`, `6_visualization_dashboard.ipynb` |
| Part 3: Standardized Benchmarking Infrastructure | `5_full_real_benchmark.ipynb` |
| Hands-On Exercise 2: Benchmarking Memory Systems | `5_full_real_benchmark.ipynb`, `6_visualization_dashboard.ipynb` |
| Part 4: Low-Resource Autonomous Research Case Study | `7_autoresearch_agent_loop.ipynb` |
| Timeline fit and packaging guidance | `8_kdd_timeline_fit_analysis.ipynb`, `9_colab_guidance_and_next_steps.ipynb` |

## 2. Notebook order

Run the notebooks in this order:

```text
1_setup_and_timeline.ipynb
2_dataset_registry_and_downloads.ipynb
3_memory_architectures.ipynb
4_three_probe_diagnostics.ipynb
5_full_real_benchmark.ipynb
6_visualization_dashboard.ipynb
7_autoresearch_agent_loop.ipynb
8_kdd_timeline_fit_analysis.ipynb
9_colab_guidance_and_next_steps.ipynb
10_external_testing_integrations.ipynb  # optional maintainer notebook
```

Executed local copies are saved as:

```text
*.executed.ipynb
```

Execution logs are saved under:

```text
experiment/notebooks/logs/
```

## 3. Local setup

From the repository root:

```bash
python -m compileall -q experiment
```

The latest validation log is:

```text
experiment/results/logs/compileall.log
```

## 4. Dataset availability

Currently available locally:

```text
LoCoMo
LongMemEval
MemoryArena
```

Optional proposal-listed extensions:

```text
LCBench
HPOBench
```

These are intentionally marked optional because their real benchmark artifacts are not installed in the current local setup.

## 5. Low-resource Colab command

For live tutorial participants, prefer the smaller command:

```bash
python experiment/run.py \
  --mode real \
  --backend offline \
  --datasets locomo longmemeval memoryarena \
  --longmemeval-files longmemeval_oracle.json \
  --max-conversations 2 \
  --max-questions 20 \
  --max-items 100 \
  --top-k 5 \
  --eval-backend offline \
  --eval-limit 50 \
  --visualize
```

This better matches the proposal's low-resource expectation.

## 6. Full local benchmark command

For full local reproduction:

```bash
python experiment/run.py \
  --mode real \
  --backend offline \
  --datasets locomo longmemeval memoryarena \
  --max-conversations 999 \
  --max-questions 999 \
  --top-k 5 \
  --eval-backend offline \
  --eval-limit 50 \
  --visualize
```

For paper-quality full semantic evaluation, use the Modal GPU runner instead of the capped local evaluator. The Modal GPU smoke test has been verified end-to-end and synced local artifacts successfully.

```bash
python experiment/run.py \
  --runner modal \
  --modal-gpu A10G \
  --modal-detach \
  --mode real \
  --backend offline \
  --datasets locomo longmemeval memoryarena \
  --max-conversations 999 \
  --max-questions 999 \
  --top-k 5 \
  --eval-backend offline \
  --visualize
```

The latest paper-quality Modal GPU log is:

```text
experiment/results/logs/fetch_modal_gpu_full_corrected.log
```

Latest paper-quality Modal GPU outputs:

```text
experiment/results/run_20260509_151530_real_raw.json
experiment/results/run_20260509_151530_real_metrics.json
experiment/results/run_20260509_151530_real_summary.tsv
```

Figures:

```text
experiment/results/figure_real_retrieval.png
experiment/results/figure_semantic_scores.png
experiment/results/figure_semantic_vs_retrieval.png
experiment/results/figure_evaluator_coverage.png
experiment/results/figure_memory_growth.png
experiment/results/figure_failures.png
```

## 7. Full benchmark results

Latest paper-quality benchmark summary from Modal GPU, with full offline semantic coverage:

| Dataset | Strategy | Questions | Precision | Recall | Hit Rate | Sem. Cov. | Sem. Score | Sem. Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | no_memory | 1540 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| LoCoMo | verbatim | 1540 | 0.057 | 0.249 | 0.277 | 1.000 | 0.432 | 0.277 |
| LoCoMo | extracted_facts | 1540 | 0.055 | 0.242 | 0.271 | 1.000 | 0.422 | 0.271 |
| LoCoMo | episodic | 1540 | 0.064 | 0.278 | 0.311 | 1.000 | 0.465 | 0.311 |
| LoCoMo | hybrid | 1540 | 0.058 | 0.256 | 0.284 | 1.000 | 0.435 | 0.284 |
| LongMemEval | no_memory | 1500 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| LongMemEval | verbatim | 1500 | 0.393 | 0.516 | 0.575 | 1.000 | 0.636 | 0.575 |
| LongMemEval | extracted_facts | 1500 | 0.393 | 0.515 | 0.573 | 1.000 | 0.635 | 0.573 |
| LongMemEval | episodic | 1500 | 0.393 | 0.516 | 0.575 | 1.000 | 0.636 | 0.575 |
| LongMemEval | hybrid | 1500 | 0.393 | 0.516 | 0.575 | 1.000 | 0.636 | 0.575 |
| MemoryArena | no_memory | 4850 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| MemoryArena | verbatim | 4850 | 0.180 | 0.348 | 0.356 | 1.000 | 0.367 | 0.356 |
| MemoryArena | extracted_facts | 4850 | 0.613 | 0.785 | 0.794 | 1.000 | 0.804 | 0.794 |
| MemoryArena | episodic | 4850 | 0.190 | 0.350 | 0.359 | 1.000 | 0.370 | 0.359 |
| MemoryArena | hybrid | 4850 | 0.180 | 0.347 | 0.356 | 1.000 | 0.366 | 0.356 |

This is now paper-quality for the offline evaluator: semantic coverage is complete (`1.000`) across all datasets and strategies, no-memory remains a clean failure baseline, LongMemEval gives consistent retrieval-backed performance, and MemoryArena strongly favors extracted facts.

## 8. How to interpret the benchmark

Main tutorial takeaways:

1. **No-memory fails across all datasets.** This supports the proposal's motivation that memory is necessary for long-horizon autonomous systems.
2. **LoCoMo is difficult.** Even the best strategy, episodic memory, reaches only 0.311 hit rate.
3. **LongMemEval is more retrieval-friendly.** Strategies reach about 0.575 hit rate.
4. **MemoryArena strongly favors extracted facts.** Extracted facts reach 0.794 hit rate, much higher than verbatim, episodic, or hybrid.
5. **No single memory strategy dominates.** This supports the proposal's benchmarking argument.

The exact expected metric ranges in the proposal should be treated as teaching expectations, not guaranteed outcomes on every dataset. The current real datasets show the more nuanced result that architecture performance depends strongly on dataset and evidence structure.

## 9. Autoresearch-agent case study

Run:

```bash
latest_real=$(ls -t experiment/results/run_*_real_metrics.json | head -1)
python experiment/run.py \
  --mode autoresearch-agent \
  --backend offline \
  --benchmark-metrics "$latest_real" \
  --use-cases locomo autoresearch hpo memoryarena longmemeval lcbench \
  --ideas-per-case 2 \
  --top-k 5
```

The latest log is:

```text
experiment/results/logs/run_autoresearch_agent_modal_corrected.log
```

Latest outputs:

```text
experiment/results/run_20260509_151732_autoresearch_agent_raw.json
experiment/results/run_20260509_151732_autoresearch_agent_metrics.json
experiment/results/run_20260509_151732_autoresearch_agent_summary.tsv
experiment/results/run_20260509_151732_autoresearch_agent_report.md
```

Latest summary:

```text
ideas: 12
kept: 3
revised: 3
discarded: 6
mean_retrieval_precision: 0.2833
mean_retrieval_recall: 0.1181
memory_utilization_rate: 0.5833
redundant_idea_rate: 0.5
mean_novelty_score: 0.3432
mean_semantic_score: 0.3047
semantic_pass_rate: 0.1667
failure_modes: none=3, redundant_idea=6, retrieval_miss=2, low_signal_prior=1
```

Interpretation:

- The agent uses benchmark results as experiment memory.
- It proposes new research ideas.
- It retrieves prior result memories.
- It decides whether to keep, revise, or discard ideas.
- It diagnoses failures such as redundant ideas, retrieval misses, and low-signal priors.

This implements the proposal's autonomous research case study at tutorial scale.

## 10. OpenAI-compatible API path

Offline mode is default and does not need an API key.

For OpenAI-compatible runs:

```bash
export OPENAI_API_KEY
python experiment/run.py \
  --mode tutorial \
  --backend openai-compatible \
  --base-url http://127.0.0.1:8317/api/provider/codex/v1 \
  --model gpt-5.5
```

Never hardcode the real API key in notebooks, source files, JSON outputs, or reports.

The notebooks include API guidance but do not call the API unless the user explicitly switches to `--backend openai-compatible` and provides a key.

## 11. Rhesis and Semantica evaluator path

For reproducible tutorial runs, use:

```bash
python experiment/run.py --mode real --eval-backend offline --eval-limit 50
```

For Rhesis maintainer smoke tests, install `rhesis-sdk`, set `RHESIS_API_KEY` outside source files, then run:

```bash
python experiment/run.py --mode real --eval-backend rhesis --eval-smoke-test --eval-limit 1
```

For Semantica smoke tests, install `semantica`, then run:

```bash
python experiment/run.py --mode real --eval-backend semantica --eval-smoke-test --eval-limit 1
```

Notebook 10 contains these optional maintainer checks. It is not required for live participants.

## 12. Proposal-fit verdict

The tutorial is correct with respect to the proposal's core design:

- It is organized around the 3-hour problem-solving tutorial timeline.
- It includes the memory architecture taxonomy.
- It includes failure mode classification.
- It implements the three-probe diagnostic framework.
- It implements the standardized INGEST → INDEX → SEARCH → ANSWER → EVALUATE → REPORT pipeline.
- It compares multiple memory providers/strategies.
- It includes visual reports.
- It includes an autonomous research-agent memory-debug case study.
- It includes low-resource Colab guidance.
- It documents the OpenAI-compatible API option safely.

Remaining limitations:

- LCBench and HPOBench are optional extensions until installed.
- Full LongMemEval is larger than ideal for live Colab, so use the subset command during the tutorial.
- The offline evaluator uses evidence IDs/context hits, not a live LLM judge by default. The OpenAI-compatible path is documented and available.
