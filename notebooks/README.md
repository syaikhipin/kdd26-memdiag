# KDD Tutorial Colab/Jupyter Notebooks

Run these notebooks in order for the accepted KDD tutorial workflow.

## Notebook sequence

1. `1_setup_and_timeline.ipynb` — environment check and mapping to the 3-hour KDD timeline.
2. `2_dataset_registry_and_downloads.ipynb` — local availability of LoCoMo, LongMemEval, and MemoryArena; LCBench and HPOBench are optional extensions when their artifacts are installed.
3. `3_memory_architectures.ipynb` — no-memory, verbatim, extracted-facts, episodic, and hybrid memory strategies.
4. `4_three_probe_diagnostics.ipynb` — retrieval relevance, context utilization, and failure root-cause probes.
5. `5_full_real_benchmark.ipynb` — full real-dataset benchmark command and result inspection.
6. `6_visualization_dashboard.ipynb` — regenerate/display tutorial figures.
7. `7_autoresearch_agent_loop.ipynb` — autonomous research-agent idea loop and memory debugging.
8. `8_kdd_timeline_fit_analysis.ipynb` — artifact-to-timeline fit analysis.
9. `9_colab_guidance_and_next_steps.ipynb` — low-resource Colab commands and packaging checklist.
10. `10_external_testing_integrations.ipynb` — optional maintainer notebook for Rhesis/Semantica semantic testing.

## Colab guidance

Upload or clone the repository into Colab, then run notebook 1 first. If auto-detection fails, edit `PROJECT_ROOT` in the first code cell to point to the repository root.

For a fast participant run, prefer the low-resource command from notebook 9:

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

For the full local run, use the complete command in notebook 5. The full LongMemEval medium file is large, so it is better suited to Modal or local maintainer machines than live Colab sessions.

For paper-quality runs, notebook 5 includes a Modal GPU command gated by `RUN_MODAL_FULL_BENCHMARK=1`. Use notebook 10 only for optional maintainer validation with Rhesis or Semantica. Keep real Rhesis keys and Modal credentials in environment variables, not in notebooks.
