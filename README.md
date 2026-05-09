# KDD Tutorial Submission: Memory Diagnosis and Benchmarking for Autonomous Research Agents

This clean submission bundle contains the source code, notebooks, walkthrough, selected logs, final paper-quality metrics, and a static GitHub Pages website for the KDD tutorial experiment.

## Contents

```text
source/                  Python implementation for the tutorial benchmark
notebooks/               Numbered Jupyter/Colab notebooks and walkthrough
results/metrics/         Final metrics JSON files
results/summaries/       Final TSV summaries
results/reports/         Autoresearch-agent case-study report
results/figures/         Paper/tutorial figures
logs/                    Selected provenance logs
data/README.md           Dataset availability and download guidance
index.html               Static GitHub Pages homepage
```

## Final paper-quality outputs

The final Modal GPU run is included as:

- `results/metrics/run_20260509_151530_real_metrics.json`
- `results/summaries/run_20260509_151530_real_summary.tsv`

The final autoresearch-agent case study is included as:

- `results/metrics/run_20260509_151732_autoresearch_agent_metrics.json`
- `results/summaries/run_20260509_151732_autoresearch_agent_summary.tsv`
- `results/reports/run_20260509_151732_autoresearch_agent_report.md`

## Reproduce locally

From this folder:

```bash
python -m compileall -q source
python source/run.py --mode synthetic --backend offline --episodes 5
```

Full paper-quality runs require the real datasets described in `data/README.md`. The original run used the Modal GPU runner from `source/modal_runner.py`.

## Modal GPU command

```bash
python source/run.py \
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

Detached Modal runs print a call id. Fetch artifacts later with:

```bash
python source/run.py --runner modal --mode real --modal-call-id <call-id>
```

## GitHub Pages

Serve the static website locally with:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/`.

## Secret safety

This submission does not include API keys. OpenAI-compatible, Rhesis, and Modal credentials must be supplied through environment variables only.
