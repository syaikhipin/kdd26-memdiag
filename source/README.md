# KDD Tutorial Memory-Debug Experiment

This experiment follows the accepted KDD tutorial proposal: **Systematic Diagnosis and Benchmarking of Memory Systems in Autonomous AI Research Agents**.

It combines:

- `autoresearch`-style autonomous research loops and experiment traces
- `memory-probe`-style retrieval, utilization, and failure diagnostics
- real PDF-listed dataset usage through local **LoCoMo**
- proposal dataset registry for **LCBench**, **LoCoMo**, **MemoryArena**, **HPOBench**, and **LongMemEval**
- optional OpenAI-compatible idea generation through the local CCS endpoint
- optional Rhesis and Semantica evaluator integrations for paper-quality testing
- visualization outputs for tutorial/demo use

## Real datasets from the PDF

Currently available locally:

- **LoCoMo**: `memory-probe/data/locomo10.json`
- **LongMemEval**: `experiment/data/real/longmemeval/*.json`
- **MemoryArena**: `experiment/data/real/memoryarena/*.jsonl`

Included in the dataset registry while download/setup remains pending:

- **LCBench** — autonomous research / learning-curve traces
- **HPOBench** — HPO experiment traces

The default tutorial run uses LoCoMo directly and does not download anything. The `real` mode runs the downloaded real datasets.

## Colab/Jupyter notebooks

A numbered notebook sequence is available in `experiment/notebooks/`:

1. `1_setup_and_timeline.ipynb`
2. `2_dataset_registry_and_downloads.ipynb`
3. `3_memory_architectures.ipynb`
4. `4_three_probe_diagnostics.ipynb`
5. `5_full_real_benchmark.ipynb`
6. `6_visualization_dashboard.ipynb`
7. `7_autoresearch_agent_loop.ipynb`
8. `8_kdd_timeline_fit_analysis.ipynb`
9. `9_colab_guidance_and_next_steps.ipynb`
10. `10_external_testing_integrations.ipynb` — optional maintainer notebook for Rhesis/Semantica checks

See `experiment/notebooks/README.md` for Colab guidance and low-resource commands. Executed local variants are also written as `*.executed.ipynb`.

## Modes

### Synthetic compatibility mode

```bash
python experiment/run.py --mode dry-run --backend offline --episodes 5 --seed 0
```

### Real LoCoMo memory-debug run

```bash
python experiment/run.py \
  --mode locomo \
  --backend offline \
  --locomo-path memory-probe/data/locomo10.json \
  --max-conversations 1 \
  --max-questions 5 \
  --top-k 5
```

This ingests LoCoMo dialogue turns into memory and diagnoses retrieval against gold evidence IDs.

### Full real-dataset memory benchmark

```bash
python experiment/run.py \
  --mode real \
  --backend offline \
  --datasets locomo longmemeval memoryarena \
  --locomo-path memory-probe/data/locomo10.json \
  --longmemeval-dir experiment/data/real/longmemeval \
  --memoryarena-dir experiment/data/real/memoryarena \
  --max-conversations 999 \
  --max-questions 999 \
  --top-k 5 \
  --eval-backend offline \
  --eval-limit 50 \
  --visualize
```

This runs the available real datasets from the proposal without smoke-test caps: LoCoMo, LongMemEval, and MemoryArena. LCBench and HPOBench remain listed in the registry until their real benchmark artifacts are downloaded or installed.

### Autoresearch agent memory-debug loop

```bash
python experiment/run.py \
  --mode autoresearch-agent \
  --backend offline \
  --benchmark-metrics experiment/results/<real_run>_real_metrics.json \
  --use-cases locomo autoresearch hpo memoryarena longmemeval lcbench \
  --ideas-per-case 2 \
  --top-k 5
```

This is the closed-loop autonomous research-agent case study: benchmark results become experiment memories, the agent proposes novel memory-system ideas, retrieves prior results, decides keep/revise/discard, and diagnoses failures such as retrieval miss, redundant idea, low-signal prior, and retrieved-but-not-used memory.

### Autoresearch trace inspection

```bash
python experiment/run.py \
  --mode inspect-autoresearch \
  --autoresearch-dir autoresearch
```

If `autoresearch/results.tsv` or `autoresearch/run.log` exist, they are parsed. If not, the output records the missing traces without failing.

### Offline tutorial mode

```bash
python experiment/run.py \
  --mode tutorial \
  --backend offline \
  --locomo-path memory-probe/data/locomo10.json \
  --autoresearch-dir autoresearch \
  --use-cases locomo autoresearch hpo memoryarena longmemeval \
  --ideas-per-case 2 \
  --max-conversations 1 \
  --max-questions 5 \
  --visualize
```

### OpenAI-compatible tutorial mode

Use the requested local endpoint and model, but keep the API key out of source files:

```bash
export OPENAI_API_KEY
python experiment/run.py \
  --mode tutorial \
  --backend openai-compatible \
  --base-url http://127.0.0.1:8317/api/provider/codex/v1 \
  --model gpt-5.5 \
  --locomo-path memory-probe/data/locomo10.json \
  --autoresearch-dir autoresearch \
  --use-cases locomo autoresearch hpo memoryarena longmemeval \
  --ideas-per-case 2 \
  --max-conversations 1 \
  --max-questions 5 \
  --visualize
```

## Memory strategies

- `no_memory` — control baseline
- `verbatim` — stores raw evidence or experiment traces
- `extracted_facts` — stores compact success/failure facts
- `episodic` — stores compressed episode summaries
- `hybrid` — stores richer evidence strings and multi-tier-style metadata

## Modal GPU runner for paper-quality runs

Local runs remain the default. For full paper-quality evaluation without the local cap, use Modal with GPU allocation:

```bash
python experiment/run.py \
  --runner modal \
  --modal-gpu T4 \
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

Use a stronger Modal GPU such as `A10G` or `L40S` if your account has quota. Detached runs print a Modal call id; fetch artifacts later with `python experiment/run.py --runner modal --mode real --modal-call-id <call-id>`. Rhesis keys and Modal credentials must stay in environment variables only.

## External semantic/testing evaluators

The default evaluator is deterministic and local:

```bash
python experiment/run.py --mode real --eval-backend offline --eval-limit 50
```

Optional Rhesis smoke test, after installing `rhesis-sdk` and setting the key outside source files:

```bash
export RHESIS_API_KEY
python experiment/run.py --mode real --eval-backend rhesis --eval-smoke-test --eval-limit 1
```

Optional Semantica smoke test, after installing `semantica`:

```bash
python experiment/run.py --mode real --eval-backend semantica --eval-smoke-test --eval-limit 1
```

The raw key is never written to source, notebooks, JSON, TSV, Markdown reports, or figures. Output config records only whether the key was present.

## Metrics

The experiment reports the tutorial metrics:

- retrieval precision and recall
- evidence hit rate
- memory utilization rate
- failure categories
- latency and cost units
- memory growth
- novelty score for generated ideas
- redundant idea rate
- dataset availability coverage
- autoresearch keep/discard/crash counts when traces are present
- semantic score, pass rate, faithfulness, context relevance, and answer correctness when semantic evaluation is enabled

## Visualizations

With `--visualize`, figures are written under `experiment/results/`:

- `figure_locomo_retrieval.png`
- `figure_real_retrieval.png`
- `figure_failures.png`
- `figure_idea_novelty.png`
- `figure_memory_growth.png`
- `figure_semantic_scores.png` when semantic metrics exist
- `figure_semantic_vs_retrieval.png` when semantic metrics exist
- `figure_evaluator_coverage.png` when semantic metrics exist
- `figure_autoresearch_trace.png` when traces exist

## API key safety

No API key is needed for offline modes. For `--backend openai-compatible`, pass the key through `OPENAI_API_KEY` or `--api-key`. For Rhesis evaluation, pass the key through `RHESIS_API_KEY`. Keys are never written to JSON, TSV, Markdown reports, or figures.
