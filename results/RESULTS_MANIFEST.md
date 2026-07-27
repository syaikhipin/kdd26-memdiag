# Results manifest

This file documents exactly what the committed result artifacts are, so tutorial organizers and
participants interpret them honestly.

## Canonical committed sample (honest, rebuilt)

`metrics/sample_real_metrics.json` (+ `summaries/sample_real_summary.tsv`,
`summaries/sample_real_utilization_lab.tsv`, `metrics/sample_real_raw.json`, `figures/*.png`) is a
**reproducible tutorial sample** produced by the rebuilt package — four genuinely distinct memory
providers, a 3-arm contrastive utilization probe, and a non-circular LLM-judge score.

| Dataset | Questions/items in this sample |
|---|---:|
| LoCoMo | 40 |
| LongMemEval (oracle subset) | 50 |
| MemoryArena (formal_reasoning_phys) | 86 |

It runs offline in seconds on a laptop with **no API key**. The committed numbers are honest and
citable as a *tutorial-scale* sample (not paper-scale). Full-scale runs (LoCoMo ~1540, LongMemEval
~1500, MemoryArena ~4850) require the complete datasets and the Modal GPU runner
(`source/modal_runner.py`).

Example headline numbers (LoCoMo, retrieval precision / recall, and `util` now decoupled from hit):

| strategy | retr_p | retr_r | evidence_hit | memory_utilized |
|---|---:|---:|---:|---:|
| no_memory | 0.000 | 0.000 | 0.000 | 0.000 |
| verbatim | 0.040 | 0.188 | 0.200 | 0.325 |
| extracted_facts | 0.040 | 0.188 | 0.200 | 0.400 |
| episodic | 0.000 | 0.000 | 0.000 | 0.375 |
| hybrid | 0.003 | 0.125 | 0.125 | 0.275 |

The strategies are genuinely distinct, and utilization is no longer a restatement of retrieval hit
(e.g. episodic `evidence_hit=0.000` but `memory_utilized=0.375` — its summaries mention the answer
without retrieving the exact gold turn).

## Autoresearch topic trace

The `Autoresearch` topic (`source/data/topics/autoresearch/`) holds the real LLM-pretraining
`val_bpb` trace from `autoresearch/train.py` run on Modal H100 via
`scripts/build_autoresearch_trace.py`. If the directory is empty, run that script on Modal to
(re)generate it. (Earlier proposal-listed LCBench/HPOBench stubs were removed in favour of this
authentic single-GPU autonomous-research trace.)

## How to regenerate

From the repository root (offline, no key):

```bash
python source/run.py --mode real --backend offline \
  --datasets locomo longmemeval memoryarena \
  --max-questions 40 --max-items 100 --top-k 5 --visualize
```

New runs are written flat to `results/` (e.g. `results/run_<timestamp>_real_metrics.json`). To
refresh the committed canonical sample, copy the newest run into the stable `sample_real_*` names
referenced by `index.html`. Pin a specific run by timestamp; do not select "latest file" blindly.
