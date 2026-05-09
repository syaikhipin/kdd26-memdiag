# Data Availability

This clean GitHub submission intentionally does not vendor the full real datasets.

## Included

- `source/data/synthetic_research_tasks.json` — small synthetic data for local smoke tests and tutorial demonstrations.

## Not included

The following real datasets were used for the final benchmark but are not bundled because of size and licensing/distribution considerations:

- LoCoMo
- LongMemEval
- MemoryArena
- optional LCBench and HPOBench extensions

The full LongMemEval artifacts used locally were about 2.8 GB, so they are unsuitable for a lightweight GitHub Pages submission. The benchmark code expects the real data layout documented in `source/README.md` and `notebooks/WALKTHROUGH.md`.

## Recommended reproduction workflow

1. Run the notebooks in order for the low-resource tutorial path.
2. Download/provide the real datasets locally following each dataset's upstream instructions.
3. Use `source/run.py --mode real` for local runs.
4. Use `source/run.py --runner modal --modal-gpu A10G` for full paper-quality runs.

No API keys are stored in this submission. Use environment variables for external services.
