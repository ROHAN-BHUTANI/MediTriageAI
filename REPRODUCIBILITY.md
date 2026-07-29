# Reproducibility Guarantee

MediTriageAI strictly adheres to research-grade determinism protocols.

## Seed Anchoring

All pseudo-random number generators (PRNGs) are strictly anchored prior to any model instantiation, dataset sampling, or pipeline initialization.

- `torch.manual_seed(42)`
- `numpy.random.seed(42)`
- `random.seed(42)`

## Environment Freezing

- The execution environment must match `requirements.txt` / `environment.yml` exactly.
- Parquet chunks and deduplication algorithms (MD5 hashing of `raw_text` strings) are mathematically stable independent of hardware environments.

## Validating Reproducibility

You can execute a smoke test loop to verify end-to-end configuration reproducibility:

```bash
pytest tests/
python -m scripts.run_experiment --mode smoke
```
