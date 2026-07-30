# Release Checklist v1.0.0

- [x] Ensure all multi-GPU integration tests pass.
- [x] Run automated linting (`flake8` / `ruff`) and formatting (`black`).
- [x] Delete `scratch/` files, log files, and intermediate `.csv` artifacts.
- [x] Finalize `dataset_adapters.py` mappings (string integers fixed for NHAMCS).
- [x] Lock configuration defaults and seeds for reproducibility.
- [x] Regenerate all markdown documentation (API, Dataset, Deploy, etc.).
- [x] Generate CITATION.cff and CITING.md for academic users.
- [x] Generate `VERSION` file set to `1.0.0`.
- [x] Freeze repository functionality.
