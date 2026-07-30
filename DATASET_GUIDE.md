# Dataset Engine Guide

The Data Engine is designed to compile highly diverse medical texts into a unified tensor-ready stream.

## Supported Acuity Levels
All incoming severity variables are normalized to a strict `1-5` integer mapping:
- `1`: Resuscitation (Immediate Life-Threatening)
- `2`: Emergent
- `3`: Urgent
- `4`: Less Urgent
- `5`: Non-Urgent

## Supported Departments
The `TriageSchema` restricts outputs to 13 target canonical departments:
`"Emergency", "Internal Medicine", "Pediatrics", "Surgery", "Orthopedics", "Obstetrics & Gynecology", "Psychiatry", "Dermatology", "Ophthalmology", "Neurology", "Cardiology", "Oncology", "UNKNOWN"`

## Adding a New Dataset Adapter
If you want to add a novel dataset (e.g. from Kaggle or HuggingFace):
1. In `src/dataset_adapters.py`, create a subclass of `DatasetAdapter`.
2. Override the `clean()` method. 
3. Coerce the custom dataset labels into the aforementioned 5 severities and 13 departments. Leave as `UNKNOWN` or `None` if missing.
4. Add the adapter to the global registry in `src/dataset_registry.py`.
5. Re-run `scripts/run_baseline.py`. Deduplication will automatically handle duplicate texts across your new source and existing sources.
