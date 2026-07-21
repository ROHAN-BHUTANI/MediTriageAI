# MediTriageAI Analysis Framework

This directory contains a modular, publication-quality, and reproducible error analysis and model calibration framework for the `MediTriageAI` multilingual clinical triage system.

---

## Directory Layout

```
analysis/
├── __init__.py           # Package entrypoint mapping public modular APIs
├── README.md             # This documentation file
├── config.py             # Centralized settings (bootstrap runs, thresholds, DPI, paths)
├── io.py                 # Data loaders, batch inference runner, Parquet caching
├── metrics.py            # Standard, Top-k, and Bootstrap accuracy/F1 metrics
├── calibration.py        # Calibration calculations (ECE, MCE, Brier, NLL)
├── agreement.py          # Pairwise percentage agreement and Cohen's Kappa
├── taxonomy.py           # Error taxonomy classification rules
├── language_detector.py  # Language identification API and rule heuristics
├── visualization.py      # Plotting helper classes (300 DPI heatmaps, reliability curves)
├── report.py             # Markdown and HTML publication compilers
├── utils.py              # Statistical McNemar test, seeding, logging, file hashing
├── run.py                # Command-line entrypoint and self-validation assertions
└── cache/                # Inter-run data caching
    ├── predictions/      # Optimized Apache Parquet cached predictions per model
    ├── embeddings/       # Placeholder subfolder for future representation experiments
    └── metadata/         # Saved YAML metadata from runs
```

---

## Pipeline Stages

Executing `python analysis/run.py` triggers the following pipeline sequence:

1. **Randomness Seeding**: Initializes random seeds across `random`, `numpy`, `torch`, and `torch.cuda` for deterministic replication.
2. **Prediction Generation & Parquet Caching**: Restores model checkpoints, runs batched evaluation over the test dataset split, compiles predictions with logits/probabilities, and saves them to `.parquet` caches.
3. **Metric Calculations & Bootstrapping**: Computes performance metrics (Specialist & Severity accuracies/F1s) alongside **95% Bootstrap Confidence Intervals** using 1,000 resamples.
4. **Calibration Analysis**: Calculates ECE, MCE, NLL, and Brier score. Saves reliability diagram curves and confidence histograms.
5. **Consensus & Significance Evaluation**: Constructs pairwise agreement matrices (Cohen's Kappa and percentage agreement) and runs McNemar significance tests.
6. **Stratifications**: Evaluates subsets partitioned by heuristic-detected language, sentence word count buckets, and rare classes.
7. **Report Compilation**: Assembles the findings into `analysis_report.md` and `analysis_report.html` embedding the relevant visual plots.
8. **Manifest Generation**: Generates `manifest.json` cataloging all generated files with their SHA256 hashes.
9. **Reproducibility Verification Check**: Asserts the presence and non-emptiness of all tables, figures, caches, and report files, outputting a verification checklist.

---

## CLI Usage

Run the entire pipeline using the default parameters:

```bash
python analysis/run.py
```

Settings such as the global seed, bootstrap iterations, plot DPI, and path locations can be configured in `analysis/config.py`.

---

## Cache Format (Apache Parquet)

Each cached model output file (e.g. `analysis/cache/predictions/xlm_roberta_large.parquet`) includes the following columns:
- `sample_id`: Unique string ID (usually tracking_id).
- `text`: Input clinical description.
- `true_specialist`: Ground truth specialist code (string).
- `pred_specialist`: Model predicted specialist code (string).
- `true_severity`: Ground truth severity code (string).
- `pred_severity`: Model predicted severity code (string).
- `specialist_logits`: List of raw specialist output logits.
- `severity_logits`: List of raw severity output logits.
- `specialist_probabilities`: Softmax probability distribution over the 13 specialist classes.
- `severity_probabilities`: Softmax probability distribution over the 5 severity classes.
- `language`: Detected language group.
- `token_count`: Whitespace token count of the text.
- `model_name`: Model name identifier.

---

## Extending the Framework

### Adding New Analyses
To add a new statistical analysis (e.g., lexical density vs. accuracy):
1. Implement the metrics function in `analysis/metrics.py`.
2. Update `analysis/run.py` to extract the metric from `enriched_dict` and save the resulting table.
3. Include the plot function in `analysis/visualization.py` if plotting is needed.
4. Update `analysis/report.py` to compile the new figures and tables into the reports.

### Integrating Future Models
To integrate a new model:
1. Define your model subclass under the `models/` directory, subclassing `BaseMediTriageModel` (inheriting `needs_vocab_injection`, `build`, etc.).
2. Add the model class mapping into `analysis/io.py` in the `MODEL_MAP` dictionary.
3. Register the model short name in the `MODELS_TO_ANALYZE` list in `analysis/config.py`.
4. Ensure the trained checkpoint is saved under `results/<model_short_name>/checkpoint.pt`.
5. Run `python analysis/run.py` to generate prediction caches and perform comparisons. No existing analysis logic needs to change.
