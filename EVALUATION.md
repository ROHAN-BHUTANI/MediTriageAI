# MediTriageAI Evaluation Guide

Post-training evaluation is built intrinsically into the orchestration loop, guaranteeing that model performance is measured consistently across identically seeded validation subsets.

## Core Metrics
We calculate multi-task metrics dynamically using `scikit-learn`:

- **Macro-F1 (Specialist)**: Measures the department classification accuracy.
- **Macro-F1 (Severity)**: Measures the triage urgency classification accuracy.
- **Adjusted Error Rate**: Calculates weighted prediction penalties for under-triage compared to over-triage severity errors.

## Exporting Metrics
During a successful execution loop, evaluated metrics are automatically written to `dashboard_web/data/results.json`. 

```bash
# Generate metrics specifically bypassing training via:
python -m scripts.run_experiment --mode smoke
# Then select [8] Export dashboard data.
```

The Web Dashboard uses `results.json` directly to render comparison bar charts.