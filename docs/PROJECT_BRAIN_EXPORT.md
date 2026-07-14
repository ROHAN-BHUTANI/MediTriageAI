# MediTriageAI Brain Export

## Final Tree
- `src/model.py` - 42 lines
- `src/dataset.py` - 68 lines
- `src/dashboard.py` - 40 lines
- `src/specialty_mapping.py` - 76 lines
- `src/severity_heuristic.py` - 28 lines
- `src/hinglish_perturbation.py` - 81 lines
- `src/leakage_safe_split.py` - 66 lines
- `src/vocab_injection.py` - 52 lines
- `src/metrics.py` - 258 lines  # Updated
- `models/base_model.py` - 115 lines
- `models/xlm_roberta.py` - 27 lines
- `models/mbert.py` - 26 lines
- `models/distilbert_multi.py` - 35 lines
- `models/indic_bert.py` - 32 lines
- `scripts/evaluate.py` - 104 lines
- `scripts/export_dashboard_data.py` - 132 lines
- `scripts/run_experiment.py` - 156 lines
- `scripts/infer.py` - 101 lines
- `scripts/train.py` - 69 lines
- `dashboard_web/index.html` - 82 lines
- `Frontend/code.html` - 158 lines
- `docs/PAPER_RESULTS_DRAFT.md` - 34 lines
- `tests/test_metrics.py` - 64 lines
- `tests/test_model_zoo.py` - 80 lines
- `tests/test_export_dashboard.py` - 44 lines
- `tests/test_run_experiment.py` - 31 lines
- `results/.gitkeep` - 0 lines

## Validation
- Tests: 29 / 29 passed (core tests) + additional test coverage for fixes
- Imports: all `src/` and `models/` imports resolve
- Dashboard export: `python scripts/export_dashboard_data.py --dry-run` passed
- Metrics module: All `_asarray` → `_as_array` fixes applied, LaTeX generation corrected
- Models: IndicBERT get_special_loading_notes() updated to specification
- All files remain under 300 line limit

## Cleanup
- Legacy `meditriage/` code and tests removed.
- Cache and scratch files removed from the tracked tree.
- Every tracked code file remains under 300 lines.

## Build Progress
✓ D1. src/metrics.py + tests/test_metrics.py - FIXED
✓ D2. models/ directory - VERIFIED/CORRECTED
✓ D3. scripts/run_experiment.py - VERIFIED
✓ D4. scripts/evaluate.py - VERIFIED
✓ D5. scripts/infer.py - VERIFIED
✓ D6. scripts/export_dashboard_data.py - VERIFIED
✓ D7. dashboard_web/ - EXISTING
✓ D8. scripts/serve_dashboard.py - EXISTING
✓ D9. docs/PAPER_RESULTS_DRAFT.md - EXISTING
▢ D10. Final test suite + update PROJECT_BRAIN_EXPORT.md - IN PROGRESS
