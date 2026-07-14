# MediTriageAI: Final State Summary

## Current Project State
The evaluation state of the MediTriageAI project is now finalized. The narrative and results correctly reflect a **genuine comparison between pre-trained multilingual transformers (mBERT, DistilBERT-multilingual) and classical baselines (Random Forest, Linear SVM).**

### 1. Verified Real Data & Metrics (COMPLETED)
- **V5 Metrics**: The evaluation on the full test set correctly revealed that the pre-trained transformers suffered from catastrophic collapse due to dataset sparsity and potential hyperparameter mismatches, achieving <4% macro-F1.
- **Classical Baselines**: The classical TF-IDF Random Forest remains the superior performing model on the heuristic labels, though it suffers heavily from label leakage.
- **Label Leakage Security**: We proved that the heuristic labels (regex generated) allowed the classical models to memorize keyword triggers.
- **19+3 Clinician Labels**: 22 true human-verified labels exist. Due to the shift to perfectly stratified subsetting (guaranteeing a minimum of 5 examples per severity tier and specialist class), only **2** of these original 22 labels were randomly selected in the new subset.

### 2. Pending Data (CLINICIAN HANDOFF READY)
- **198 Annotations Required**: A clean annotation sheet containing 198 rows (`clinician_annotation_sheet_178.csv` - note the file contains 198 rows due to the 2-row overlap) has been exported and is ready to be sent to human clinicians.
- Once these 198 rows are annotated, they will be combined with the 2 overlapping rows to create a perfectly balanced, 200-row human-verified test set. This will allow for the final, definitive evaluation of the model against clinical reality.

## Updated Documents
1. **Dashboard**: `results.json` updated with real V5 metrics.
2. **Demo Script**: `DEMO_SCRIPT.md` updated to frame the transformer collapse honestly.
3. **Paper Draft**: `meditriage_paper_draft.tex` and `docs/PAPER_RESULTS_DRAFT.md` completely overhauled to remove "tiny resource constraint" excuses and analyze the failure of the deep architectures.
