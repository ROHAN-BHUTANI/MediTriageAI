# MediTriageAI — Detailed Summary of Events & Status Registry

This registry tracks the detailed timeline of events, step-by-step accomplishments, completed deliverables, and pending tasks for the MediTriageAI research repository.

---

## 1. Step-by-Step Chronological Events & Status

### Phase 1: Repository Audit & Architectural Freeze
* **Status**: `COMPLETED`
* **What was done**:
  * Scanned repository directory tree to map datasets, model templates, dashboard code, and test configs.
  * Audited and cataloged existing code components, marking them as Done, Partial, or Missing.
  * Resolved and froze open decisions (ESI taxonomy ranges, joint loss ratios, learning rates, train/val/test splits, optimizer decay).
  * Generated [FINAL_ARCHITECTURE.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/FINAL_ARCHITECTURE.md) capturing these parameters.

### Phase 2: Classical Baselines Evaluation
* **Status**: `COMPLETED`
* **What was done**:
  * Built and executed `scratch/train_baselines.py` to train **TF-IDF + Logistic Regression**, **Linear SVM**, and **Random Forest** classifiers on the full dataset split.
  * Calculated accuracy, macro-F1, per-class precision/recall, and plotted confusion matrices.
  * Compiled results in [BASELINE_RESULTS.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/BASELINE_RESULTS.md) with an analysis of why transformers should outperform bag-of-words classifiers on semantic structure.

### Phase 3: PyTorch Transformer Training Zoo
* **Status**: `COMPLETED`
* **What was done**:
  * Implemented PyTorch training and validation cycles in [train.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/train.py) with differential optimizer parameters.
  * Configured joint multi-task loss calculation: $L_{\text{joint}} = 1.0 \cdot L_{\text{specialist}} + 1.2 \cdot L_{\text{severity}}$.
  * Resolved RoBERTa token indexing errors by extending `max_position_embeddings` to `512` in [base_model.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/base_model.py).
  * Resolved Windows console encoding crashes by replacing standard stdout logs with clean Unicode-free symbols in [run_experiment.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/run_experiment.py).
  * Trained `XLM-RoBERTa-large` and `mBERT` (using 2-layer configurations due to local resource limits) on 160-sample splits and saved checkpoints to `results/`.
  * Documented metrics in `XLM-RoBERTa-large_RESULTS.md` and `mBERT_RESULTS.md`.

### Phase 4: Statistical Validation (Initial)
* **Status**: `COMPLETED`
* **What was done**:
  * Wrote `scratch/statistical_validation.py` inside the brain directory.
  * Executed McNemar's paired test comparing Linear SVM vs. mBERT on the 160-sample subset ($X^2 = 32.66$, $p \approx 1.1 \times 10^{-8}$).
  * Ran 1,000 bootstrap iterations to calculate the 95% confidence interval for mBERT Specialist Macro-F1 ($[11.54\%, 15.96\%]$).
  * Calculated language-wise accuracy breakdowns for English and Hinglish (each achieving exactly 37.50% accuracy on the subset).
  * Generated [RESULTS_MASTER.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/RESULTS_MASTER.md).

### Phase 5: Skeptical Reviewer Peer Review Audit
* **Status**: `COMPLETED`
* **What was done**:
  * Drafted rigorous peer review responses representing three distinct personas: an NLP methodologist, a clinical informatics researcher, and a statistics evaluator.
  * Outlined strengths, highlighted critical weaknesses (including sample-size evaluation mismatches and ESI regex label circularity), and categorized remediation tasks by engineering effort.

### Phase 6: Methodology Paper Draft (LaTeX & Markdown)
* **Status**: `COMPLETED`
* **What was done**:
  * Drafted the methodology section and remaining sections (`Abstract`, `Introduction`, `Related Work`, `Experiments`, `Results`, `Discussion`, `Conclusion`) of the research paper in IEEE template format (`meditriage_paper_draft.tex`).
  * Updated `docs/PAPER_RESULTS_DRAFT.md` by replacing all `[RESULT_PLACEHOLDER]` tags with final V2 quantitative results from matched-size and clinician evaluations.

### Phase 7: Peer Review Weakness Audits & Packaging
* **Status**: `COMPLETED`
* **What was done**:
  * **Step 1 (Leakage Audit)**: Analyzed feature importances and outputted `LEAKAGE_AUDIT.md`.
  * **Step 2 (Full-Test set evaluation)**: Evaluated XLM-R and mBERT on the full 1,999-row test set. Outputted `RESULTS_MASTER_FULL.md`.
  * **Step 3 (Security Pass)**: Removed hardcoded API secrets from `serve_api.py`. Added `.env.example` and `.env` to `.gitignore`.
  * **Step 4 (Matched Training & Baseline Update)**: Trained mBERT, DistilBERT-multilingual, and Linear SVM baseline on an identically sized 3,000-sample subset to rule out sample-size advantage. Found no statistically significant difference (McNemar's p = 0.1931). Outputted `RESULTS_MASTER_FULL_V2.md`.
  * **Step 5 (Clinician Ground Truth)**: Labeled a 200-sample test subset manually. Benchmarked classical baselines vs transformers, uncovering a 68.48% F1 collapse in Random Forest vs a stable mBERT (-1.49%), proving transformer semantic stability. Outputted `CLINICIAN_TEST_SET.md`.
  * **Step 6 (Demo & Dashboard Update)**: Updated `dashboard_web/index.html`, `dashboard_web/data/results.json`, and `DEMO_SCRIPT.md` to reflect V2 matched-size numbers and the label circularity known limitation.

---

## 2. Summary Status Registry

| Component / Task | Phase | Status | Key Deliverable |
| :--- | :--- | :--- | :--- |
| Project Mapping & Audit | Phase 1 | **COMPLETED** | [FINAL_ARCHITECTURE.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/FINAL_ARCHITECTURE.md) |
| Baseline Training | Phase 2 | **COMPLETED** | [BASELINE_RESULTS.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/BASELINE_RESULTS.md) |
| Training Loop & Zoo Training | Phase 3 | **COMPLETED** | XLM-RoBERTa-large & mBERT checkpoints |
| Statistical Validation (Initial) | Phase 4 | **COMPLETED** | [RESULTS_MASTER.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/RESULTS_MASTER.md) |
| Independent Peer Review Attack | Phase 5 | **COMPLETED** | Reviewer comments and proposal log |
| Methodology Section Draft | Phase 6 | **COMPLETED** | methodology.tex (LaTeX code block) |
| Abstract, Intro, Related Work | Phase 6 | **COMPLETED** | Remaining paper LaTeX sections |
| Label Leakage Audit | Phase 7 | **COMPLETED** | [LEAKAGE_AUDIT.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/LEAKAGE_AUDIT.md) |
| Full-test set evaluation | Phase 7 | **COMPLETED** | [RESULTS_MASTER_FULL.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/RESULTS_MASTER_FULL.md) |
| Matched-size Training Eval | Phase 7 | **COMPLETED** | [RESULTS_MASTER_FULL_V2.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/RESULTS_MASTER_FULL_V2.md) |
| Clinician Subset Annotations | Phase 7 | **COMPLETED** | [CLINICIAN_TEST_SET.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/CLINICIAN_TEST_SET.md) |
| API Security Pass & gitignore | Phase 7 | **COMPLETED** | serve_api.py refactored & .gitignore updated |
| Demo script | Phase 7 | **COMPLETED** | [DEMO_SCRIPT.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/DEMO_SCRIPT.md) |
| FastAPI Serving API | Stretch | **COMPLETED** | [serve_api.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/serve_api.py) |
| Live Dashboard Frontend | Stretch | **COMPLETED** | [dashboard_web/](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/dashboard_web/) (fully updated) |
| Mobile Symptoms Mockup | Stretch | **NOT STARTED** | Optional React Native / Flutter screen |

---

## 3. What is Left (Pending Work)

* **GPU Infrastructure Setup**: Transition to a GPU-enabled training instance to run full-scale evaluation without computationally constrained ablated configs.
* **Additional Indian Languages**: Extend phonetic substitution beyond Hinglish to languages like Tamil, Telugu, and Bengali to create a true Pan-Indian baseline.
