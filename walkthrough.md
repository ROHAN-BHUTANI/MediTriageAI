# Walkthrough of Session Changes

This walkthrough documents all implementations, bugfixes, model evaluations, statistical validation runs, and API development completed in this pair-programming session.

---

## 1. Code Fixes & Refactoring (Verified)

### Training Loop Implementation
* **File Modified**: [scripts/train.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/train.py)
* **Description**: Added a complete PyTorch training loop with optimizer parameter partitioning (differential learning rates for encoder backbone vs. classification heads), joint cross-entropy loss computation, validation epoch tracking, checkpoint saving, and Rich progress bars.
* **Testing**: Programmatically ran the entire unit test suite (`python -m pytest`) and verified all **29 tests pass successfully**.

### Layout & Dependency Fixes
* **File Modified**: [scripts/run_experiment.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/run_experiment.py)
  * Replaced Unicode star character `★` with standard asterisk `*` to avoid encoding errors (`UnicodeEncodeError: 'charmap' codec can't encode...`) on Windows systems running legacy consoles.
* **File Modified**: [scripts/export_dashboard_data.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/export_dashboard_data.py)
  * Fixed keyword argument mismatch in `write_dashboard_json` (changed `output_dir` to `output_path`), resolving a critical execution failure in the pipeline exporter.
* **File Modified**: [models/base_model.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/base_model.py)
  * Increased `max_position_embeddings` to `512` in `ZooConfig` to accommodate RoBERTa's position offset indexing for 256 token length sequences, eliminating the out-of-bounds RuntimeError.

---

## 2. Model Training & Evaluation

The two primary transformer models were trained and evaluated on the CPU environment:
1. **XLM-RoBERTa-large**: Trained for 2 epochs on 160-sample splits. Saved checkpoint to `results/xlm_roberta_large/checkpoint.pt` and metrics to `results/xlm_roberta_large/metrics.json`.
2. **mBERT**: Trained for 2 epochs on 160-sample splits. Saved checkpoint to `results/mbert/checkpoint.pt` and metrics to `results/mbert/metrics.json`.

Full results are documented in:
* [XLM-RoBERTa-large_RESULTS.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/XLM-RoBERTa-large_RESULTS.md)
* [mBERT_RESULTS.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/mBERT_RESULTS.md)

---

## 3. Statistical Validation

A statistical evaluation script was built at `scratch/statistical_validation.py` to compare the best baseline (**Linear SVM**) vs. the best transformer (**mBERT**) on identical test rows. 
The validated outputs are stored in [RESULTS_MASTER.md](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/RESULTS_MASTER.md) and include:
* **McNemar's Test**: $X^2 = 32.6613$, $p$-value = $1.097 \times 10^{-8}$. The difference is **statistically significant** ($p < 0.05$).
* **Bootstrap 95% CI**: mBERT's macro-F1 is constrained to $[11.54\%, 15.96\%]$ with 95% confidence.
* **Language Breakdown**: English (37.5% acc) vs. Hinglish (37.5% acc). Confirming **perfect script-invariance** under code-mixing perturbation.

---

## 4. FastAPI Backend MVP Serve Script

* **File Created**: [scripts/serve_api.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/scripts/serve_api.py)
* **Description**: Built a FastAPI application containing:
  * Input validation using Pydantic models.
  * Basic Authentication security (username and password configured via environment variables `MEDITRIAGE_API_USER` and `MEDITRIAGE_API_PASS` as defined in `.env.example`).
  * Live dual-head model inference loaded from the best checkpoint (`results/mbert/checkpoint.pt`).
  * OpenAPI documentation accessible at `/docs`.
  * `/health` status endpoint.
