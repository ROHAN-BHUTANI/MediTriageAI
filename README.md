# MediTriageAI Data Engine

Welcome to the MediTriageAI repository. This repository encapsulates the full end-to-end data processing, model training, and evaluation lifecycle.

## Documentation Links
- [Architecture Guide](ARCHITECTURE.md)
- [Dataset Guide](DATASET.md)
- [Training Guide](TRAINING.md)
- [Evaluation Guide](EVALUATION.md)
- [Reproducibility Guarantee](REPRODUCIBILITY.md)

## Getting Started
`ash
conda env create -f environment.yml
conda activate meditriageai
python -m scripts.run_experiment --mode smoke
`
# MediTriageAI: Hybrid Clinical Triage Architecture

![MediTriageAI Architecture](docs/architecture.png)

*A robust, publication-ready machine learning pipeline for clinical triage and severity assessment.*

## Research Disclaimer & Ethics Statement
**WARNING:** This repository contains a RESEARCH PROTOTYPE. The models and heuristics provided are NOT clinically validated and must NOT be used for real-world patient triage, diagnostic decisions, or medical advice.
*Ethics:* The dataset used for training contains de-identified clinical transcriptions. Any adaptation of this work for clinical environments must undergo rigorous fairness, bias, and clinical safety audits.

## Overview
MediTriageAI employs a Hybrid Triage Architecture designed to mitigate the risks of purely automated ML decision-making. It combines:
1. **Rule-Based Red-Flag Layer**: Deterministic heuristics for immediate escalation of critical keywords (e.g., stroke, chest pain).
2. **Top-3 Specialist Shortlist**: Probabilistic routing using a multilingual transformer architecture (e.g., mBERT, DistilBERT) to reduce taxonomic overlap errors.
3. **Low-Confidence Escalation**: Automatic flagging of cases where the model's confidence falls below safety thresholds.

## Table of Contents
- [Installation Guide](#installation-guide)
- [Quick Start](#quick-start)
- [Dataset Guide](#dataset-guide)
- [Training Guide](#training-guide)
- [Evaluation Guide](#evaluation-guide)
- [Deployment Guide](#deployment-guide)
- [Model Cards](#model-cards)

## Installation Guide
### Prerequisites
- Python 3.10+
- PyTorch (CUDA recommended)
- Node.js (for the dashboard)

### Setup
```bash
git clone https://github.com/ROHAN-BHUTANI/MediTriageAI.git
cd MediTriageAI
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements_colab.txt
```

## Quick Start
Run the FastAPI inference endpoint locally:
```bash
export MEDITRIAGE_API_USER=admin
export MEDITRIAGE_API_PASS=secret
uvicorn scripts.serve_api:app --reload
```
Test the endpoint:
```bash
curl -u admin:secret -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Patient has severe chest pain and radiation to the left arm."}'
```

## Dataset Guide
The canonical dataset is located at `meditriage/data/processed/dataset.csv`. It contains pre-processed clinical transcriptions mapped to consolidated specialist labels (5 supergroups) to reduce noise. Ensure you maintain this structure for further training.

## Training Guide
Due to computational constraints, training on large sequence lengths (`max_length=128`) requires dedicated GPU hardware. 
A Colab-ready script is provided:
```bash
python scripts/colab_train.py
```
*See `COLAB_SETUP.md` for detailed instructions.*

## Evaluation Guide
Evaluation is tracked in `RESULTS.md`. To run a local evaluation suite across your checkpoints:
```bash
python scripts/evaluate.py --data meditriage/data/processed/dataset.csv
```

## Deployment Guide
The pipeline is designed for containerized deployment. A standard Dockerfile can wrap the FastAPI service in `scripts/serve_api.py`. Ensure API credentials are injected via secure environment variables.

## Model Cards
- **Architecture**: DistilBERT-multilingual / mBERT
- **Task**: Multi-class Classification (Specialist Routing & Severity)
- **Weights**: Due to GitHub size limits, weights are not version-controlled. Please download them from the designated HuggingFace repository and place them in the `results/` directories.