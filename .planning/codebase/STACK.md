# MediTriageAI — Technology Stack (STACK.md)

**Generated:** 2026-08-14  
**Repository State:** Frozen Baseline (v1.0.0)

---

## 1. Core Languages & Execution Runtimes

| Technology | Version / Requirement | Purpose |
|:---|:---|:---|
| **Python** | `>=3.10` (Tested on 3.10, 3.11) | Core application, data pipeline, and model framework |
| **CUDA** | `11.8` / `12.1` | GPU acceleration for PyTorch on NVIDIA GPUs / DGX clusters |
| **JavaScript / HTML / CSS** | Vanilla ES6+ | Lightweight interactive visual dashboard (`dashboard_web/`) |

---

## 2. Deep Learning & Transformer Ecosystem

| Package | Version Range | Key Modules & Usage |
|:---|:---|:---|
| **`torch`** | `>=2.0.0` | Tensor computations, `nn.Module`, AMP (`autocast`, `GradScaler`), `torch.compile`, `torch.distributed` (NCCL) |
| **`transformers`** | `>=4.36.0, <4.45.0` | Pretrained Transformer backbones (`XLM-RoBERTa`, `mBERT`, `DistilBERT`, `IndicBERT`), tokenizers, cosine LR schedulers |
| **`huggingface_hub`** | `>=0.20.0, <0.37.0` | Model and dataset downloading from Hugging Face Hub |
| **`datasets`** | `>=2.14.0, <3.0.0` | Loading external clinical datasets and tabular partitions |
| **`tokenizers`** | `>=0.15.0, <0.23.0` | Fast subword tokenization and custom vocab injections |

---

## 3. Data Processing, Storage & Serialization

| Package | Version Range | Key Usage |
|:---|:---|:---|
| **`pandas`** | `>=2.0.0` | Primary tabular manipulation for clinical transcripts, annotations, and metadata |
| **`numpy`** | `>=1.24.0` | Numerical array operations, bootstrapping, seeding matrices |
| **`pyarrow`** | `>=12.0.0` | Columnar disk caching (`predictions.parquet`, `cache/predictions/`) |
| **`scikit-learn`** | `>=1.3.0` | Metric computations (Macro-F1, AUROC, ECE, Cohen's Kappa), stratified splits |
| **`ijson`** | `>=3.2.0` | Streaming JSON parsing for large clinical corpora without memory exhaustion |
| **`fsspec`** | `>=2023.6.0, <=2025.2.0` | File system abstraction for local and remote dataset access |
| **`pyyaml`** | `>=6.0` | Configuration parsing (`config/*.yaml`, `configs/*.yaml`) |

---

## 4. Distributed Training & Acceleration

- **`torch.distributed` (DDP)**: Multi-GPU NCCL backend with Rank-0 safe I/O shielding (`scripts/train_ddp.py`).
- **`torch.amp` (Mixed Precision)**: Automated Mixed Precision with `GradScaler` and `autocast(device_type="cuda")`.
- **`torch.compile`**: Inductor graph compilation enabled conditionally for high-throughput GPU nodes.
- **Gradient Checkpointing**: Configurable per-backbone to prevent CUDA OOM on long sequence lengths (up to 512 tokens).

---

## 5. Web Serving & Interface

| Component | Framework / Tool | Description |
|:---|:---|:---|
| **Inference API** | **`FastAPI` + `Uvicorn`** | REST inference service (`scripts/serve_api.py`) with HTTP Basic Auth and red-flag rule fallback |
| **Dashboard** | **Vanilla HTML5 / Modern CSS** | Browser dashboard (`dashboard_web/index.html`) driven by exported JSON/Parquet |
| **CLI Styling** | **`rich`** | Terminal formatting, interactive tables, and panels for experiment execution |

---

## 6. Testing, Quality & Tooling

| Tool | Version Range | Configuration / Purpose |
|:---|:---|:---|
| **`pytest`** | `>=7.4.0` | Test runner (`pytest.ini` with `pythonpath = .`, custom `--run-slow` marker) |
| **`tabulate`** | `>=0.9.0` | Terminal and report table formatting |
| **`matplotlib` / `seaborn`** | `>=3.7.0` / `>=0.12.0` | 300-DPI publication figures, calibration curves, confusion matrices |
| **`black` / `ruff`** | Standard | Code formatting and linting targets |
