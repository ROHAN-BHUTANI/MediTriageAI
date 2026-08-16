# MediTriageAI — Coding Standards & Engineering Conventions (CONVENTIONS.md)

**Generated:** 2026-08-14  
**Repository State:** Frozen Baseline (v1.0.0)

---

## 1. Python Style & Typing Conventions

- **Python Version**: Minimum Python 3.10 required.
- **Future Annotations**: All core Python files begin with `from __future__ import annotations` to support modern union types (e.g. `str | None`, `tuple[Tensor, Tensor]`).
- **Strict Typing**: Public interfaces, methods, and functions must define explicit type hints for inputs and outputs.
- **Data Models**: Configuration structures and domain entities are modeled using `@dataclass` or `@dataclass(frozen=True)` (e.g., `TrainingConfig`, `ExperimentModel`).

---

## 2. Determinism & Seeding Protocols

To guarantee scientific reproducibility across single-GPU, Colab, and multi-GPU DDP clusters:

1. **Global Seeding**: A fixed seed (default `1337`) is injected into `random`, `numpy`, `torch.manual_seed()`, and `os.environ["PYTHONHASHSEED"]`.
2. **Deterministic CuDNN**: `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` are enforced.
3. **DDP Rank Offsets**: Multi-GPU DDP ranks use `seed = global_seed + rank` to avoid identical mini-batch shuffling while maintaining cross-run determinism.
4. **Dataset Fingerprinting**: `dataset_manifest.json` contains a SHA256 checksum of the training data. Checkpoint resumption halts if the dataset fingerprint does not match.

---

## 3. Distributed Training & Safe File I/O

- **Rank-0 Shielding**: All disk write operations (saving checkpoints, writing prediction Parquet files, generating HTML/Markdown reports, logging metrics) are restricted strictly to Rank 0 (`if rank == 0:` or `is_main_process()`).
- **Atomic File Writing**: Checkpoints and reports are written to temporary files and atomically renamed or verified before finalizing to prevent corrupted state from job interruptions.
- **Device Management**: Models and tensors explicitly respect the assigned `device` or local rank GPU without hardcoding `"cuda:0"`.

---

## 4. Error Handling & Custom Exceptions

The codebase establishes a unified exception hierarchy rooted in `MediTriageError`:

```
MediTriageError
├── ConfigurationError      # Missing/malformed YAML or hyperparameters
├── CompatibilityError      # Checkpoint schema mismatch or latent dimension drift
├── RoutingError            # Unrecognized specialist or severity label
└── InterfaceError          # Abstract method contract violation
```

Silent failures and bare `except:` clauses are prohibited. All caught exceptions log informative error traces via standard logging before raising or gracefully degrading.

---

## 5. Logging & Observability

- Standard `logging` module is utilized with scoped namespaces:
  - `meditriage.training`
  - `meditriage.builder`
  - `models.emergent_path_triage`
  - `analysis`
- Console output is formatted with timestamps and log levels: `[%(asctime)s] %(levelname)s %(name)s: %(message)s`.
- StreamHandlers default to `sys.stderr` to prevent buffering delays in containerized and headless environments.
