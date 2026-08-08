"""Training pipeline module for MediTriageAI. Designed to be imported by run_experiment.py.

Forensic instrumentation added for production observability.
All logging is INFO/DEBUG level via the standard logging module.
Zero behavioural impact on training pipeline.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

logger = logging.getLogger("meditriage.training.scripts.train")
# Ensure console output even if no handler is configured yet
if not logger.handlers:
    _sh = logging.StreamHandler(sys.stderr)
    _sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_sh)
    logger.setLevel(logging.DEBUG)


def seed_everything(seed: int = 1337):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# DirectML-specific monkeypatch was removed to prepare for clean Google Colab T4 run.
class DummyTask:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add_task(self, *args, **kwargs):
        return 1

    def advance(self, *args, **kwargs):
        pass


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.base_model import BaseMediTriageModel
from models.emergent_path_triage import apply_loss_hook
from src.checkpoint_manager import save_checkpoint
from src.dashboard import build_metrics_table, build_val_summary_table
from src.dataset import MediTriageDataset, RunningMetrics, load_split_rows
from src.model import SEVERITY_LABELS, SPECIALIST_CLASSES, JointLoss, JointLossWeights

DEFAULT_DATASET = REPO_ROOT / "meditriage" / "data" / "processed" / "dataset.parquet"


@dataclass(frozen=True)
class TrainingConfig:
    model_cls: type[BaseMediTriageModel]
    dataset_path: Path = DEFAULT_DATASET
    batch_size: int = 32
    max_length: int = 64
    max_rows: int | None = None
    epochs: int = 2
    encoder_lr: float = 2e-5
    classifier_lr: float = 1e-4
    weight_decay: float = 0.01
    train_time_seconds: float = 0.0
    early_stopping_patience: int | None = None
    resume_checkpoint: Path | None = None

    @property
    def model_display_name(self) -> str:
        return self.model_cls.display_name

    @property
    def model_short_name(self) -> str:
        return self.model_cls.short_name

    @property
    def is_novel_contribution(self) -> bool:
        return bool(self.model_cls.is_novel_contribution)


@dataclass
class TrainingArtifacts:
    model: Any
    tokenizer: Any
    test_loader: DataLoader
    config: TrainingConfig
    history: dict[str, list[float]] = None


def _build_split_loader(
    split: str,
    tokenizer: Any,
    dataset_path: Path,
    batch_size: int,
    max_length: int,
    max_rows: int | None,
) -> DataLoader | None:
    if not dataset_path.exists():
        return None
    rows = load_split_rows(dataset_path, split, max_rows=max_rows)
    if not rows:
        return None

    cuda_available = torch.cuda.is_available()
    cpu_count = os.cpu_count() or 1
    num_workers = 0 if sys.platform == "win32" else min(8, cpu_count)

    dl_kwargs = {
        "batch_size": batch_size,
        "shuffle": (split == "train"),
        "pin_memory": cuda_available,
    }
    if num_workers > 0:
        dl_kwargs["num_workers"] = num_workers
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    return DataLoader(
        MediTriageDataset(rows, tokenizer, max_length=max_length), **dl_kwargs
    )


def run_training(config: TrainingConfig) -> TrainingArtifacts:
    logger.info("[RUN-TRAINING] ══════════════════════════════════════════")
    logger.info("[RUN-TRAINING] ENTER run_training()")
    logger.info("[RUN-TRAINING] Model: %s (%s)", config.model_display_name, config.model_short_name)
    logger.info("[RUN-TRAINING] Dataset: %s", config.dataset_path)
    logger.info("[RUN-TRAINING] Epochs: %d | Batch: %d | Max rows: %s",
                 config.epochs, config.batch_size, config.max_rows)
    logger.info("[RUN-TRAINING] ══════════════════════════════════════════")

    seed_everything(1337)

    from rich.console import Console

    console = Console()

    # ── Model & Tokenizer ──
    logger.info("[RUN-TRAINING] Creating model meta & tokenizer...")
    t_model_start = time.monotonic()
    model_meta = config.model_cls()
    tokenizer = model_meta.get_tokenizer()
    logger.info("[RUN-TRAINING] Tokenizer created: %s (vocab_size=%s)",
                 type(tokenizer).__name__, getattr(tokenizer, 'vocab_size', '?'))

    built_model = model_meta.build(None)
    logger.info("[RUN-TRAINING] Model built: %s (params=%d)",
                 type(built_model).__name__,
                 sum(p.numel() for p in built_model.parameters()))

    if config.model_cls.needs_vocab_injection():
        model_meta.inject_vocab(built_model, tokenizer)
        logger.info("[RUN-TRAINING] Vocab injection applied")
    t_model = time.monotonic() - t_model_start
    logger.info("[RUN-TRAINING] Model construction complete in %.2fs", t_model)

    # ── Device & Resume ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[RUN-TRAINING] Device: %s", device)

    resume_checkpoint_dict = None
    if config.resume_checkpoint and config.resume_checkpoint.exists():
        console.print(
            f"[bold green]Resuming training from checkpoint:[/bold green] {config.resume_checkpoint}"
        )
        logger.info("[RUN-TRAINING] Loading resume checkpoint: %s", config.resume_checkpoint)
        from src.checkpoint_manager import load_checkpoint

        resume_checkpoint_dict = load_checkpoint(
            config.resume_checkpoint, map_location="cpu"
        )
        state_dict = resume_checkpoint_dict.get(
            "model_state_dict", resume_checkpoint_dict
        )
        built_model.load_state_dict(state_dict)
        logger.info("[RUN-TRAINING] Resume checkpoint loaded successfully")
    built_model.to(device)
    logger.info("[RUN-TRAINING] Model moved to %s", device)

    # ── Build DataLoaders ──
    logger.info("[RUN-TRAINING] Building dataloaders...")
    t_dl_start = time.monotonic()

    logger.info("[RUN-TRAINING] Loading train split...")
    t_split = time.monotonic()
    train_loader = _build_split_loader(
        "train",
        tokenizer,
        config.dataset_path,
        config.batch_size,
        config.max_length,
        config.max_rows,
    )
    logger.info("[RUN-TRAINING] Train split: %s (%.2fs)",
                 f"len={len(train_loader)}" if train_loader else "None",
                 time.monotonic() - t_split)

    logger.info("[RUN-TRAINING] Loading val split...")
    t_split = time.monotonic()
    val_loader = _build_split_loader(
        "val",
        tokenizer,
        config.dataset_path,
        config.batch_size,
        config.max_length,
        config.max_rows,
    )
    logger.info("[RUN-TRAINING] Val split: %s (%.2fs)",
                 f"len={len(val_loader)}" if val_loader else "None",
                 time.monotonic() - t_split)

    logger.info("[RUN-TRAINING] Loading test split...")
    t_split = time.monotonic()
    test_loader = _build_split_loader(
        "test",
        tokenizer,
        config.dataset_path,
        config.batch_size,
        config.max_length,
        config.max_rows,
    )
    logger.info("[RUN-TRAINING] Test split: %s (%.2fs)",
                 f"len={len(test_loader)}" if test_loader else "None",
                 time.monotonic() - t_split)

    t_dl = time.monotonic() - t_dl_start
    logger.info("[RUN-TRAINING] All dataloaders built in %.2fs", t_dl)

    if train_loader is None or val_loader is None or test_loader is None:
        console.print(
            "[yellow]Dataset not found or empty splits; running scaffold dry-run (no training).[/yellow]"
        )
        logger.info("[RUN-TRAINING] Missing splits — entering dry-run scaffold mode")
        demo_rows = [
            {
                "text": "Patient has severe abdominal pain and fever.",
                "label_specialist_id": 4,
                "label_severity_id": 1,
            },
            {
                "text": "Mild headache with stable vitals.",
                "label_specialist_id": 5,
                "label_severity_id": 3,
            },
        ]
        cuda_available = torch.cuda.is_available()
        cpu_count = os.cpu_count() or 1
        num_workers = 0 if sys.platform == "win32" else min(8, cpu_count)

        dl_kwargs = {
            "batch_size": config.batch_size,
            "pin_memory": cuda_available,
        }
        if num_workers > 0:
            dl_kwargs["num_workers"] = num_workers
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = 2

        test_loader = DataLoader(
            MediTriageDataset(demo_rows, tokenizer, max_length=config.max_length),
            **dl_kwargs,
        )
        logger.info("[RUN-TRAINING] EXIT run_training() — dry-run scaffold (no real training)")
        return TrainingArtifacts(
            model=built_model,
            tokenizer=tokenizer,
            test_loader=test_loader,
            config=config,
            history={"train_loss": [], "val_loss": []},
        )

    # ── Instantiate production Trainer from meditriage.training.trainer ──
    logger.info("[RUN-TRAINING] Instantiating production MeditriageTrainer...")
    from meditriage.training.config import TrainingConfig as MeditriageConfig
    from meditriage.training.trainer import Trainer as MeditriageTrainer

    res_dir = REPO_ROOT / "results" / config.model_short_name
    res_dir.mkdir(parents=True, exist_ok=True)

    meditriage_cfg = MeditriageConfig(
        experiment_name=config.model_short_name,
        output_dir=str(res_dir),
        num_epochs=config.epochs,
        batch_size=config.batch_size,
        max_length=config.max_length,
        learning_rate=config.encoder_lr,
        weight_decay=config.weight_decay,
        loss_type="focal_ordinal",
        early_stopping_patience=config.early_stopping_patience or 3,
    )
    logger.info("[RUN-TRAINING] MeditriageConfig created: experiment=%s, output=%s",
                 meditriage_cfg.experiment_name, meditriage_cfg.output_dir)

    t_trainer_init = time.monotonic()
    trainer_obj = MeditriageTrainer(
        model=built_model,
        config=meditriage_cfg,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        device=device,
    )
    logger.info("[RUN-TRAINING] MeditriageTrainer instantiated in %.2fs",
                 time.monotonic() - t_trainer_init)

    if resume_checkpoint_dict is not None:
        trainer_obj.current_epoch = resume_checkpoint_dict.get("epoch", -1) + 1
        trainer_obj.global_step = resume_checkpoint_dict.get("global_step", 0)
        logger.info("[RUN-TRAINING] Trainer state restored: epoch=%d, step=%d",
                     trainer_obj.current_epoch, trainer_obj.global_step)

    console.print(
        f"[bold green]Executing production training via meditriage.training.trainer (Loss: FocalOrdinalLoss)...[/bold green]"
    )
    logger.info("[RUN-TRAINING] ── Calling trainer_obj.train() ──")
    try:
        _train_summary = trainer_obj.train()
    except Exception:
        logger.error("[RUN-TRAINING] EXCEPTION during trainer_obj.train()!\n%s",
                      traceback.format_exc())
        raise
    logger.info("[RUN-TRAINING] ── trainer_obj.train() returned: %s ──", _train_summary)

    logger.info("[RUN-TRAINING] Saving final checkpoint...")
    t_save = time.monotonic()
    save_checkpoint(
        path=res_dir / "checkpoint.pt",
        model_short_name=config.model_short_name,
        backbone_name=getattr(config.model_cls, "model_name", "xlm-roberta-base"),
        config=meditriage_cfg,
        state_dict={k: v.cpu().clone() for k, v in built_model.state_dict().items()},
        extra_states={
            "epoch": config.epochs - 1,
            "global_step": trainer_obj.global_step,
            "best_val_metric": trainer_obj.best_metric,
        },
    )
    logger.info("[RUN-TRAINING] Final checkpoint saved in %.2fs", time.monotonic() - t_save)

    logger.info("[RUN-TRAINING] EXIT run_training() — training complete")
    return TrainingArtifacts(
        model=built_model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        config=config,
        history={"train_loss": [], "val_loss": []},
    )
