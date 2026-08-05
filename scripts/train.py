"""Training pipeline module for MediTriageAI. Designed to be imported by run_experiment.py."""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


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
    max_rows: int | None = 3000
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
    seed_everything(1337)

    from rich.console import Console

    console = Console()
    model_meta = config.model_cls()
    tokenizer = model_meta.get_tokenizer()
    built_model = model_meta.build(None)

    if config.model_cls.needs_vocab_injection():
        model_meta.inject_vocab(built_model, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_checkpoint_dict = None
    if config.resume_checkpoint and config.resume_checkpoint.exists():
        console.print(
            f"[bold green]Resuming training from checkpoint:[/bold green] {config.resume_checkpoint}"
        )
        from src.checkpoint_manager import load_checkpoint

        resume_checkpoint_dict = load_checkpoint(
            config.resume_checkpoint, map_location="cpu"
        )
        state_dict = resume_checkpoint_dict.get(
            "model_state_dict", resume_checkpoint_dict
        )
        built_model.load_state_dict(state_dict)
    built_model.to(device)

    # Build Dataloaders
    train_loader = _build_split_loader(
        "train",
        tokenizer,
        config.dataset_path,
        config.batch_size,
        config.max_length,
        config.max_rows,
    )
    val_loader = _build_split_loader(
        "val",
        tokenizer,
        config.dataset_path,
        config.batch_size,
        config.max_length,
        config.max_rows,
    )
    test_loader = _build_split_loader(
        "test",
        tokenizer,
        config.dataset_path,
        config.batch_size,
        config.max_length,
        config.max_rows,
    )

    if train_loader is None or val_loader is None or test_loader is None:
        console.print(
            "[yellow]Dataset not found or empty splits; running scaffold dry-run (no training).[/yellow]"
        )
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
        return TrainingArtifacts(
            model=built_model,
            tokenizer=tokenizer,
            test_loader=test_loader,
            config=config,
            history={"train_loss": [], "val_loss": []},
        )

    # Instantiate production Trainer from meditriage.training.trainer
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

    trainer_obj = MeditriageTrainer(
        model=built_model,
        config=meditriage_cfg,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        device=device,
    )

    if resume_checkpoint_dict is not None:
        trainer_obj.current_epoch = resume_checkpoint_dict.get("epoch", -1) + 1
        trainer_obj.global_step = resume_checkpoint_dict.get("global_step", 0)

    console.print(
        f"[bold green]Executing production training via meditriage.training.trainer (Loss: FocalOrdinalLoss)...[/bold green]"
    )
    _train_summary = trainer_obj.train()

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

    return TrainingArtifacts(
        model=built_model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        config=config,
        history={"train_loss": [], "val_loss": []},
    )
