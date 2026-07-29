"""Training pipeline module for MediTriageAI. Designed to be imported by run_experiment.py."""

from __future__ import annotations

import sys
import time
import contextlib
import os
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

def seed_everything(seed: int = 1337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# DirectML-specific monkeypatch was removed to prepare for clean Google Colab T4 run.
class DummyTask:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def add_task(self, *args, **kwargs): return 1
    def advance(self, *args, **kwargs): pass


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.base_model import BaseMediTriageModel
from models.emergent_path_triage import apply_loss_hook
from src.dataset import MediTriageDataset, load_split_rows, RunningMetrics
from src.model import JointLoss, JointLossWeights, MediTriageTransformer, SPECIALIST_CLASSES, SEVERITY_LABELS
from src.dashboard import make_epoch_progress, build_metrics_table, build_val_summary_table
from src.checkpoint_manager import save_checkpoint

DEFAULT_DATASET = REPO_ROOT / "meditriage" / "data" / "processed" / "dataset.parquet"


@dataclass(frozen=True)
class TrainingConfig:
    model_cls: Type[BaseMediTriageModel]
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


def _build_split_loader(split: str, tokenizer: Any, dataset_path: Path, batch_size: int, max_length: int, max_rows: int | None) -> DataLoader | None:
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
        
    return DataLoader(MediTriageDataset(rows, tokenizer, max_length=max_length), **dl_kwargs)


def run_training(config: TrainingConfig) -> TrainingArtifacts:
    seed_everything(1337)
    
    from rich.console import Console
    console = Console()
    model_meta = config.model_cls()
    tokenizer = model_meta.get_tokenizer()
    built_model = model_meta.build(None)
    
    if config.model_cls.needs_vocab_injection():
        model_meta.inject_vocab(built_model, tokenizer)

    # Build Dataloaders
    train_loader = _build_split_loader("train", tokenizer, config.dataset_path, config.batch_size, config.max_length, config.max_rows)
    val_loader = _build_split_loader("val", tokenizer, config.dataset_path, config.batch_size, config.max_length, config.max_rows)
    test_loader = _build_split_loader("test", tokenizer, config.dataset_path, config.batch_size, config.max_length, config.max_rows)

    if train_loader is None or val_loader is None or test_loader is None:
        console.print("[yellow]Dataset not found or empty splits; running scaffold dry-run (no training).[/yellow]")
        demo_rows = [
            {"text": "Patient has severe abdominal pain and fever.", "label_specialist_id": 4, "label_severity_id": 1},
            {"text": "Mild headache with stable vitals.", "label_specialist_id": 5, "label_severity_id": 3},
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
            
        test_loader = DataLoader(MediTriageDataset(demo_rows, tokenizer, max_length=config.max_length), **dl_kwargs)
        return TrainingArtifacts(model=built_model, tokenizer=tokenizer, test_loader=test_loader, config=config, history={"train_loss": [], "val_loss": []})

    # Optimization Setup
    # Differentiate parameters between encoder and heads/architecture layers
    encoder_params = []
    head_params = []
    # E-PATH-CO-REASON architecture layer prefixes (randomly initialized, need higher LR)
    _ARCH_PREFIXES = ("classifier_", "dces.", "router.", "blocks.", "engine.", "step_engine.", "dcp.", "loss_balancer.", "specialist_calibrator.", "severity_calibrator.", "trace_recorder.")
    for name, param in built_model.named_parameters():
        if any(name.startswith(prefix) for prefix in _ARCH_PREFIXES):
            head_params.append(param)
        else:
            encoder_params.append(param)
            param.requires_grad = False # Start frozen

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": config.encoder_lr, "weight_decay": config.weight_decay},
        {"params": head_params, "lr": config.classifier_lr, "weight_decay": config.weight_decay}
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    built_model.to(device)

    # Dynamic Class Weight Calculation (computed exclusively from train split)
    spec_counts = [0] * len(SPECIALIST_CLASSES)
    for row in train_loader.dataset.rows:
        spec_counts[row["label_specialist_id"]] += 1
    total_spec = len(train_loader.dataset.rows)
    num_spec_classes = len(SPECIALIST_CLASSES)
    spec_weights = [total_spec / (num_spec_classes * count) if count > 0 else 1.0 for count in spec_counts]
    spec_weights_tensor = torch.tensor(spec_weights, dtype=torch.float).to(device)

    sev_counts = [0] * len(SEVERITY_LABELS)
    for row in train_loader.dataset.rows:
        sev_counts[row["label_severity_id"]] += 1
    total_sev = len(train_loader.dataset.rows)
    num_sev_classes = len(SEVERITY_LABELS)
    sev_weights = [(total_sev / (num_sev_classes * count)) ** 0.5 if count > 0 else 1.0 for count in sev_counts]
    console.print(f"[bold blue]Computed Severity Weights (Square-Root Inverse Frequency): {dict(zip(SEVERITY_LABELS, sev_weights))}[/bold blue]")
    sev_weights_tensor = torch.tensor(sev_weights, dtype=torch.float).to(device)

    loss_fn = JointLoss(
        JointLossWeights(alpha_specialist=1.0, beta_severity=1.2),
        specialist_class_weights=spec_weights_tensor,
        severity_class_weights=sev_weights_tensor
    )

    # Setup Scheduler
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    history = {"train_loss": [], "val_loss": [], "train_spec_acc": [], "train_sev_acc": [], "val_spec_acc": [], "val_sev_acc": []}
    
    # Early Stopping state
    best_val_metric = -1.0
    early_stopping_counter = 0
    best_model_state = None
    metric_name = "Joint Val Macro-F1"
    
    start_time = time.time()
    TIME_LIMIT_SECONDS = 90 * 60
    timeout_reached = False
    
    first_step = True

    for epoch in range(config.epochs):
        if timeout_reached:
            break
            
        if epoch == 1:
            console.print("[yellow]Unfreezing encoder for epoch 2...[/yellow]")
            for param in encoder_params:
                param.requires_grad = True

        built_model.train()
        running_metrics = RunningMetrics()
        
        progress = DummyTask()
        task_id = progress.add_task(f"Epoch {epoch + 1}/{config.epochs}", total=len(train_loader))
        
        with progress:
            for batch in train_loader:
                if time.time() - start_time > TIME_LIMIT_SECONDS:
                    console.print("[red]HARD TIME LIMIT (90m) REACHED! Stopping training early.[/red]")
                    timeout_reached = True
                    break

                if first_step:
                    console.print(f"[green]Step 0 Learning Rates: Encoder={optimizer.param_groups[0]['lr']:.2e}, Classifier={optimizer.param_groups[1]['lr']:.2e}[/green]")
                    first_step = False

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_spec = batch["labels_specialist"].to(device)
                labels_sev = batch["labels_severity"].to(device)
                
                optimizer.zero_grad()
                spec_logits, sev_logits = built_model(input_ids, attention_mask)
                loss_dict = apply_loss_hook(built_model, spec_logits, sev_logits, labels_spec, labels_sev, loss_fn)
                
                loss = loss_dict["joint_loss"]
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                spec_preds = spec_logits.argmax(dim=-1).tolist()
                sev_preds = sev_logits.argmax(dim=-1).tolist()
                
                running_metrics.update(
                    loss.item(),
                    loss_dict["specialist_loss"].item(),
                    loss_dict["severity_loss"].item(),
                    spec_preds,
                    labels_spec.tolist(),
                    sev_preds,
                    labels_sev.tolist()
                )
                progress.advance(task_id)
        
        epoch_metrics = running_metrics.compute()
        history["train_loss"].append(epoch_metrics["loss"])
        history["train_spec_acc"].append(epoch_metrics["specialist_acc"])
        history["train_sev_acc"].append(epoch_metrics["severity_acc"])
        
        console.print(build_metrics_table(epoch_metrics, epoch, optimizer.param_groups[0]["lr"]))
        
        # Validation Epoch
        built_model.eval()
        val_metrics = RunningMetrics()
        val_spec_preds = []
        val_spec_labels = []
        val_sev_preds = []
        val_sev_labels = []
        with torch.no_grad():
            for batch in val_loader:
                if time.time() - start_time > TIME_LIMIT_SECONDS:
                    timeout_reached = True
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_spec = batch["labels_specialist"].to(device)
                labels_sev = batch["labels_severity"].to(device)
                
                spec_logits, sev_logits = built_model(input_ids, attention_mask)
                loss_dict = apply_loss_hook(built_model, spec_logits, sev_logits, labels_spec, labels_sev, loss_fn)
                
                spec_preds = spec_logits.argmax(dim=-1).tolist()
                sev_preds = sev_logits.argmax(dim=-1).tolist()
                
                val_metrics.update(
                    loss_dict["joint_loss"].item(),
                    loss_dict["specialist_loss"].item(),
                    loss_dict["severity_loss"].item(),
                    spec_preds,
                    labels_spec.tolist(),
                    sev_preds,
                    labels_sev.tolist()
                )
                val_spec_preds.extend(spec_preds)
                val_spec_labels.extend(labels_spec.tolist())
                val_sev_preds.extend(sev_preds)
                val_sev_labels.extend(labels_sev.tolist())
        
        val_epoch_metrics = val_metrics.compute()
        history["val_loss"].append(val_epoch_metrics["loss"])
        history["val_spec_acc"].append(val_epoch_metrics["specialist_acc"])
        history["val_sev_acc"].append(val_epoch_metrics["severity_acc"])
        
        console.print(build_val_summary_table(epoch, val_epoch_metrics, time.time() - start_time))

        # Check early stopping patience
        if config.early_stopping_patience is not None:
            try:
                from src.metrics import compute_macro_f1
                val_spec_macro_f1 = compute_macro_f1(val_spec_labels, val_spec_preds, "specialist")
                val_sev_macro_f1 = compute_macro_f1(val_sev_labels, val_sev_preds, "severity")
                val_metric_value = (val_spec_macro_f1 + val_sev_macro_f1) / 2.0
                metric_name = "Joint Val Macro-F1"
                
                if best_val_metric == -1.0 or best_val_metric == float("inf"):
                    best_val_metric = -1.0
                
                is_better = (epoch == 0) or (val_metric_value > best_val_metric)
            except Exception:
                val_metric_value = val_epoch_metrics["loss"]
                metric_name = "Val Loss"
                
                if best_val_metric == -1.0:
                    best_val_metric = float("inf")
                
                is_better = (epoch == 0) or (val_metric_value < best_val_metric)

            if is_better:
                best_val_metric = val_metric_value
                early_stopping_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in built_model.state_dict().items()}
                
                # Save the BEST checkpoint immediately
                results_subdir = REPO_ROOT / "results" / model_meta.short_name
                results_subdir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = results_subdir / "checkpoint.pt"
                
                # Serialize config safely
                config_dict = config.__dict__.copy()
                serialized_config = {k: str(v) if k == "model_cls" else v for k, v in config_dict.items()}
                
                save_checkpoint(
                    path=checkpoint_path,
                    model_short_name=model_meta.short_name,
                    backbone_name=getattr(model_meta, "model_name", "xlm-roberta-base"),
                    config=serialized_config,
                    state_dict=best_model_state
                )
                console.print(f"[green]Saved best model checkpoint to: {checkpoint_path} (Best {metric_name}: {best_val_metric:.4f})[/green]")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config.early_stopping_patience:
                    console.print(f"[yellow]Early stopping triggered at epoch {epoch + 1} (Patience of {config.early_stopping_patience} reached)[/yellow]")
                    break

    elapsed_time = time.time() - start_time
    # Update config with train time
    config_dict = config.__dict__.copy()
    config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}
    config_dict["train_time_seconds"] = elapsed_time
    updated_config = TrainingConfig(**config_dict)

    # If early stopping was active, restore the best weights for final return/evaluation
    if config.early_stopping_patience is not None and best_model_state is not None:
        console.print("[green]Restoring best checkpoint weights for final evaluation...[/green]")
        built_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    elif best_model_state is None:
        # Fallback if early stopping was disabled or no epochs trained: save current state
        results_subdir = REPO_ROOT / "results" / model_meta.short_name
        results_subdir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = results_subdir / "checkpoint.pt"
        cpu_state_dict = {k: v.cpu() for k, v in built_model.state_dict().items()}
        
        # Serialize config safely
        config_dict = updated_config.__dict__.copy()
        serialized_config = {k: str(v) if k == "model_cls" else v for k, v in config_dict.items()}
        
        save_checkpoint(
            path=checkpoint_path,
            model_short_name=model_meta.short_name,
            backbone_name=getattr(model_meta, "model_name", "xlm-roberta-base"),
            config=serialized_config,
            state_dict=cpu_state_dict
        )
        console.print(f"[green]Saved final model checkpoint to: {checkpoint_path}[/green]")

    return TrainingArtifacts(model=built_model, tokenizer=tokenizer, test_loader=test_loader, config=updated_config, history=history)

