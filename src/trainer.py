"""Google Colab Research Training Framework for E-PATH-CO-REASON."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.emergent_path_triage.exceptions import (
    CompatibilityError,
    ConfigurationError,
)
from src.checkpoint_manager import load_checkpoint as mgr_load_checkpoint
from src.checkpoint_manager import save_checkpoint as mgr_save_checkpoint
from src.config_manager import TrainingConfig
from src.data_pipeline import detect_colab_environment, set_global_seeds


def get_git_commit() -> str:
    """Safely extract git commit hash if available in path."""
    try:
        import subprocess

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit
    except Exception:
        return "N/A"


def compute_ece(
    confidences: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) for a set of predictions and confidences."""
    confidences = confidences.cpu()
    predictions = predictions.cpu()
    labels = labels.cpu()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in the current bin
        in_bin = (
            (confidences >= bin_lower) & (confidences < bin_upper)
            if i < num_bins - 1
            else (confidences >= bin_lower) & (confidences <= bin_upper)
        )
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = (
                (predictions[in_bin] == labels[in_bin]).float().mean().item()
            )
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += prop_in_bin * abs(accuracy_in_bin - avg_confidence_in_bin)

    return ece


def compute_brier_score(
    probabilities: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> float:
    """Computes multiclass Brier score."""
    probabilities = probabilities.cpu()
    labels = labels.cpu()

    # One-hot encode labels
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    # Compute mean squared error
    brier_score = torch.mean(torch.sum((probabilities - one_hot) ** 2, dim=-1)).item()
    return brier_score


class MetricTracker:
    """Accumulates and handles epoch-wise training and validation metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset loss and accuracy accumulators."""
        self.loss_sum = 0.0
        self.specialist_loss_sum = 0.0
        self.severity_loss_sum = 0.0
        self.ortho_loss_sum = 0.0
        self.cons_loss_sum = 0.0
        self.div_loss_sum = 0.0
        self.specialist_correct = 0
        self.severity_correct = 0
        self.total_samples = 0

    def update(
        self,
        loss_dict: dict[str, torch.Tensor],
        spec_preds: torch.Tensor,
        spec_labels: torch.Tensor,
        sev_preds: torch.Tensor,
        sev_labels: torch.Tensor,
    ) -> None:
        """Update metrics for a single batch iteration."""
        batch_size = spec_labels.size(0)
        self.total_samples += batch_size

        self.loss_sum += (
            loss_dict.get("joint_loss", loss_dict.get("loss", torch.zeros(()))).item()
            * batch_size
        )
        self.specialist_loss_sum += (
            loss_dict.get("specialist_loss", torch.zeros(())).item() * batch_size
        )
        self.severity_loss_sum += (
            loss_dict.get("severity_loss", torch.zeros(())).item() * batch_size
        )
        self.ortho_loss_sum += (
            loss_dict.get("ortho_loss", torch.zeros(())).item() * batch_size
        )
        self.cons_loss_sum += (
            loss_dict.get("cons_loss", torch.zeros(())).item() * batch_size
        )
        self.div_loss_sum += (
            loss_dict.get("div_loss", torch.zeros(())).item() * batch_size
        )

        self.specialist_correct += (spec_preds == spec_labels).sum().item()
        self.severity_correct += (sev_preds == sev_labels).sum().item()

    def get_summary(self) -> dict[str, float]:
        """Compute average losses and accuracies."""
        if self.total_samples == 0:
            return {
                "loss": 0.0,
                "specialist_loss": 0.0,
                "severity_loss": 0.0,
                "ortho_loss": 0.0,
                "cons_loss": 0.0,
                "div_loss": 0.0,
                "specialist_acc": 0.0,
                "severity_acc": 0.0,
            }
        return {
            "loss": self.loss_sum / self.total_samples,
            "specialist_loss": self.specialist_loss_sum / self.total_samples,
            "severity_loss": self.severity_loss_sum / self.total_samples,
            "ortho_loss": self.ortho_loss_sum / self.total_samples,
            "cons_loss": self.cons_loss_sum / self.total_samples,
            "div_loss": self.div_loss_sum / self.total_samples,
            "specialist_acc": self.specialist_correct / self.total_samples,
            "severity_acc": self.severity_correct / self.total_samples,
        }


class EmergentTrainer:
    """Modular research training framework optimized for Google Colab GPUs."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader | None = None,
        tokenizer: Any = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer

        # Colab Detection & AMP support
        self.env_meta = detect_colab_environment()

        import os

        import torch.distributed as dist

        self.is_distributed = dist.is_initialized()
        self.local_rank = (
            int(os.environ.get("LOCAL_RANK", 0)) if self.is_distributed else 0
        )

        self.device = torch.device(
            f"cuda:{self.local_rank}" if self.env_meta["has_gpu"] else "cpu"
        )
        self.model.to(self.device)

        if getattr(self.config, "use_torch_compile", False) and hasattr(
            torch, "compile"
        ):
            self.model = torch.compile(self.model)

        if self.is_distributed:
            from torch import nn

            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=True,
            )

        self.use_amp = (
            self.config.mixed_precision and self.env_meta["mixed_precision_available"]
        )
        self.scaler = GradScaler(
            device="cuda" if self.device.type == "cuda" else "cpu", enabled=self.use_amp
        )

        # Setup Optimizer and Scheduler
        self._init_optimizer()
        self._init_scheduler()

        # Metrics lists
        self.history: list[dict[str, Any]] = []
        self.best_metrics: dict[str, Any] = {}
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 1

        # Destination Paths
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def is_rank_0(self) -> bool:
        """Returns True if the current process is rank 0 or if not distributed."""
        if getattr(self, "is_distributed", False):
            import torch.distributed as dist

            return dist.get_rank() == 0
        return True

    def _init_optimizer(self) -> None:
        """Partition parameter sets to apply custom encoder vs head learning rates."""
        encoder_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": self.config.encoder_lr}
            )
        if head_params:
            param_groups.append(
                {"params": head_params, "lr": self.config.learning_rate}
            )

        opt_type = self.config.optimizer.lower()
        if opt_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                param_groups, weight_decay=self.config.weight_decay
            )
        elif opt_type == "adam":
            self.optimizer = torch.optim.Adam(param_groups)
        elif opt_type == "sgd":
            self.optimizer = torch.optim.SGD(param_groups, momentum=0.9)
        else:
            raise ConfigurationError(
                f"Unsupported optimizer_type: '{self.config.optimizer}'"
            )

    def _init_scheduler(self) -> None:
        """Set up learning rate schedulers."""
        from transformers import (
            get_cosine_schedule_with_warmup,
            get_linear_schedule_with_warmup,
        )

        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(self.config.warmup_ratio * total_steps)

        sched_type = self.config.scheduler.lower()
        if sched_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif sched_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif sched_type == "none":
            self.scheduler = None
        else:
            raise ConfigurationError(
                f"Unsupported scheduler_type: '{self.config.scheduler}'"
            )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute a single epoch training loop."""
        self.model.train()
        tracker = MetricTracker()
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels_spec = batch["labels_specialist"].to(self.device)
            labels_sev = batch["labels_severity"].to(self.device)

            # Auto-cast Mixed Precision
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                spec_logits, sev_logits = self.model(input_ids, attention_mask)

                # Check for apply_loss_hook
                from models.emergent_path_triage.hooks import apply_loss_hook
                from src.model import JointLoss

                loss_fn = JointLoss()
                loss_dict = apply_loss_hook(
                    self.model,
                    spec_logits,
                    sev_logits,
                    labels_spec,
                    labels_sev,
                    loss_fn,
                )
                loss = loss_dict["joint_loss"] / self.config.gradient_accumulation

                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(
                        f"Numerical Stability Error! Loss is NaN or Inf at epoch {epoch}, batch {step}. LR: {self.optimizer.param_groups[-1]['lr']}"
                    )

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.gradient_accumulation == 0:
                self.scaler.unscale_(self.optimizer)

                # Check for NaN/Inf in gradients
                has_nan_inf = False
                for p in self.model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            has_nan_inf = True
                            break
                if has_nan_inf:
                    raise RuntimeError(
                        f"Numerical Stability Error! NaN or Inf in gradients at epoch {epoch}, batch {step}. LR: {self.optimizer.param_groups[-1]['lr']}"
                    )

                # Gradient Clipping
                if self.config.gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clipping
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            # Record metrics
            spec_preds = spec_logits.argmax(dim=-1)
            sev_preds = sev_logits.argmax(dim=-1)
            tracker.update(loss_dict, spec_preds, labels_spec, sev_preds, labels_sev)

        metrics = tracker.get_summary()
        metrics["lr"] = self.optimizer.param_groups[-1]["lr"]
        return metrics

    def validate(self) -> dict[str, float]:
        """Perform evaluation pass over validation split."""
        self.model.eval()
        tracker = MetricTracker()

        all_spec_confidences = []
        all_sev_confidences = []
        all_spec_probs = []
        all_sev_probs = []
        all_spec_preds = []
        all_sev_preds = []
        all_spec_labels = []
        all_sev_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_spec = batch["labels_specialist"].to(self.device)
                labels_sev = batch["labels_severity"].to(self.device)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    spec_logits, sev_logits = self.model(input_ids, attention_mask)

                    from models.emergent_path_triage.hooks import apply_loss_hook
                    from src.model import JointLoss

                    loss_fn = JointLoss()
                    loss_dict = apply_loss_hook(
                        self.model,
                        spec_logits,
                        sev_logits,
                        labels_spec,
                        labels_sev,
                        loss_fn,
                    )

                spec_preds = spec_logits.argmax(dim=-1)
                sev_preds = sev_logits.argmax(dim=-1)
                tracker.update(
                    loss_dict, spec_preds, labels_spec, sev_preds, labels_sev
                )

                # Extract and store confidence scoring outputs for calibration audit
                if (
                    hasattr(spec_logits, "specialist_confidence")
                    and spec_logits.specialist_confidence is not None
                ):
                    spec_conf = spec_logits.specialist_confidence.confidence_score
                    spec_probs = (
                        spec_logits.specialist_confidence.calibrated_probabilities
                    )
                    all_spec_confidences.append(spec_conf.cpu())
                    all_spec_probs.append(spec_probs.cpu())

                if (
                    hasattr(sev_logits, "severity_confidence")
                    and sev_logits.severity_confidence is not None
                ):
                    sev_conf = sev_logits.severity_confidence.confidence_score
                    sev_probs = sev_logits.severity_confidence.calibrated_probabilities
                    all_sev_confidences.append(sev_conf.cpu())
                    all_sev_probs.append(sev_probs.cpu())

                all_spec_preds.append(spec_preds.cpu())
                all_sev_preds.append(sev_preds.cpu())
                all_spec_labels.append(labels_spec.cpu())
                all_sev_labels.append(labels_sev.cpu())

        summary = tracker.get_summary()

        if getattr(self, "is_distributed", False):
            import torch.distributed as dist

            # Average the scalar metrics across ranks
            for k, v in summary.items():
                v_tensor = torch.tensor(v, dtype=torch.float32, device=self.device)
                dist.all_reduce(v_tensor, op=dist.ReduceOp.SUM)
                summary[k] = (v_tensor / dist.get_world_size()).item()

            def gather_t(t_list):
                if not t_list:
                    return None
                local_t = torch.cat(t_list, dim=0).cpu()
                obj = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(obj, local_t)
                return torch.cat(obj, dim=0)

            if all_spec_confidences and all_spec_probs:
                spec_conf_tensor = gather_t(all_spec_confidences)
                sev_conf_tensor = gather_t(all_sev_confidences)
                spec_prob_tensor = gather_t(all_spec_probs)
                sev_prob_tensor = gather_t(all_sev_probs)
                spec_pred_tensor = gather_t(all_spec_preds)
                sev_pred_tensor = gather_t(all_sev_preds)
                spec_label_tensor = gather_t(all_spec_labels)
                sev_label_tensor = gather_t(all_sev_labels)
            else:
                spec_conf_tensor = None
        else:
            if all_spec_confidences and all_spec_probs:
                spec_conf_tensor = torch.cat(all_spec_confidences, dim=0)
                sev_conf_tensor = torch.cat(all_sev_confidences, dim=0)
                spec_prob_tensor = torch.cat(all_spec_probs, dim=0)
                sev_prob_tensor = torch.cat(all_sev_probs, dim=0)
                spec_pred_tensor = torch.cat(all_spec_preds, dim=0)
                sev_pred_tensor = torch.cat(all_sev_preds, dim=0)
                spec_label_tensor = torch.cat(all_spec_labels, dim=0)
                sev_label_tensor = torch.cat(all_sev_labels, dim=0)
            else:
                spec_conf_tensor = None

        # Calculate ECE and Brier scores if calibration data was collected
        if self.is_rank_0() and spec_conf_tensor is not None:
            spec_conf_tensor = torch.cat(all_spec_confidences, dim=0)
            sev_conf_tensor = torch.cat(all_sev_confidences, dim=0)
            spec_prob_tensor = torch.cat(all_spec_probs, dim=0)
            sev_prob_tensor = torch.cat(
                all_spec_probs, dim=0
            )  # Note: Typo here. Wait, let's make sure we write sev_prob_tensor = torch.cat(all_sev_probs, dim=0)
            spec_pred_tensor = torch.cat(all_spec_preds, dim=0)
            sev_pred_tensor = torch.cat(all_sev_preds, dim=0)
            spec_label_tensor = torch.cat(all_spec_labels, dim=0)
            sev_label_tensor = torch.cat(all_sev_labels, dim=0)

            # Correcting typo: make sure we use all_sev_probs for sev_prob_tensor
            sev_prob_tensor = torch.cat(all_sev_probs, dim=0)

            summary["specialist_ece"] = compute_ece(
                spec_conf_tensor, spec_pred_tensor, spec_label_tensor
            )
            summary["severity_ece"] = compute_ece(
                sev_conf_tensor, sev_pred_tensor, sev_label_tensor
            )
            summary["specialist_brier"] = compute_brier_score(
                spec_prob_tensor, spec_label_tensor, 13
            )
            summary["severity_brier"] = compute_brier_score(
                sev_prob_tensor, sev_label_tensor, 5
            )

        return summary

    def fit(self) -> dict[str, Any]:
        """Run the complete multi-epoch train and evaluation lifecycle."""
        set_global_seeds(self.config.seed)

        if self.start_epoch > 1:
            print("==================================================")
            print("RESUMING TRAINING")
            print(f"Checkpoint : {getattr(self, 'resumed_from', 'N/A')}")
            print(f"Checkpoint Epoch : {self.start_epoch - 1}")
            print(f"Next Epoch : {self.start_epoch}")
            print(
                f"Remaining Epochs : {max(0, self.config.epochs - (self.start_epoch - 1))}"
            )
            print(f"AMP : {self.use_amp}")
            print(f"History Entries : {len(self.history)}")
            print("==================================================")
        else:
            print("==================================================")
            print("STARTING FRESH TRAINING")
            print(f"Output Directory : {self.checkpoint_dir}")
            print(f"Target Epochs : {self.config.epochs}")
            print("==================================================")

        print(f"Beginning E-PATH-CO-REASON training framework on {self.device}.")
        print(
            f"AMP (Mixed Precision): {self.use_amp}, Gradient Accumulation Steps: {self.config.gradient_accumulation}"
        )

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            if hasattr(self, "train_sampler") and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            t0 = time.time()
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            epoch_time = time.time() - t0

            # Store metrics
            epoch_data = {
                "epoch": epoch,
                "time": epoch_time,
                "train_loss": train_metrics["loss"],
                "train_specialist_loss": train_metrics["specialist_loss"],
                "train_severity_loss": train_metrics["severity_loss"],
                "train_ortho_loss": train_metrics["ortho_loss"],
                "train_cons_loss": train_metrics["cons_loss"],
                "train_div_loss": train_metrics["div_loss"],
                "train_specialist_acc": train_metrics["specialist_acc"],
                "train_severity_acc": train_metrics["severity_acc"],
                "val_loss": val_metrics["loss"],
                "val_specialist_loss": val_metrics["specialist_loss"],
                "val_severity_loss": val_metrics["severity_loss"],
                "val_ortho_loss": val_metrics["ortho_loss"],
                "val_cons_loss": val_metrics["cons_loss"],
                "val_div_loss": val_metrics["div_loss"],
                "val_specialist_acc": val_metrics["specialist_acc"],
                "val_severity_acc": val_metrics["severity_acc"],
                "lr": train_metrics["lr"],
            }
            self.history.append(epoch_data)

            # Console output formatting
            if self.is_rank_0():
                print(
                    f"Epoch {epoch:02d} | Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | Spec Acc: {val_metrics['specialist_acc']:.2%} | "
                    f"Sev Acc: {val_metrics['severity_acc']:.2%} | Time: {epoch_time:.1f}s"
                )

            # Check if this is the best epoch
            monitored = self.config.early_stopping_metric
            val_val = (
                val_metrics["loss"]
                if monitored == "val_loss"
                else val_metrics.get(monitored.replace("val_", ""), 0.0)
            )

            # For validation accuracy, we want to maximize it, for loss we minimize it
            is_better = False
            if "acc" in monitored:
                if (
                    epoch == 1
                    or val_val
                    > self.best_val_loss + self.config.early_stopping_min_improvement
                ):
                    is_better = True
            else:
                if (
                    epoch == 1
                    or val_val
                    < self.best_val_loss - self.config.early_stopping_min_improvement
                ):
                    is_better = True

            if is_better:
                self.best_val_loss = val_val
                self.patience_counter = 0
                self.best_metrics = epoch_data
                if self.is_rank_0():
                    self.save_checkpoint(
                        self.checkpoint_dir / "best_model.pt", epoch, is_best=True
                    )
            else:
                self.patience_counter += 1

            # Always save latest model checkpoint
            if self.is_rank_0():
                self.save_checkpoint(
                    self.checkpoint_dir / "latest_model.pt", epoch, is_best=False
                )

            # Early stopping check
            if self.patience_counter >= self.config.early_stopping_patience:
                if self.is_rank_0():
                    print(
                        f"Early stopping triggered at epoch {epoch} (metric '{monitored}' did not improve for {self.config.early_stopping_patience} epochs)."
                    )
                break

        # Export report files
        if self.is_rank_0():
            self.export_metrics()
        return self.best_metrics

    def save_checkpoint(self, path: Path, epoch: int, is_best: bool = False) -> None:
        """Create and save persistent training state checkpoint."""
        extra_states = {
            "epoch": epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "random_seed_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "triage_config": (
                self.model.config.to_dict()
                if hasattr(self.model, "config")
                and hasattr(self.model.config, "to_dict")
                else {}
            ),
            "metadata": {
                "random_seed": self.config.seed,
                "git_commit": get_git_commit(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "is_best": is_best,
                "use_amp": self.use_amp,
            },
        }

        # Safe config serialization
        serialized_config = {
            k: v
            for k, v in self.config.__dict__.copy().items()
            if not k.startswith("_")
        }

        mgr_save_checkpoint(
            path=path,
            model_short_name="emergent_path_triage",
            backbone_name="xlm-roberta-base",
            config=serialized_config,
            state_dict=self.model.state_dict(),
            extra_states=extra_states,
        )

    def load_checkpoint(self, path: Path) -> int:
        """Reload saved model parameters and training states from a checkpoint."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        checkpoint = mgr_load_checkpoint(path, map_location=self.device)

        # Checkpoint validation metadata diagnostics
        metadata = checkpoint.get("metadata", {})
        opt_groups = (
            len(checkpoint["optimizer_state_dict"]["param_groups"])
            if "optimizer_state_dict" in checkpoint
            else 0
        )
        has_scheduler = (
            "Present" if checkpoint.get("scheduler_state_dict") is not None else "N/A"
        )
        amp_enabled = metadata.get("use_amp", False)

        # Calculate model parameter shape hash
        model_hash_list = [str(p.shape) for p in self.model.parameters()]
        model_hash = str(hash(tuple(model_hash_list)))

        # Check compatibility version
        compat_passed = "Fail"
        try:
            from models.emergent_path_triage.model import EmergentPathCheckpointRegistry

            registry = EmergentPathCheckpointRegistry()
            registry.verify_compatibility(metadata, self.model.config)
            compat_passed = "Pass"
        except Exception as e:
            print(f"Warning: Checkpoint compatibility check failed: {e}")

        print("=" * 60)
        print("CHECKPOINT METADATA VALIDATION REPORT")
        print("=" * 60)
        print(f"  Checkpoint Path       : {path}")
        print(f"  Epoch                 : {checkpoint.get('epoch', 'N/A')}")
        print(f"  Optimizer Groups      : {opt_groups}")
        print(f"  Scheduler State       : {has_scheduler}")
        print(f"  AMP Enabled           : {amp_enabled}")
        print(f"  Model Hash            : {model_hash}")
        print(f"  PyTorch Version        : {torch.__version__}")
        print(
            f"  CUDA Version          : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}"
        )
        print(
            f"  Checkpoint Compat     : {metadata.get('compatibility_version', 'N/A')}"
        )
        print(f"  Pass/Fail Status      : {compat_passed}")
        print("=" * 60)

        try:
            self.resumed_from = path

            # Restore AMP & scaler settings dynamically
            if "use_amp" in metadata:
                self.use_amp = metadata["use_amp"]
                device_type = "cuda" if self.device.type == "cuda" else "cpu"
                self.scaler = GradScaler(device=device_type, enabled=self.use_amp)
                print(f"Restored AMP status from checkpoint: {self.use_amp}")

            # Store precision mode on model config for extended diagnostics
            self.model.config.checkpoint_precision_mode = (
                "AMP-Enabled" if amp_enabled else "AMP-Disabled"
            )

            # Intelligent Checkpoint Loading Sequence
            try:
                # 1. Attempt strict=True first
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                print("Successfully loaded model parameters with strict=True.")
                loading_strategy = "strict"
                compatibility_verdict = "VERIFIED_MATCH"
                incompatible_keys = None
            except RuntimeError as e:
                # 2. Check if the checkpoint is a verified legacy baseline model
                print(f"Warning: Strict loading failed: {e}")
                print("Running checkpoint compatibility fingerprint analysis...")

                state_keys = set(checkpoint["model_state_dict"].keys())
                model_keys = set(self.model.state_dict().keys())

                missing_keys = list(model_keys - state_keys)
                unexpected_keys = list(state_keys - model_keys)

                # Fingerprint check:
                # a) Must contain primary encoder backbone weights (e.g. embeddings weight)
                has_encoder = any("encoder.embeddings." in k for k in state_keys)
                # b) Must contain prediction classifier weights
                has_classifiers = any(
                    "classifier_specialist." in k for k in state_keys
                ) and any("classifier_severity." in k for k in state_keys)

                # c) Check if missing keys are auxiliary/new additions (e.g. router parameters)
                core_missing = [
                    k
                    for k in missing_keys
                    if "encoder." in k
                    or "classifier_specialist." in k
                    or "classifier_severity." in k
                ]

                if has_encoder and has_classifiers and not core_missing:
                    print(
                        "Positive identification: Verified legacy baseline/ablation checkpoint. Retrying with strict=False..."
                    )
                    incompatible_keys = self.model.load_state_dict(
                        checkpoint["model_state_dict"], strict=False
                    )
                    loading_strategy = "non-strict"
                    compatibility_verdict = "VERIFIED_LEGACY"
                else:
                    print(
                        "Checkpoint compatibility analysis failed: missing core parameters."
                    )
                    raise CompatibilityError(
                        f"Incompatible checkpoint. Missing core parameters: {core_missing}. "
                        "Cannot load with strict=False."
                    ) from e

                # 3. Generate checkpoint_loading_report.md
                report_path = self.checkpoint_dir / "checkpoint_loading_report.md"
                try:
                    with open(report_path, "w", encoding="utf-8") as rf:
                        rf.write("# Checkpoint Loading Report\n\n")
                        rf.write(f"- **Checkpoint Path**: `{path}`\n")
                        rf.write(
                            f"- **Checkpoint Version**: `{metadata.get('compatibility_version', 'N/A')}`\n"
                        )
                        rf.write(
                            f"- **Repository Commit**: `{metadata.get('git_commit', 'N/A')}`\n"
                        )
                        rf.write(
                            f"- **Architecture Fingerprint**: `LatentDim: {self.model.config.latent_dim}, Enc: {getattr(self.model, 'model_name', self.model.__class__.__name__)}`\n"
                        )
                        rf.write(
                            f"- **Loaded Parameters Count**: `{len(state_keys)}`\n"
                        )
                        rf.write(f"- **Missing Keys**: `{len(missing_keys)}` missing\n")
                        rf.write(
                            f"- **Unexpected Keys**: `{len(unexpected_keys)}` unexpected\n"
                        )
                        rf.write(
                            f"- **Ignored Keys**: `{len(unexpected_keys)}` ignored\n"
                        )
                        rf.write(
                            f"- **Compatibility Verdict**: `{compatibility_verdict}`\n"
                        )
                        rf.write(f"- **Loading Strategy Used**: `{loading_strategy}`\n")
                        rf.write("- **Final Status**: `SUCCESS`\n\n")
                        rf.write("## Missing Keys Details\n\n")
                        if missing_keys:
                            rf.writelines(f"- `{k}`\n" for k in sorted(missing_keys))
                        else:
                            rf.write("*None*\n")
                        rf.write("\n## Unexpected Keys Details\n\n")
                        if unexpected_keys:
                            rf.writelines(f"- `{k}`\n" for k in sorted(unexpected_keys))
                        else:
                            rf.write("*None*\n")
                    print(f"Logged checkpoint loading report to {report_path}")
                except Exception as report_err:
                    print(
                        f"Warning: Could not write checkpoint loading report: {report_err}"
                    )

            # Try loading optimizer state gracefully (mismatches can occur with architecture changes/baselines)
            if (
                "optimizer_state_dict" in checkpoint
                and checkpoint["optimizer_state_dict"] is not None
            ):
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    print("Successfully reloaded optimizer state.")
                except Exception as opt_err:
                    raise RuntimeError(
                        f"Failed to restore optimizer state: {opt_err}"
                    ) from opt_err
            else:
                print(
                    "Warning: optimizer_state_dict not found in checkpoint. Skipping optimizer restoration."
                )

            # Try loading scheduler state gracefully
            if (
                self.scheduler is not None
                and "scheduler_state_dict" in checkpoint
                and checkpoint["scheduler_state_dict"] is not None
            ):
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    print("Successfully reloaded scheduler state.")
                except Exception as sched_err:
                    raise RuntimeError(
                        f"Failed to restore scheduler state: {sched_err}"
                    ) from sched_err

            # Try loading scaler state gracefully
            if (
                "scaler_state_dict" in checkpoint
                and checkpoint["scaler_state_dict"]
                and len(checkpoint["scaler_state_dict"]) > 0
            ):
                try:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    print("Successfully reloaded GradScaler state.")
                except Exception as scaler_err:
                    raise RuntimeError(
                        f"Failed to restore scaler state: {scaler_err}"
                    ) from scaler_err

            # Restore seed states
            if (
                "random_seed_states" in checkpoint
                and checkpoint["random_seed_states"] is not None
            ):
                seeds = checkpoint["random_seed_states"]
                try:
                    random.setstate(seeds["python"])
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to restore Python RNG state: {e}"
                    ) from e

                try:
                    np_state = seeds["numpy"]
                    if isinstance(np_state, list):
                        if len(np_state) == 5 and isinstance(np_state[1], list):
                            np_state = (
                                np_state[0],
                                np.array(np_state[1], dtype=np.uint32),
                                np_state[2],
                                np_state[3],
                                np_state[4],
                            )
                        else:
                            np_state = tuple(np_state)
                    np.random.set_state(np_state)
                except Exception as e:
                    raise RuntimeError(f"Failed to restore NumPy RNG state: {e}") from e

                try:
                    torch_state = seeds["torch"]
                    if isinstance(torch_state, list):
                        torch_state = torch.ByteTensor(torch_state)
                    elif (
                        isinstance(torch_state, torch.Tensor)
                        and torch_state.dtype != torch.uint8
                    ):
                        torch_state = torch_state.to(torch.uint8)
                    if isinstance(torch_state, torch.Tensor):
                        torch_state = torch_state.cpu()
                    torch.set_rng_state(torch_state)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to restore PyTorch CPU RNG state: {e}"
                    ) from e

                if torch.cuda.is_available() and seeds.get("torch_cuda") is not None:
                    try:
                        num_devices = torch.cuda.device_count()
                        for idx in range(min(num_devices, len(seeds["torch_cuda"]))):
                            state = seeds["torch_cuda"][idx]
                            if isinstance(state, list):
                                state = torch.ByteTensor(state)
                            elif (
                                isinstance(state, torch.Tensor)
                                and state.dtype != torch.uint8
                            ):
                                state = state.to(torch.uint8)
                            if isinstance(state, torch.Tensor):
                                state = state.cpu()
                            torch.cuda.set_rng_state(state, device=idx)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to restore PyTorch CUDA RNG state: {e}"
                        ) from e

            self.history = checkpoint.get("history", [])
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            self.patience_counter = checkpoint.get("patience_counter", 0)

            start_epoch = checkpoint.get("epoch", 0)
            self.start_epoch = start_epoch + 1
            print(f"Resumed training from checkpoint: {path} (Epoch {start_epoch})")
            return start_epoch

        except Exception as e:
            print("=" * 60)
            print("CRITICAL ERROR: CHECKPOINT RESTORATION FAILED")
            print("=" * 60)
            print(f"  Checkpoint Path : {path}")
            print(f"  Error Details   : {e}")
            import traceback

            traceback.print_exc()
            print("=" * 60)
            raise RuntimeError(f"Atomic checkpoint load failed: {e}") from e

    def export_metrics(self) -> None:
        """Write evaluation reports to disk."""
        # 1. training_history.csv
        csv_path = self.checkpoint_dir / "training_history.csv"
        if self.history:
            keys = self.history[0].keys()
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.history)

        # 2. experiment_config.json
        cfg_path = self.checkpoint_dir / "experiment_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=4)

        # 3. best_metrics.json
        best_path = self.checkpoint_dir / "best_metrics.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(self.best_metrics, f, indent=4)

        print(f"Exported metrics reports to checkpoint folder: {self.checkpoint_dir}")
