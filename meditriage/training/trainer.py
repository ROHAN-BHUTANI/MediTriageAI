"""Production-Grade Multi-Task Clinical Transformer Trainer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from meditriage.training.callbacks import Callback, EarlyStopping
from meditriage.training.checkpoint import CheckpointManager
from meditriage.training.config import TrainingConfig
from meditriage.training.logger import ExperimentLogger
from meditriage.training.losses import MultiTaskLoss
from meditriage.training.metrics import ClinicalMetricsCalculator
from meditriage.training.optimizer import get_optimizer
from meditriage.training.scheduler import get_scheduler
from meditriage.training.seed import set_seed

logger = logging.getLogger("meditriage.training")


class MultiTaskClinicalClassifier(nn.Module):
    """Multi-task Classification Head wrapping a HuggingFace Transformer backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int = 768,
        num_triage_classes: int = 5,
        num_dept_classes: int = 8,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout_rate)
        self.triage_head = nn.Linear(hidden_size, num_triage_classes)
        self.dept_head = nn.Linear(hidden_size, num_dept_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through backbone and multi-task heads."""
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**kwargs)
        # Extract pooled or [CLS] representations
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        pooled = self.dropout(pooled)
        triage_logits = self.triage_head(pooled)
        dept_logits = self.dept_head(pooled)
        return triage_logits, dept_logits


class Trainer:
    """Production-grade Trainer for multi-task clinical classification models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader | None = None,
        eval_dataloader: DataLoader | None = None,
        callbacks: list[Callback] | None = None,
        device: torch.device | str | None = None,
    ):
        self.cfg = config
        set_seed(self.cfg.seed)

        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []

        self.optimizer = get_optimizer(self.model, self.cfg)
        num_train_steps = (
            len(self.train_dataloader)
            * getattr(self.cfg, "num_epochs", getattr(self.cfg, "epochs", 3))
            // getattr(self.cfg, "gradient_accumulation_steps", 1)
            if self.train_dataloader
            else 100
        )
        self.scheduler = get_scheduler(self.optimizer, self.cfg, num_train_steps)
        loss_type = getattr(self.cfg, "loss_type", "focal_ordinal")
        self.loss_fn = MultiTaskLoss(
            loss_type=loss_type,
            triage_weight=getattr(self.cfg, "triage_loss_weight", 1.0),
            dept_weight=getattr(self.cfg, "dept_loss_weight", 1.0),
            focal_gamma=getattr(self.cfg, "focal_gamma", 2.0),
        ).to(self.device)

        use_amp = getattr(self.cfg, "use_amp", True) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        output_dir = getattr(self.cfg, "output_dir", "results")
        self.checkpoint_manager = CheckpointManager(output_dir)
        self.logger = ExperimentLogger(output_dir)

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

    def _forward_model(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.model(input_ids, attention_mask)
        if hasattr(outputs, "severity_logits") and hasattr(outputs, "specialist_logits"):
            return outputs.severity_logits, outputs.specialist_logits
        elif isinstance(outputs, tuple):
            if len(outputs) == 2:
                first, second = outputs
                if first.shape[-1] == 5:
                    return first, second
                else:
                    return second, first
        return outputs, None

    def _extract_batch_targets(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        triage_targets = batch.get("labels_severity")
        if triage_targets is None:
            triage_targets = batch.get("triage_label")
        if triage_targets is not None:
            triage_targets = triage_targets.to(self.device)

        dept_targets = batch.get("labels_specialist")
        if dept_targets is None:
            dept_targets = batch.get("dept_label")
        if dept_targets is not None:
            dept_targets = dept_targets.to(self.device)

        return triage_targets, dept_targets

    def train(self) -> dict[str, Any]:
        """Execute full training loop across epochs."""
        if not self.train_dataloader:
            raise ValueError("train_dataloader is required for training.")

        logger.info("Starting training loop on device: %s", self.device)
        for cb in self.callbacks:
            cb.on_train_begin()

        num_epochs = getattr(self.cfg, "num_epochs", getattr(self.cfg, "epochs", 3))
        grad_accum = getattr(self.cfg, "gradient_accumulation_steps", 1)
        use_amp = getattr(self.cfg, "use_amp", True) and torch.cuda.is_available()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.model.train()
            total_train_loss = 0.0

            for cb in self.callbacks:
                cb.on_epoch_begin(epoch)

            self.optimizer.zero_grad()
            for step, batch in enumerate(self.train_dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                triage_targets, dept_targets = self._extract_batch_targets(batch)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    triage_logits, dept_logits = self._forward_model(input_ids, attention_mask)
                    loss, _loss_metrics = self.loss_fn(
                        triage_logits, triage_targets, dept_logits, dept_targets
                    )
                    loss = loss / grad_accum

                self.scaler.scale(loss).backward()
                total_train_loss += loss.item() * grad_accum

                if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = float(
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.max_grad_norm
                        )
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        pass
                    else:
                        self.scheduler.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                    current_lr = self.optimizer.param_groups[0]["lr"]
                    for cb in self.callbacks:
                        cb.on_batch_end(
                            self.global_step, {"lr": current_lr, "grad_norm": grad_norm}
                        )

            # Epoch Validation
            avg_train_loss = total_train_loss / max(len(self.train_dataloader), 1)
            epoch_logs = {"train_loss": round(avg_train_loss, 4), "epoch": epoch}

            if self.eval_dataloader:
                eval_metrics = self.validate(self.eval_dataloader)
                epoch_logs.update(eval_metrics)

                # Save best checkpoint
                main_metric = eval_metrics.get("eval_macro_f1", 0.0)
                if main_metric > self.best_metric:
                    self.best_metric = main_metric
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        global_step=self.global_step,
                        config=self.cfg,
                        metrics=eval_metrics,
                        filename="best_model.pt",
                    )

            self.logger.log_metrics(epoch, epoch_logs, prefix="epoch")

            # Save latest checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                epoch=epoch,
                global_step=self.global_step,
                config=self.cfg,
                metrics=epoch_logs,
                filename="checkpoint_latest.pt",
            )

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, epoch_logs)
                if isinstance(cb, EarlyStopping) and cb.should_stop:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        for cb in self.callbacks:
            cb.on_train_end()

        logger.info("Training complete. Best eval metric: %.4f", self.best_metric)
        return {"best_eval_metric": self.best_metric, "global_step": self.global_step}

    def validate(self, dataloader: DataLoader | None = None) -> dict[str, Any]:
        """Evaluate model on dataloader."""
        dl = dataloader or self.eval_dataloader
        if not dl:
            raise ValueError("No dataloader provided for evaluation.")

        self.model.eval()
        all_triage_logits = []
        all_triage_labels = []
        use_amp = getattr(self.cfg, "use_amp", True) and torch.cuda.is_available()

        with torch.no_grad():
            for batch in dl:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                triage_targets, _ = self._extract_batch_targets(batch)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    triage_logits, _ = self._forward_model(input_ids, attention_mask)

                all_triage_logits.append(triage_logits.cpu().numpy())
                all_triage_labels.append(triage_targets.cpu().numpy())

        logits_arr = np.concatenate(all_triage_logits, axis=0)
        labels_arr = np.concatenate(all_triage_labels, axis=0)

        metrics = ClinicalMetricsCalculator.compute_all_metrics(
            logits_arr, labels_arr, prefix="eval"
        )
        return metrics

    def test(self, test_dataloader: DataLoader) -> dict[str, Any]:
        """Evaluate model on test dataset."""
        logger.info("Evaluating on test dataloader...")
        metrics = self.validate(test_dataloader)
        test_metrics = {k.replace("eval_", "test_"): v for k, v in metrics.items()}
        self.logger.log_metrics(self.global_step, test_metrics, prefix="test")
        return test_metrics

    def predict(self, dataloader: DataLoader) -> dict[str, np.ndarray]:
        """Run inference and return probabilities and predictions."""
        self.model.eval()
        all_triage_probs = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                triage_logits, _ = self._forward_model(input_ids, attention_mask)
                probs = torch.softmax(triage_logits, dim=-1).cpu().numpy()
                all_triage_probs.append(probs)

        probs_arr = np.concatenate(all_triage_probs, axis=0)
        preds_arr = np.argmax(probs_arr, axis=1)

        return {"probabilities": probs_arr, "predictions": preds_arr}

    def resume(self, checkpoint_path: str | Path) -> None:
        """Resume training from a saved checkpoint."""
        logger.info("Resuming training from checkpoint: %s", checkpoint_path)
        info = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        self.current_epoch = info["epoch"] + 1
        self.global_step = info["global_step"]
        logger.info(
            "Resumed state at epoch %d, step %d", self.current_epoch, self.global_step
        )
