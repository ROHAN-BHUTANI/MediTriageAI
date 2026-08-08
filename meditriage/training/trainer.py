"""Production-Grade Multi-Task Clinical Transformer Trainer.

Forensic instrumentation added for production observability.
All logging is INFO/DEBUG level via the existing meditriage.training logger.
Zero behavioural impact on training mathematics, model, loss, optimizer,
scheduler, gradient accumulation, checkpointing, evaluation, or metrics.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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

# ── Ensure the logger can emit to stderr even before ExperimentLogger adds
#    its file handler.  Without a StreamHandler, logger.info() calls inside
#    __init__ and early train() are silently swallowed because the logger has
#    no handler attached yet (the file handler is added later by
#    ExperimentLogger.__init__).
if not any(isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
           for h in logger.handlers):
    _console_handler = logging.StreamHandler(sys.stderr)
    _console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_console_handler)
    logger.setLevel(logging.DEBUG)


class MultiTaskClinicalClassifier(nn.Module):
    """Multi-task Classification Head wrapping a HuggingFace Transformer backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int = 768,
        num_triage_classes: int = 5,
        num_dept_classes: int = 13,
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
        logger.info("[TRAINER-INIT] Trainer.__init__() ENTER")
        self.cfg = config
        set_seed(self.cfg.seed)

        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("[TRAINER-INIT] Device resolved: %s", self.device)

        self.model = model.to(self.device)
        logger.info("[TRAINER-INIT] Model moved to device. Parameters: %d",
                     sum(p.numel() for p in self.model.parameters()))

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []
        logger.info("[TRAINER-INIT] Callbacks registered: %d (%s)",
                     len(self.callbacks),
                     [type(cb).__name__ for cb in self.callbacks])

        # ── DataLoader diagnostics (logged once) ──
        self._log_dataloader_config("train", self.train_dataloader)
        self._log_dataloader_config("eval", self.eval_dataloader)

        self.optimizer = get_optimizer(self.model, self.cfg)
        logger.info("[TRAINER-INIT] Optimizer created: %s", type(self.optimizer).__name__)

        num_train_steps = (
            len(self.train_dataloader)
            * getattr(self.cfg, "num_epochs", getattr(self.cfg, "epochs", 3))
            // getattr(self.cfg, "gradient_accumulation_steps", 1)
            if self.train_dataloader
            else 100
        )
        logger.info("[TRAINER-INIT] Computed num_train_steps: %d", num_train_steps)

        self.scheduler = get_scheduler(self.optimizer, self.cfg, num_train_steps)
        logger.info("[TRAINER-INIT] Scheduler created: %s", type(self.scheduler).__name__)

        loss_type = getattr(self.cfg, "loss_type", "focal_ordinal")
        self.loss_fn = MultiTaskLoss(
            loss_type=loss_type,
            triage_weight=getattr(self.cfg, "triage_loss_weight", 1.0),
            dept_weight=getattr(self.cfg, "dept_loss_weight", 1.0),
            focal_gamma=getattr(self.cfg, "focal_gamma", 2.0),
        ).to(self.device)
        logger.info("[TRAINER-INIT] Loss function created: %s (type=%s)", type(self.loss_fn).__name__, loss_type)

        use_amp = getattr(self.cfg, "use_amp", True) and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        logger.info("[TRAINER-INIT] AMP enabled: %s | GradScaler enabled: %s", use_amp, self.scaler.is_enabled())

        output_dir = getattr(self.cfg, "output_dir", "results")
        self.checkpoint_manager = CheckpointManager(output_dir)
        self.logger = ExperimentLogger(output_dir)
        logger.info("[TRAINER-INIT] CheckpointManager & ExperimentLogger initialized (output_dir=%s)", output_dir)

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # ── GPU audit (logged once) ──
        self._log_gpu_audit()

        logger.info("[TRAINER-INIT] Trainer.__init__() EXIT — ready for train()")

    def _log_dataloader_config(self, name: str, dl: DataLoader | None) -> None:
        """Log DataLoader configuration details once."""
        if dl is None:
            logger.info("[DATALOADER-AUDIT] %s_dataloader is None", name)
            return
        logger.info("[DATALOADER-AUDIT] %s_dataloader: len=%d, batch_size=%s, "
                     "num_workers=%s, persistent_workers=%s, pin_memory=%s, prefetch_factor=%s",
                     name, len(dl),
                     getattr(dl, "batch_size", "?"),
                     getattr(dl, "num_workers", "?"),
                     getattr(dl, "persistent_workers", "?"),
                     getattr(dl, "pin_memory", "?"),
                     getattr(dl, "prefetch_factor", "?"))

    def _log_gpu_audit(self) -> None:
        """Log GPU information once."""
        if torch.cuda.is_available():
            logger.info("[GPU-AUDIT] CUDA device: %s", torch.cuda.get_device_name(0))
            logger.info("[GPU-AUDIT] Allocated: %.2f MB | Reserved: %.2f MB",
                         torch.cuda.memory_allocated(0) / 1e6,
                         torch.cuda.memory_reserved(0) / 1e6)
        else:
            logger.info("[GPU-AUDIT] CUDA not available — running on CPU")

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
        logger.info("[TRAIN] ══════════════════════════════════════════")
        logger.info("[TRAIN] ENTER train()")
        logger.info("[TRAIN] ══════════════════════════════════════════")

        if not self.train_dataloader:
            raise ValueError("train_dataloader is required for training.")

        logger.info("[TRAIN] Starting training loop on device: %s", self.device)

        num_epochs = getattr(self.cfg, "num_epochs", getattr(self.cfg, "epochs", 3))
        grad_accum = getattr(self.cfg, "gradient_accumulation_steps", 1)
        use_amp = getattr(self.cfg, "use_amp", True) and torch.cuda.is_available()
        total_batches = len(self.train_dataloader)

        logger.info("[TRAIN] Config: num_epochs=%d, grad_accum=%d, use_amp=%s, "
                     "total_batches_per_epoch=%d, current_epoch=%d, global_step=%d",
                     num_epochs, grad_accum, use_amp, total_batches,
                     self.current_epoch, self.global_step)

        for cb in self.callbacks:
            cb.on_train_begin()
        logger.info("[TRAIN] on_train_begin callbacks fired")

        train_start = time.monotonic()

        for epoch in range(self.current_epoch, num_epochs):
            epoch_start = time.monotonic()
            self.current_epoch = epoch
            self.model.train()
            total_train_loss = 0.0

            logger.info("[EPOCH %d/%d] ──────── BEGIN ────────", epoch, num_epochs - 1)

            for cb in self.callbacks:
                cb.on_epoch_begin(epoch)

            self.optimizer.zero_grad()

            # ── Create DataLoader iterator ──
            logger.info("[EPOCH %d] Creating DataLoader iterator...", epoch)
            iter_start = time.monotonic()
            try:
                train_iter = iter(self.train_dataloader)
            except Exception:
                logger.error("[EPOCH %d] FATAL: Failed to create DataLoader iterator!\n%s",
                              epoch, traceback.format_exc())
                raise
            iter_time = time.monotonic() - iter_start
            logger.info("[EPOCH %d] DataLoader iterator created in %.4fs", epoch, iter_time)

            last_grad_norm_str = "N/A"
            pbar = tqdm(
                range(total_batches),
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                unit="batch",
                dynamic_ncols=True,
                file=sys.stdout,
            )

            for step in pbar:
                try:
                    # ── Fetch batch ──
                    t_fetch_start = time.monotonic()
                    logger.debug("[EPOCH %d][STEP %d/%d] Waiting for batch...", epoch, step, total_batches - 1)
                    batch = next(train_iter)
                    t_fetch = time.monotonic() - t_fetch_start

                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    triage_targets, dept_targets = self._extract_batch_targets(batch)

                    # ── First-batch diagnostics (once per epoch) ──
                    if step == 0:
                        logger.info("[EPOCH %d][BATCH-0] First batch received in %.4fs", epoch, t_fetch)
                        logger.info("[EPOCH %d][BATCH-0] batch_size=%d | input_ids.shape=%s | "
                                     "attention_mask.shape=%s | device=%s | dtype=%s",
                                     epoch, input_ids.shape[0], list(input_ids.shape),
                                     list(attention_mask.shape), input_ids.device, input_ids.dtype)
                        if triage_targets is not None:
                            logger.info("[EPOCH %d][BATCH-0] labels_severity.shape=%s | device=%s",
                                         epoch, list(triage_targets.shape), triage_targets.device)
                        if dept_targets is not None:
                            logger.info("[EPOCH %d][BATCH-0] labels_specialist.shape=%s | device=%s",
                                         epoch, list(dept_targets.shape), dept_targets.device)
                        if torch.cuda.is_available():
                            logger.info("[EPOCH %d][BATCH-0] GPU peak memory: %.2f MB",
                                         epoch, torch.cuda.max_memory_allocated(0) / 1e6)

                    # ── Forward ──
                    t_fwd_start = time.monotonic()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        triage_logits, dept_logits = self._forward_model(input_ids, attention_mask)
                        loss, _loss_metrics = self.loss_fn(
                            triage_logits, triage_targets, dept_logits, dept_targets
                        )
                        loss = loss / grad_accum
                    t_fwd = time.monotonic() - t_fwd_start

                    # ── Backward ──
                    t_bwd_start = time.monotonic()
                    self.scaler.scale(loss).backward()
                    t_bwd = time.monotonic() - t_bwd_start

                    total_train_loss += loss.item() * grad_accum

                    # ── Optimizer step (gradient accumulation boundary) ──
                    if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                        t_opt_start = time.monotonic()
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
                        t_opt = time.monotonic() - t_opt_start

                        current_lr = self.optimizer.param_groups[0]["lr"]
                        gn_val = grad_norm
                        last_grad_norm_str = (
                            "AMP overflow (skipped)"
                            if (np.isnan(gn_val) or np.isinf(gn_val))
                            else f"{gn_val:.4f}"
                        )
                        for cb in self.callbacks:
                            cb.on_batch_end(
                                self.global_step, {"lr": current_lr, "grad_norm": grad_norm}
                            )

                    # ── Update tqdm progress bar on every step ──
                    running_loss_avg = total_train_loss / (step + 1)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(
                        loss=f"{running_loss_avg:.4f}",
                        lr=f"{current_lr:.2e}",
                        grad_norm=last_grad_norm_str,
                        refresh=False,
                    )

                except Exception:
                    logger.error("[EPOCH %d][STEP %d] EXCEPTION during training step!\n%s",
                                  epoch, step, traceback.format_exc())
                    raise

            pbar.close()

            # ── Epoch summary ──
            avg_train_loss = total_train_loss / max(total_batches, 1)
            epoch_logs = {"train_loss": round(avg_train_loss, 4), "epoch": epoch}

            logger.info("[EPOCH %d] Training phase complete: avg_train_loss=%.4f | steps=%d",
                         epoch, avg_train_loss, total_batches)

            # ── Validation ──
            if self.eval_dataloader:
                logger.info("[EPOCH %d] Starting validation...", epoch)
                t_val_start = time.monotonic()
                try:
                    eval_metrics = self.validate(self.eval_dataloader)
                except Exception:
                    logger.error("[EPOCH %d] EXCEPTION during validation!\n%s",
                                  epoch, traceback.format_exc())
                    raise
                t_val = time.monotonic() - t_val_start
                epoch_logs.update(eval_metrics)
                logger.info("[EPOCH %d] Validation complete in %.2fs: %s",
                             epoch, t_val,
                             {k: (f"{v:.4f}" if isinstance(v, float) else v)
                              for k, v in eval_metrics.items()
                              if not k.endswith("_matrix") and not isinstance(v, (dict, list))})

                # Save best checkpoint
                main_metric = eval_metrics.get("eval_macro_f1", 0.0)
                is_best = main_metric > self.best_metric
                if is_best:
                    logger.info("[EPOCH %d] New best metric: %.4f → %.4f — saving best checkpoint",
                                 epoch, self.best_metric, main_metric)
                    self.best_metric = main_metric
                    t_ckpt_start = time.monotonic()
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
                    logger.info("[EPOCH %d] Best checkpoint saved in %.2fs",
                                 epoch, time.monotonic() - t_ckpt_start)

            self.logger.log_metrics(epoch, epoch_logs, prefix="epoch")

            # Save latest checkpoint
            logger.info("[EPOCH %d] Saving latest checkpoint...", epoch)
            t_ckpt_start = time.monotonic()
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
            logger.info("[EPOCH %d] Latest checkpoint saved in %.2fs",
                         epoch, time.monotonic() - t_ckpt_start)

            epoch_duration = time.monotonic() - epoch_start
            logger.info("[EPOCH %d/%d] ──────── COMPLETE (%.2fs) ────────",
                         epoch, num_epochs - 1, epoch_duration)

            gpu_mem = (
                round(torch.cuda.max_memory_allocated(0) / 1e6, 1)
                if torch.cuda.is_available()
                else 0.0
            )
            ckpt_status = (
                "Saved (Best)"
                if (self.eval_dataloader and epoch_logs.get("eval_macro_f1", 0.0) == self.best_metric)
                else "Saved"
            )

            # ── Render Rich 15-Column Publication Terminal Table ──
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                pub_table = Table(
                    title=f"Publication Training Epoch {epoch + 1}/{num_epochs} Completion Summary",
                    show_header=True,
                    header_style="bold magenta",
                )
                pub_table.add_column("Epoch", justify="center")
                pub_table.add_column("Train Loss", justify="right")
                pub_table.add_column("Val Loss", justify="right")
                pub_table.add_column("Spec Acc", justify="right")
                pub_table.add_column("Sev Acc", justify="right")
                pub_table.add_column("Joint Acc", justify="right")
                pub_table.add_column("Spec F1", justify="right")
                pub_table.add_column("Sev F1", justify="right")
                pub_table.add_column("Macro F1", justify="right")
                pub_table.add_column("Weighted F1", justify="right")
                pub_table.add_column("LR", justify="right")
                pub_table.add_column("Grad Norm", justify="right")
                pub_table.add_column("Time", justify="right")
                pub_table.add_column("GPU Mem", justify="right")
                pub_table.add_column("Checkpoint", justify="center")

                pub_table.add_row(
                    f"{epoch + 1}/{num_epochs}",
                    f"{avg_train_loss:.4f}",
                    f"{epoch_logs.get('eval_loss', 0.0):.4f}",
                    f"{epoch_logs.get('eval_specialist_accuracy', 0.0):.2%}",
                    f"{epoch_logs.get('eval_accuracy', 0.0):.2%}",
                    f"{epoch_logs.get('eval_joint_accuracy', 0.0):.2%}",
                    f"{epoch_logs.get('eval_specialist_macro_f1', 0.0):.4f}",
                    f"{epoch_logs.get('eval_macro_f1', 0.0):.4f}",
                    f"{epoch_logs.get('eval_macro_f1', 0.0):.4f}",
                    f"{epoch_logs.get('eval_weighted_f1', 0.0):.4f}",
                    f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    last_grad_norm_str,
                    f"{epoch_duration:.1f}s",
                    f"{gpu_mem:.0f} MB",
                    ckpt_status,
                )
                console.print(pub_table)
            except Exception as e:
                logger.warning("[TRAINER] Rich table render exception: %s", e)

            # ── Export Publication Metrics (CSV, JSON, Markdown) ──
            try:
                output_path = Path(getattr(self.cfg, "output_dir", "results"))
                output_path.mkdir(parents=True, exist_ok=True)

                json_path = output_path / "epoch_metrics.json"
                epoch_rec = {
                    "epoch": epoch + 1,
                    "train_loss": round(avg_train_loss, 4),
                    "val_loss": round(float(epoch_logs.get("eval_loss", 0.0)), 4),
                    "specialist_accuracy": round(float(epoch_logs.get("eval_specialist_accuracy", 0.0)), 4),
                    "severity_accuracy": round(float(epoch_logs.get("eval_accuracy", 0.0)), 4),
                    "joint_accuracy": round(float(epoch_logs.get("eval_joint_accuracy", 0.0)), 4),
                    "specialist_macro_f1": round(float(epoch_logs.get("eval_specialist_macro_f1", 0.0)), 4),
                    "severity_macro_f1": round(float(epoch_logs.get("eval_macro_f1", 0.0)), 4),
                    "macro_f1": round(float(epoch_logs.get("eval_macro_f1", 0.0)), 4),
                    "weighted_f1": round(float(epoch_logs.get("eval_weighted_f1", 0.0)), 4),
                    "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                    "grad_norm": last_grad_norm_str,
                    "epoch_duration_s": round(epoch_duration, 2),
                    "gpu_memory_mb": gpu_mem,
                    "checkpoint_status": ckpt_status,
                }
                history = []
                if json_path.exists():
                    try:
                        import json as _json
                        history = _json.loads(json_path.read_text(encoding="utf-8"))
                    except Exception:
                        history = []
                history.append(epoch_rec)
                import json as _json
                json_path.write_text(_json.dumps(history, indent=2), encoding="utf-8")

                csv_path = output_path / "epoch_metrics.csv"
                import pandas as pd
                pd.DataFrame(history).to_csv(csv_path, index=False)

                md_path = output_path / "summary.md"
                md_lines = [
                    "# Publication Training Summary\n",
                    "| Epoch | Train Loss | Val Loss | Spec Acc | Sev Acc | Joint Acc | Spec F1 | Sev F1 | Macro F1 | Weighted F1 | LR | Grad Norm | Time | GPU Mem | Checkpoint |",
                    "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
                ]
                for rec in history:
                    md_lines.append(
                        f"| {rec['epoch']}/{num_epochs} | {rec['train_loss']:.4f} | {rec['val_loss']:.4f} | "
                        f"{rec['specialist_accuracy']:.2%} | {rec['severity_accuracy']:.2%} | {rec['joint_accuracy']:.2%} | "
                        f"{rec['specialist_macro_f1']:.4f} | {rec['severity_macro_f1']:.4f} | {rec['macro_f1']:.4f} | "
                        f"{rec['weighted_f1']:.4f} | {rec['learning_rate']:.2e} | {rec['grad_norm']} | "
                        f"{rec['epoch_duration_s']:.1f}s | {rec['gpu_memory_mb']:.0f} MB | {rec['checkpoint_status']} |"
                    )
                md_path.write_text("\n".join(md_lines), encoding="utf-8")
            except Exception as e:
                logger.warning("[TRAINER] Failed to export publication metric artifacts: %s", e)

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, epoch_logs)
                if isinstance(cb, EarlyStopping) and cb.should_stop:
                    logger.info("[TRAIN] Early stopping triggered at epoch %d", epoch)
                    break

        for cb in self.callbacks:
            cb.on_train_end()

        total_duration = time.monotonic() - train_start
        logger.info("[TRAIN] ══════════════════════════════════════════")
        logger.info("[TRAIN] Training complete. Best eval metric: %.4f | "
                     "Total time: %.2fs | Global steps: %d",
                     self.best_metric, total_duration, self.global_step)
        logger.info("[TRAIN] EXIT train()")
        logger.info("[TRAIN] ══════════════════════════════════════════")
        return {"best_eval_metric": self.best_metric, "global_step": self.global_step}

    def validate(self, dataloader: DataLoader | None = None) -> dict[str, Any]:
        """Evaluate model on dataloader."""
        dl = dataloader or self.eval_dataloader
        if not dl:
            raise ValueError("No dataloader provided for evaluation.")

        logger.debug("[VALIDATE] ENTER — dataloader len=%d", len(dl))
        self.model.eval()
        all_triage_logits = []
        all_triage_labels = []
        all_dept_logits = []
        all_dept_labels = []
        use_amp = getattr(self.cfg, "use_amp", True) and torch.cuda.is_available()

        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_step, batch in enumerate(dl):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                triage_targets, dept_targets = self._extract_batch_targets(batch)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    triage_logits, dept_logits = self._forward_model(
                        input_ids, attention_mask
                    )
                    v_loss, _ = self.loss_fn(triage_logits, triage_targets, dept_logits, dept_targets)
                    total_val_loss += v_loss.item()
                    val_steps += 1

                if triage_logits is not None and triage_targets is not None:
                    all_triage_logits.append(triage_logits.cpu().numpy())
                    all_triage_labels.append(triage_targets.cpu().numpy())
                if dept_logits is not None and dept_targets is not None:
                    all_dept_logits.append(dept_logits.cpu().numpy())
                    all_dept_labels.append(dept_targets.cpu().numpy())

        metrics = {"eval_loss": round(total_val_loss / max(val_steps, 1), 4)}
        if all_triage_logits and all_triage_labels:
            triage_logits_arr = np.concatenate(all_triage_logits, axis=0)
            triage_labels_arr = np.concatenate(all_triage_labels, axis=0)
            metrics = ClinicalMetricsCalculator.compute_all_metrics(
                triage_logits_arr, triage_labels_arr, prefix="eval", ignore_index=-1
            )

        if all_dept_logits and all_dept_labels:
            dept_logits_arr = np.concatenate(all_dept_logits, axis=0)
            dept_labels_arr = np.concatenate(all_dept_labels, axis=0)
            dept_metrics = ClinicalMetricsCalculator.compute_all_metrics(
                dept_logits_arr,
                dept_labels_arr,
                prefix="eval_specialist",
                ignore_index=-1,
            )
            metrics.update(dept_metrics)

            triage_f1 = metrics.get("eval_macro_f1", 0.0)
            spec_f1 = dept_metrics.get("eval_specialist_macro_f1", 0.0)
            if spec_f1 > 0.0:
                if triage_f1 > 0.0:
                    metrics["eval_macro_f1"] = round((triage_f1 + spec_f1) / 2.0, 4)
                else:
                    metrics["eval_macro_f1"] = spec_f1

        if all_triage_logits and all_triage_labels and all_dept_logits and all_dept_labels:
            triage_preds_arr = np.argmax(triage_logits_arr, axis=1)
            dept_preds_arr = np.argmax(dept_logits_arr, axis=1)
            valid_both_mask = (triage_labels_arr != -1) & (dept_labels_arr != -1)
            if valid_both_mask.any():
                joint_hits = (triage_preds_arr[valid_both_mask] == triage_labels_arr[valid_both_mask]) & (
                    dept_preds_arr[valid_both_mask] == dept_labels_arr[valid_both_mask]
                )
                metrics["eval_joint_accuracy"] = round(float(np.mean(joint_hits)), 4)
            else:
                metrics["eval_joint_accuracy"] = 0.0

        logger.debug("[VALIDATE] EXIT — computed %d metric keys", len(metrics))
        return metrics


    def test(self, test_dataloader: DataLoader) -> dict[str, Any]:
        """Evaluate model on test dataset."""
        logger.info("[TEST] Evaluating on test dataloader...")
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
        logger.info("[RESUME] Resuming training from checkpoint: %s", checkpoint_path)
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
            "[RESUME] Resumed state at epoch %d, step %d", self.current_epoch, self.global_step
        )
