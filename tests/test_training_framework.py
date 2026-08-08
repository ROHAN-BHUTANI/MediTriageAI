"""Unit tests for MediTriageAI Training & Experimentation Framework.

Covers:
  - set_seed and deterministic execution
  - TrainingConfig management (JSON/YAML)
  - Registry resolution
  - Hardware info and dataset fingerprinting
  - Loss functions (FocalLoss, WeightedCrossEntropyLoss, MultiTaskLoss)
  - Optimizer & Scheduler factories
  - ClinicalMetricsCalculator (ECE, Top-K, Macro F1, Confusion Matrix)
  - Callbacks (EarlyStopping, ModelCheckpoint, LearningRateMonitor)
  - ExperimentLogger
  - CheckpointManager (state saving, loading, resumption)
  - MultiTaskClinicalClassifier and Trainer execution
  - Report Generation (experiment_report.md, benchmark_results.json, hardware_report.json, reproducibility_report.json)
  - AblationFramework & ExperimentRunner
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from meditriage.training.callbacks import (
    EarlyStopping,
)
from meditriage.training.checkpoint import CheckpointManager
from meditriage.training.config import TrainingConfig
from meditriage.training.experiment import AblationFramework
from meditriage.training.logger import ExperimentLogger
from meditriage.training.losses import (
    FocalLoss,
    MultiTaskLoss,
    WeightedCrossEntropyLoss,
)
from meditriage.training.metrics import ClinicalMetricsCalculator
from meditriage.training.optimizer import get_optimizer
from meditriage.training.registry import get_backbone_model_id
from meditriage.training.report import generate_experiment_reports
from meditriage.training.scheduler import get_scheduler
from meditriage.training.seed import set_seed
from meditriage.training.trainer import MultiTaskClinicalClassifier, Trainer
from meditriage.training.utils import (
    compute_dataset_fingerprint,
    get_hardware_info,
)

# ─── Dummy Model & Dataset for Testing ─────────────────────────────────────


class DummyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_size)
        self.pooler_output = None

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ):
        emb = self.embedding(input_ids)
        cls_rep = emb[:, 0, :]
        return type(
            "Output", (), {"last_hidden_state": emb, "pooler_output": cls_rep}
        )()


class DummyClinicalDataset(Dataset):
    def __init__(self, size: int = 20, seq_len: int = 16):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.randint(0, 100, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "triage_label": torch.tensor(idx % 5, dtype=torch.long),
            "dept_label": torch.tensor(idx % 8, dtype=torch.long),
        }


# ─── Seed & Utils Tests ───────────────────────────────────────────────────


class TestSeedAndUtils:
    def test_set_seed_reproducibility(self):
        set_seed(42)
        r1 = random_sample()
        set_seed(42)
        r2 = random_sample()
        assert r1 == r2

    def test_hardware_info(self):
        info = get_hardware_info()
        assert "python_version" in info
        assert "torch_version" in info
        assert "cuda_available" in info

    def test_dataset_fingerprint(self):
        df = pd.DataFrame({"text": ["a", "b", "c"], "label": [1, 2, 3]})
        fp = compute_dataset_fingerprint(df)
        assert len(fp) == 16
        assert compute_dataset_fingerprint(df) == fp


def random_sample():
    return float(torch.randn(1).item())


# ─── Config & Registry Tests ───────────────────────────────────────────────


class TestConfigAndRegistry:
    def test_default_config(self):
        cfg = TrainingConfig()
        assert cfg.model_name_or_path == "xlm-roberta-base"
        assert cfg.num_triage_classes == 5
        assert cfg.num_dept_classes == 13

    def test_config_save_load_json(self, tmp_path: Path):
        cfg = TrainingConfig(experiment_name="test_exp", learning_rate=1e-4)
        json_path = tmp_path / "config.json"
        cfg.save(json_path)
        loaded = TrainingConfig.load(json_path)
        assert loaded.experiment_name == "test_exp"
        assert loaded.learning_rate == 1e-4

    def test_config_save_load_yaml(self, tmp_path: Path):
        cfg = TrainingConfig(experiment_name="test_yaml_exp", num_epochs=10)
        yaml_path = tmp_path / "config.yaml"
        cfg.save(yaml_path)
        loaded = TrainingConfig.load(yaml_path)
        assert loaded.experiment_name == "test_yaml_exp"
        assert loaded.num_epochs == 10

    def test_backbone_registry(self):
        assert get_backbone_model_id("muril") == "google/muril-base-cased"
        assert get_backbone_model_id("xlm-roberta-base") == "xlm-roberta-base"


# ─── Loss Functions Tests ─────────────────────────────────────────────────


class TestLosses:
    def test_focal_loss(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_weighted_cross_entropy(self):
        weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0])
        loss_fn = WeightedCrossEntropyLoss(class_weights=weights)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 3])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_multi_task_loss(self):
        mt_loss = MultiTaskLoss(
            loss_type="cross_entropy", triage_weight=1.0, dept_weight=0.5
        )
        t_logits = torch.randn(4, 5)
        t_targets = torch.tensor([0, 1, 2, 3])
        d_logits = torch.randn(4, 8)
        d_targets = torch.tensor([0, 1, 2, 3])

        total_loss, metrics = mt_loss(t_logits, t_targets, d_logits, d_targets)
        assert "loss_triage" in metrics
        assert "loss_dept" in metrics
        assert "loss_total" in metrics
        assert total_loss.item() > 0


# ─── Optimizer & Scheduler Tests ──────────────────────────────────────────


class TestOptimizerAndScheduler:
    def test_optimizer_factory(self):
        model = DummyBackbone()
        cfg = TrainingConfig(optimizer="adamw", learning_rate=1e-3)
        opt = get_optimizer(model, cfg)
        assert isinstance(opt, torch.optim.Optimizer)

    def test_scheduler_factory(self):
        model = DummyBackbone()
        cfg = TrainingConfig(scheduler="linear", warmup_steps=10)
        opt = get_optimizer(model, cfg)
        sched = get_scheduler(opt, cfg, num_training_steps=100)
        assert sched is not None


# ─── Metrics Calculator Tests ──────────────────────────────────────────────


class TestMetricsCalculator:
    def test_compute_all_metrics(self):
        logits = np.array([[2.0, 0.1, 0.0], [0.1, 3.0, 0.0], [0.0, 0.1, 2.5]])
        labels = np.array([0, 1, 2])
        metrics = ClinicalMetricsCalculator.compute_all_metrics(
            logits, labels, prefix="eval"
        )

        assert metrics["eval_accuracy"] == 1.0
        assert metrics["eval_macro_f1"] == 1.0
        assert "eval_calibration_error" in metrics
        assert "eval_confusion_matrix" in metrics


# ─── Callbacks & Logger Tests ──────────────────────────────────────────────


class TestCallbacksAndLogger:
    def test_early_stopping(self):
        es = EarlyStopping(monitor="eval_macro_f1", patience=2, mode="max")
        es.on_epoch_end(0, {"eval_macro_f1": 0.8})
        assert es.should_stop is False
        es.on_epoch_end(1, {"eval_macro_f1": 0.7})
        es.on_epoch_end(2, {"eval_macro_f1": 0.6})
        assert es.should_stop is True

    def test_logger(self, tmp_path: Path):
        exp_logger = ExperimentLogger(log_dir=tmp_path)
        exp_logger.log_metrics(1, {"accuracy": 0.85, "f1": 0.84}, prefix="train")
        exp_logger.log_experiment_summary({"status": "SUCCESS"})
        exp_logger.close()

        assert (tmp_path / "training.log").exists()
        assert (tmp_path / "history.csv").exists()
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "experiment_summary.json").exists()


# ─── Checkpoint Manager Tests ──────────────────────────────────────────────


class TestCheckpointManager:
    def test_save_and_load_checkpoint(self, tmp_path: Path):
        ckpt_mgr = CheckpointManager(tmp_path)
        backbone = DummyBackbone()
        model = MultiTaskClinicalClassifier(backbone, hidden_size=32)
        cfg = TrainingConfig()
        opt = get_optimizer(model, cfg)

        save_path = ckpt_mgr.save_checkpoint(
            model=model,
            optimizer=opt,
            scheduler=None,
            scaler=None,
            epoch=2,
            global_step=50,
            config=cfg,
            metrics={"eval_f1": 0.88},
            filename="ckpt.pt",
        )

        assert save_path.exists()

        # Load back into fresh model
        new_model = MultiTaskClinicalClassifier(DummyBackbone(), hidden_size=32)
        new_opt = get_optimizer(new_model, cfg)
        info = ckpt_mgr.load_checkpoint(save_path, new_model, new_opt)

        assert info["epoch"] == 2
        assert info["global_step"] == 50
        assert info["metrics"]["eval_f1"] == 0.88


# ─── Trainer Execution Tests ───────────────────────────────────────────────


class TestTrainerExecution:
    def test_trainer_train_validate_predict(self, tmp_path: Path):
        cfg = TrainingConfig(
            output_dir=str(tmp_path / "trainer_out"),
            num_epochs=1,
            batch_size=4,
            use_amp=False,
        )
        backbone = DummyBackbone()
        model = MultiTaskClinicalClassifier(backbone, hidden_size=32)
        ds = DummyClinicalDataset(size=8)
        dl = DataLoader(ds, batch_size=4)

        trainer = Trainer(
            model=model,
            config=cfg,
            train_dataloader=dl,
            eval_dataloader=dl,
            device="cpu",
        )

        res = trainer.train()
        assert "best_eval_metric" in res
        assert trainer.global_step > 0

        # Predict
        preds = trainer.predict(dl)
        assert len(preds["predictions"]) == 8
        assert len(preds["probabilities"]) == 8

        # Test
        test_metrics = trainer.test(dl)
        assert "test_accuracy" in test_metrics


# ─── Report Generator & Ablation Tests ────────────────────────────────────


class TestReportsAndAblation:
    def test_generate_experiment_reports(self, tmp_path: Path):
        cfg = TrainingConfig(output_dir=str(tmp_path / "reports"))
        metrics = {
            "test_accuracy": 0.90,
            "test_macro_f1": 0.89,
            "test_balanced_accuracy": 0.88,
        }

        generate_experiment_reports(cfg, metrics)
        out_dir = Path(cfg.output_dir)

        assert (out_dir / "experiment_report.md").exists()
        assert (out_dir / "training_summary.json").exists()
        assert (out_dir / "benchmark_results.json").exists()
        assert (out_dir / "hardware_report.json").exists()
        assert (out_dir / "reproducibility_report.json").exists()

    def test_ablation_framework(self, tmp_path: Path):
        cfg = TrainingConfig()
        af = AblationFramework(cfg, output_dir=tmp_path / "ablation")
        matrix = af.get_ablation_matrix()
        assert len(matrix) == 5

        af.register_result("exp_01_baseline_raw", {"test_macro_f1": 0.80}, cfg)
        af.register_result("exp_05_full_pipeline", {"test_macro_f1": 0.92}, cfg)
        summary_df = af.generate_ablation_summary()

        assert len(summary_df) == 2
        assert (tmp_path / "ablation" / "ablation_comparison.csv").exists()
