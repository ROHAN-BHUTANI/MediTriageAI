"""Diagnostic script to investigate E-PATH-CO-REASON training abnormalities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

# Insert project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.emergent_path_triage.model import EmergentPathTriageConfig, EmergentPathTriageModel
from src.data_pipeline import (
    EmergentPathDataConfig,
    EmergentTriageDataset,
    TokenizerPipeline,
    get_dataloader,
    get_leakage_safe_splits,
    LabelValidator,
)


def run_diagnostics() -> None:
    print("Running baseline diagnostics and failure analysis...")
    
    # Setup folders
    checkpoint_dir = "./results/baseline_campaign"
    out_dir = Path("./results/diagnostics_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    data_config = EmergentPathDataConfig()
    df = pd.read_csv(data_config.dataset_path)
    if df["text"].isna().sum() > 0:
        df = df.dropna(subset=["text"])
        
    # Keep subset of 200 for fast diagnostics
    df = df.sample(200, random_state=1337)
    
    train_df, val_df, test_df = get_leakage_safe_splits(
        df,
        train_ratio=data_config.train_ratio,
        val_ratio=data_config.val_ratio,
        test_ratio=data_config.test_ratio,
        seed=1337,
        stratify=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    pipeline = TokenizerPipeline(tokenizer, max_length=data_config.max_length)
    validator = LabelValidator()
    
    def process_df(target_df: pd.DataFrame) -> EmergentTriageDataset:
        texts = target_df["text"].tolist()
        spec_ids = [validator.validate_specialist(str(c)) for c in target_df["department_code"]]
        sev_ids = [validator.validate_severity(str(l)) for l in target_df["severity_heuristic"]]
        return EmergentTriageDataset(texts, spec_ids, sev_ids, pipeline)

    test_ds = process_df(test_df)
    test_loader = get_dataloader(test_ds, batch_size=data_config.batch_size, shuffle=False)
    
    # 2. Build model and load best parameters
    class TinyConfig:
        hidden_size: int = 16
        num_hidden_layers: int = 1
        num_attention_heads: int = 2
        intermediate_size: int = 32
        max_position_embeddings: int = 32
        
    config = EmergentPathTriageConfig(latent_dim=8)
    model_meta = EmergentPathTriageModel()
    model = model_meta.build(TinyConfig(), triage_config=config)
    
    best_ckpt_path = Path(checkpoint_dir) / "best_model.pt"
    if not best_ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint best_model.pt not found under {checkpoint_dir}")
        
    checkpoint = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 3. Investigation 1: Severity Prediction Failure & Logits statistics
    print("Running Investigation 1 (Logits and Class Frequencies)...")
    all_spec_logits = []
    all_sev_logits = []
    all_spec_labels = []
    all_sev_labels = []
    all_spec_preds = []
    all_sev_preds = []
    
    # Extract latent representations
    latent_before_dces = []
    latent_after_engine = []
    latent_after_dcp = []
    
    # Routing statistics lists
    all_routing_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels_spec = batch["labels_specialist"]
            labels_sev = batch["labels_severity"]
            
            # Forward pass hooks to collect latent variables
            # 1. Before DCES (encoder pooler CLS)
            encoder_output = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_out = encoder_output.last_hidden_state[:, 0, :]
            latent_before_dces.append(cls_out.cpu().numpy())
            
            # 2. Run model forward
            outputs = model(input_ids, attention_mask)
            
            all_spec_logits.append(outputs.specialist_logits.numpy())
            all_sev_logits.append(outputs.severity_logits.numpy())
            
            spec_preds = outputs.specialist_logits.argmax(dim=-1).numpy()
            sev_preds = outputs.severity_logits.argmax(dim=-1).numpy()
            
            all_spec_labels.extend(labels_spec.numpy())
            all_sev_labels.extend(labels_sev.numpy())
            all_spec_preds.extend(spec_preds)
            all_sev_preds.extend(sev_preds)
            
            # 3. After execution engine
            latent_after_engine.append(model._last_final_state.cpu().numpy())
            
            # DCP projection state
            # Reasoning state mapped to DCP urgency space
            proj_h = model.dcp.reasoning_proj(model._last_final_state)
            latent_after_dcp.append(proj_h.cpu().numpy())
            
            # Routing probabilities
            probs = model._last_routing_decision.routing_probabilities.cpu().numpy()
            all_routing_probs.append(probs)

    spec_logits = np.concatenate(all_spec_logits, axis=0)
    sev_logits = np.concatenate(all_sev_logits, axis=0)
    
    # Stats on severity logits
    sev_logits_mean = float(np.mean(sev_logits))
    sev_logits_std = float(np.std(sev_logits))
    sev_logits_min = float(np.min(sev_logits))
    sev_logits_max = float(np.max(sev_logits))
    
    # Severity Prediction Distributions
    unique_sev_preds, counts_sev_preds = np.unique(all_sev_preds, return_counts=True)
    sev_pred_freq = {validator.severity_labels[int(k)]: int(v) for k, v in zip(unique_sev_preds, counts_sev_preds)}
    
    unique_sev_labels, counts_sev_labels = np.unique(all_sev_labels, return_counts=True)
    sev_label_freq = {validator.severity_labels[int(k)]: int(v) for k, v in zip(unique_sev_labels, counts_sev_labels)}
    
    # Entropy of outputs
    # Apply softmax first
    sev_probs = torch.softmax(torch.tensor(sev_logits), dim=-1).numpy()
    sev_entropy = float(-np.sum(sev_probs * np.log(sev_probs + 1e-9), axis=-1).mean())
    
    spec_probs = torch.softmax(torch.tensor(spec_logits), dim=-1).numpy()
    spec_entropy = float(-np.sum(spec_probs * np.log(spec_probs + 1e-9), axis=-1).mean())

    # 4. Investigation 2: Routing Collapse Analysis
    print("Running Investigation 2 (Routing Analysis)...")
    routing_probs = np.concatenate(all_routing_probs, axis=0) # [B, M, N]
    B, M, N = routing_probs.shape
    epsilon = 1e-9
    entropies = -np.sum(routing_probs * np.log(routing_probs + epsilon), axis=-1)
    mean_routing_entropy = float(entropies.mean())
    routing_entropy_per_step = entropies.mean(axis=0).tolist()
    
    # CTB Utilizations
    routing_argmax = routing_probs.argmax(axis=-1)
    utilization_counts = [np.bincount(routing_argmax[:, s], minlength=N).tolist() for s in range(M)]

    # 5. Investigation 3: Gradient Norms Measuring
    print("Running Investigation 3 (Gradient Flow)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    # Run a dummy backward pass to compute gradients
    dummy_input = torch.zeros((2, data_config.max_length), dtype=torch.long)
    dummy_mask = torch.ones((2, data_config.max_length), dtype=torch.long)
    dummy_labels_spec = torch.zeros(2, dtype=torch.long)
    dummy_labels_sev = torch.zeros(2, dtype=torch.long)
    
    outputs = model(dummy_input, dummy_mask)
    from models.emergent_path_triage.hooks import apply_loss_hook
    from src.model import JointLoss
    loss_fn = JointLoss()
    loss_dict = apply_loss_hook(
        model, outputs.specialist_logits, outputs.severity_logits,
        dummy_labels_spec, dummy_labels_sev, loss_fn
    )
    loss_dict["joint_loss"].backward()
    
    # Calculate gradient norms for modules
    def get_grad_norm(parameters) -> float:
        grads = [p.grad.data.norm(2).item() for p in parameters if p.grad is not None]
        return float(np.sqrt(sum(g ** 2 for g in grads))) if grads else 0.0

    grad_norms = {
        "encoder": get_grad_norm(model.encoder.parameters()),
        "dces": get_grad_norm(model.dces.parameters()),
        "router": get_grad_norm(model.router.parameters()),
        "ctbs": get_grad_norm(model.blocks.parameters()),
        "classifier_specialist": get_grad_norm(model.classifier_specialist.parameters()),
        "classifier_severity": get_grad_norm(model.classifier_severity.parameters()),
        "dcp": get_grad_norm(model.dcp.parameters())
    }

    # 6. Investigation 4: Loss behaviour plot
    print("Running Investigation 4 (Loss Breakdown)...")
    history_df = pd.read_csv(Path(checkpoint_dir) / "training_history.csv")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Total Joint Loss", marker="o")
    plt.plot(history_df["epoch"], history_df["train_specialist_loss"], label="Specialist Loss", marker="x")
    plt.plot(history_df["epoch"], history_df["train_severity_loss"], label="Severity Loss", marker="s")
    plt.plot(history_df["epoch"], history_df["train_cons_loss"], label="Consistency Loss", marker="d")
    plt.plot(history_df["epoch"], history_df["train_div_loss"], label="Diversity Loss (Entropy)", marker="v")
    plt.plot(history_df["epoch"], history_df["train_ortho_loss"], label="Orthogonality Loss", marker="^")
    plt.title("E-PATH-CO-REASON Loss Components per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Magnitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_breakdown.png")
    plt.close()

    # 7. Investigation 5: Prediction Confidence Histogram
    print("Running Investigation 5 (Confidence Histograms)...")
    spec_conf = np.max(spec_probs, axis=-1)
    sev_conf = np.max(sev_probs, axis=-1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(spec_conf, bins=10, color="skyblue", edgecolor="black")
    plt.title("Specialist Confidence Histogram")
    plt.xlabel("Max Prob")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 2, 2)
    plt.hist(sev_conf, bins=10, color="lightcoral", edgecolor="black")
    plt.title("Severity Confidence Histogram")
    plt.xlabel("Max Prob")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_confidence_hist.png")
    plt.close()

    # 8. Investigation 6: Representation Analysis PCA & t-SNE
    print("Running Investigation 6 (PCA & t-SNE Visualizations)...")
    latent_before = np.concatenate(latent_before_dces, axis=0) # [B, encoder_dim]
    latent_engine = np.concatenate(latent_after_engine, axis=0) # [B, latent_dim]
    latent_dcp = np.concatenate(latent_after_dcp, axis=0) # [B, urgency_dim]
    
    # Check dimensions
    print(f"Shape before DCES: {latent_before.shape}, After Engine: {latent_engine.shape}, After DCP: {latent_dcp.shape}")
    
    # We apply PCA to project down to 2D
    pca_before = PCA(n_components=2).fit_transform(latent_before)
    pca_engine = PCA(n_components=2).fit_transform(latent_engine)
    pca_dcp = PCA(n_components=2).fit_transform(latent_dcp)
    
    # We also apply t-SNE (with perplexity adjusted for small sample sizes)
    perp = min(5, len(df) - 1)
    tsne_before = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(latent_before)
    tsne_engine = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(latent_engine)
    tsne_dcp = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(latent_dcp)
    
    # Plot PCA side by side
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(pca_before[:, 0], pca_before[:, 1], c=all_spec_labels, cmap="tab10", alpha=0.8)
    plt.title("Before DCES (PCA)")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.scatter(pca_engine[:, 0], pca_engine[:, 1], c=all_spec_labels, cmap="tab10", alpha=0.8)
    plt.title("After Engine (PCA)")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.scatter(pca_dcp[:, 0], pca_dcp[:, 1], c=all_spec_labels, cmap="tab10", alpha=0.8)
    plt.title("After DCP (PCA)")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(out_dir / "representation_pca.png")
    plt.close()

    # Plot t-SNE side by side
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(tsne_before[:, 0], tsne_before[:, 1], c=all_spec_labels, cmap="tab10", alpha=0.8)
    plt.title("Before DCES (t-SNE)")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.scatter(tsne_engine[:, 0], tsne_engine[:, 1], c=all_spec_labels, cmap="tab10", alpha=0.8)
    plt.title("After Engine (t-SNE)")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.scatter(tsne_dcp[:, 0], tsne_dcp[:, 1], c=all_spec_labels, cmap="tab10", alpha=0.8)
    plt.title("After DCP (t-SNE)")
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(out_dir / "representation_tsne.png")
    plt.close()

    # 9. Save Diagnostic Summary JSON
    summary_json = {
        "severity_logits": {
            "mean": sev_logits_mean,
            "std": sev_logits_std,
            "min": sev_logits_min,
            "max": sev_logits_max
        },
        "entropy": {
            "specialist_entropy": spec_entropy,
            "severity_entropy": sev_entropy
        },
        "predictions_frequency": {
            "specialist_predicted_counts": pd.Series(all_spec_preds).value_counts().to_dict(),
            "severity_predicted_counts": sev_pred_freq,
            "severity_label_counts": sev_label_freq
        },
        "routing": {
            "routing_entropy": mean_routing_entropy,
            "routing_entropy_per_step": routing_entropy_per_step,
            "ctb_utilizations_per_step": utilization_counts
        },
        "gradients": grad_norms
    }
    
    with open(out_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=4)
        
    print(f"Diagnostics complete! Assets successfully exported to {out_dir}.")


if __name__ == "__main__":
    run_diagnostics()
