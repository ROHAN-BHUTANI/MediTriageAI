"""Standalone audit execution script for E-PATH-CO-REASON ReasoningPathExecutionEngine."""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, XLMRobertaConfig

# Resolve repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.emergent_path_triage.model import EmergentPathTriageModel, EmergentPathTriageConfig
from src.data_pipeline import (
    get_leakage_safe_splits,
    TokenizerPipeline,
    EmergentTriageDataset,
    get_dataloader,
    LabelValidator,
)

def run_audit(checkpoint_path: str, dataset_csv: str, export_dir: str):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at: {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract state dict
    model_state = checkpoint.get("model_state_dict", checkpoint)
    
    # 1. Determine encoder size and layers from state dict
    hidden_size = 768
    num_hidden_layers = 12
    vocab_size = 250002
    
    # Check word embeddings size
    for k, v in model_state.items():
        if "word_embeddings.weight" in k:
            hidden_size = v.shape[1]
            vocab_size = v.shape[0]
            break
            
    # Count encoder layers
    layer_indices = set()
    for k in model_state.keys():
        if "encoder.layer." in k or "encoder.encoder.layer." in k:
            parts = k.split(".")
            for part in parts:
                if part.isdigit():
                    layer_indices.add(int(part))
                    break
    if layer_indices:
        num_hidden_layers = max(layer_indices) + 1
    else:
        num_hidden_layers = 1
        
    print(f"Reconstructed encoder: hidden_size={hidden_size}, num_hidden_layers={num_hidden_layers}, vocab_size={vocab_size}")
    
    # 2. Reconstruct latent dimension
    latent_dim = 128
    for k, v in model_state.items():
        if "classifier_specialist.fc1.weight" in k:
            latent_dim = v.shape[1]
            break
    print(f"Reconstructed latent_dim: {latent_dim}")
    
    triage_config = EmergentPathTriageConfig(latent_dim=latent_dim)
    model_meta = EmergentPathTriageModel()
    
    model_config = XLMRobertaConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2 if hidden_size < 100 else 12,
        intermediate_size=hidden_size * 2 if hidden_size < 100 else hidden_size * 4,
        max_position_embeddings=512,
        vocab_size=vocab_size
    )
    
    model = model_meta.build(model_config, triage_config=triage_config)
    
    # Load state dict
    model.load_state_dict(model_state, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 3. Load a single batch
    dataset_csv = Path(dataset_csv)
    batch = None
    if dataset_csv.exists():
        print(f"Loading data from {dataset_csv}...")
        try:
            df = pd.read_csv(dataset_csv)
            df = df.dropna(subset=["text"])
            _, val_df, _ = get_leakage_safe_splits(df, seed=1337, stratify=False)
            if len(val_df) == 0:
                val_df = df
            
            tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            pipeline = TokenizerPipeline(tokenizer, max_length=64)
            validator = LabelValidator()
            
            sample_df = val_df.head(2)
            texts = sample_df["text"].tolist()
            spec_ids = [validator.validate_specialist(str(c)) for c in sample_df["department_code"]]
            sev_ids = [validator.validate_severity(str(l)) for l in sample_df["severity_heuristic"]]
            
            dataset = EmergentTriageDataset(texts, spec_ids, sev_ids, pipeline)
            loader = get_dataloader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(loader))
        except Exception as e:
            print(f"Warning: Could not process dataset {dataset_csv}: {e}. Falling back to mock batch.")
            
    if batch is None:
        print("Dataset not found or failed to load. Constructing a mock batch of shape (2, 32).")
        input_ids = torch.randint(0, vocab_size, (2, 32), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels_specialist = torch.zeros(2, dtype=torch.long, device=device)
        labels_severity = torch.zeros(2, dtype=torch.long, device=device)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels_specialist": labels_specialist,
            "labels_severity": labels_severity
        }
        
    # 4. Perform the audit pass
    print("Running execution engine audit pass...")
    # Initialize auditor explicitly
    if not hasattr(model.engine, "auditor") or model.engine.auditor is None:
        from models.emergent_path_triage.hooks import ExecutionEngineAuditor
        model.engine.auditor = ExecutionEngineAuditor(model.engine)
        
    model.engine.reset_audit()
    
    # Standard forward and backward pass with gradients enabled
    model.zero_grad()
    with torch.enable_grad():
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_spec = batch["labels_specialist"].to(device)
        labels_sev = batch["labels_severity"].to(device)
        
        outputs = model(input_ids, attention_mask)
        
        from models.emergent_path_triage.hooks import apply_loss_hook
        from src.model import JointLoss
        loss_fn = JointLoss()
        loss_dict = apply_loss_hook(
            model,
            outputs.specialist_logits,
            outputs.severity_logits,
            labels_spec,
            labels_sev,
            loss_fn
        )
        loss = loss_dict["joint_loss"]
        
        # Backward timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        import time
        t_start = time.perf_counter()
        
        loss.backward()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        backward_time = time.perf_counter() - t_start
        
    # Export stats
    print(f"Finalizing audit. Exporting logs to: {export_dir}")
    model.engine.finalize_and_export_audit(
        model=model,
        last_batch=batch,
        device=device,
        use_amp=False,
        checkpoint_dir=export_dir
    )
    
    # Print metrics report paths
    export_path = Path(export_dir)
    print("Audit artifacts generated successfully:")
    print(f"  - {export_path / 'execution_engine_audit.json'}")
    print(f"  - {export_path / 'execution_engine_gradients.json'}")
    print(f"  - {export_path / 'execution_engine_statistics.json'}")
    print(f"  - {export_path / 'execution_engine_activations.json'}")
    print(f"  - {export_path / 'execution_engine_memory.json'}")
    print(f"  - {export_path / 'execution_engine_timing.json'}")
    print(f"  - {export_path / 'execution_engine_summary.md'}")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Standalone audit runner for E-PATH-CO-REASON Execution Engine.")
    parser.add_argument("--checkpoint", type=str, default="results/baseline_campaign/best_model.pt", help="Path to checkpoint file.")
    parser.add_argument("--dataset", type=str, default="data/processed/enriched/dataset_enriched.csv", help="Path to processed dataset CSV.")
    parser.add_argument("--export-dir", type=str, default=".", help="Directory to export audit logs.")
    args = parser.parse_args()
    
    run_audit(args.checkpoint, args.dataset, args.export_dir)

if __name__ == "__main__":
    main()
