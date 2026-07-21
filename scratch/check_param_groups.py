import os
import sys
from pathlib import Path
repo_root = Path(os.getcwd()).resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from models.emergent_path_triage.model import EmergentPathTriageConfig, EmergentPathTriageModel

checkpoint_path = Path("results/baseline_campaign/best_model.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("Saved optimizer state dict param groups:")
for i, group in enumerate(checkpoint["optimizer_state_dict"]["param_groups"]):
    print(f"  Group {i}: lr={group.get('lr')}, weight_decay={group.get('weight_decay')}, len_params={len(group.get('params', []))}")

triage_config = EmergentPathTriageConfig(latent_dim=8)
model_meta = EmergentPathTriageModel()
model = model_meta.build(None, triage_config=triage_config)

print("\nModel parameters requires_grad:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: requires_grad={param.requires_grad}")
