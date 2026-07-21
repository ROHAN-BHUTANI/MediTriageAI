import json
import sys
import os
from pathlib import Path

# Load notebook
notebook_path = Path("EPATH_CO_REASON_Training.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find specific cells to execute
cells_to_run = {}
for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "CENTRAL EXPERIMENT CONFIGURATION BLOCK" in source:
            cells_to_run["Cell 3"] = source
        elif "SECTION 1 & 3: REPOSITORY DETECT & IMPORT CHECKS" in source:
            cells_to_run["Cell 10"] = source
        elif "SECTION 3: DRIVE MOUNT & FOLDERS VERIFICATION" in source:
            cells_to_run["Cell 14"] = source
        elif "SECTION 8: EVALUATION METRICS" in source:
            cells_to_run["Section 8"] = source
        elif "SECTION 9: FINAL EXPERIMENT SUMMARY" in source:
            cells_to_run["Section 9"] = source
        elif "SECTION 10.1: FULL DTYPE TRACE & CONTRACT VALIDATION" in source:
            cells_to_run["Section 10.1"] = source
        elif "SECTION 10.2: TRAINING DRY RUN" in source:
            cells_to_run["Section 10.2"] = source
        elif "SECTION 10.3: RECOVERY DRY RUN" in source:
            cells_to_run["Section 10.3"] = source

print(f"Discovered {len(cells_to_run)} cells to execute: {list(cells_to_run.keys())}")

# Execute in order in a clean local dict
local_scope = {}
local_scope["__file__"] = str(Path("scratch/test_recovery_sequence.py").resolve())

def clean_code(code_str):
    lines = code_str.split("\n")
    cleaned = []
    for line in lines:
        if line.strip().startswith("!"):
            print(f"Mock Shell Command: {line}")
            if "git clone" in line:
                cleaned.append("# Mock git clone")
            elif "pip install" in line:
                cleaned.append("# Mock pip install")
            else:
                cleaned.append("pass")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)

# Run Cell 3
print("\nRunning Cell 3 (Central Config)...")
exec(clean_code(cells_to_run["Cell 3"]), local_scope)

# Optimize for fast CPU test
local_scope["EXPERIMENT_CONFIG"]["max_samples"] = 50

# Run Cell 10
print("\nRunning Cell 10 (Repo Setup)...")
exec(clean_code(cells_to_run["Cell 10"]), local_scope)

# Run Cell 14
print("\nRunning Cell 14 (Drive Setup)...")
exec(clean_code(cells_to_run["Cell 14"]), local_scope)

# Run Section 8
print("\nRunning Section 8 (Evaluation)...")
exp_name = local_scope["EXPERIMENT_CONFIG"]["experiment_name"]
ckpt_dir = local_scope["dirs"]["checkpoints"]
ckpt_dir.mkdir(parents=True, exist_ok=True)

# Copy the baseline checkpoint to experiments checkpoints folder to simulate recovery from drive/local experiments folder
src_best = Path("results/baseline_campaign/best_model.pt")
src_latest = Path("results/baseline_campaign/latest_model.pt")

if src_best.exists() and not (ckpt_dir / "best_model.pt").exists():
    import shutil
    print(f"Copying checkpoint {src_best} to {ckpt_dir}...")
    shutil.copy(src_best, ckpt_dir / "best_model.pt")
    shutil.copy(src_latest, ckpt_dir / "latest_model.pt")

# Write a dummy configuration.json so loader has it
config_json = local_scope["dirs"]["logs"] / "configuration.json"
if not config_json.exists():
    with open(config_json, "w") as f:
        json.dump(local_scope["EXPERIMENT_CONFIG"], f, indent=4)

# Execute Section 8 with a single namespace dict to ensure scoping works perfectly
exec(clean_code(cells_to_run["Section 8"]), local_scope)

# Run Section 9
print("\nRunning Section 9 (Summary Report)...")
exec(clean_code(cells_to_run["Section 9"]), local_scope)

# Run Section 10.1
print("\nRunning Section 10.1 (DType Trace & Contract Validation)...")
exec(clean_code(cells_to_run["Section 10.1"]), local_scope)

# Run Section 10.2
print("\nRunning Section 10.2 (Training Dry Run)...")
exec(clean_code(cells_to_run["Section 10.2"]), local_scope)

# Run Section 10.3
print("\nRunning Section 10.3 (Recovery Dry Run)...")
exec(clean_code(cells_to_run["Section 10.3"]), local_scope)

print("\nRecovery execution sequence validation PASSED successfully!")
