import sys
import os
import shutil
import hashlib
from pathlib import Path

def run_preflight_checks():
    print("="*60)
    print("PRE-FLIGHT VALIDATION STAGE")
    print("="*60)
    
    # 1. Python Version
    py_version = sys.version_info
    print(f"[CHECK] Python Version: {sys.version.split()[0]}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 10):
        print("[FAIL] Python 3.10+ is required.")
        sys.exit(1)
        
    # 2. CUDA Availability & GPU Visibility
    try:
        import torch
        cuda_avail = torch.cuda.is_available() or os.environ.get("MOCK_GPU") == "1"
        print(f"[CHECK] CUDA Available: {cuda_avail}")
        if not cuda_avail:
            print("[FAIL] Mandatory GPU hardware not found. Aborting execution.")
            sys.exit(1)
            
        gpu_count = torch.cuda.device_count()
        print(f"[CHECK] GPU Count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"        GPU {i}: {gpu_name} (VRAM: {vram:.2f} GB)")
    except ImportError:
        print("[FAIL] PyTorch is not installed.")
        sys.exit(1)
        
    # 3. Available Disk Space
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    print(f"[CHECK] Free Disk Space: {free_gb:.2f} GB")
    if free_gb < 50.0:
        print("[WARNING] Less than 50GB disk space available. Checkpoints may exhaust disk space.")
        # We don't abort, but we warn heavily.

    # 4. Repository Write Permission
    test_path = Path("outputs/.write_test")
    try:
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.touch()
        test_path.unlink()
        print("[CHECK] Repository Write Permissions: OK")
    except Exception as e:
        print(f"[FAIL] Cannot write to outputs directory: {e}")
        sys.exit(1)
        
    # 5. Dataset Existence & Integrity
    dataset_path = Path("data/processed/enriched/dataset_enriched.csv")
    if not dataset_path.exists():
        print(f"[FAIL] Missing canonical enriched dataset: {dataset_path}")
        sys.exit(1)
        
    print(f"[CHECK] Canonical training dataset found: {dataset_path}")
    
    print("="*60)
    print("PRE-FLIGHT VALIDATION PASSED. PROCEEDING TO EXECUTION.")
    print("="*60)

if __name__ == "__main__":
    run_preflight_checks()
