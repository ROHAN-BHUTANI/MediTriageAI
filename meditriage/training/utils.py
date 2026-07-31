"""Training Utilities for Environment, Hardware, and Dataset Provenance."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from typing import Any

import pandas as pd
import torch


def get_git_commit_hash() -> str:
    """Get the current Git commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return commit.decode("utf-8").strip()
    except Exception:
        return "unknown_commit"


def get_hardware_info() -> dict[str, Any]:
    """Retrieve detailed hardware, OS, and framework environment metrics."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else "N/A",
        "gpu_model": gpu_name,
        "gpu_count": gpu_count,
        "os_platform": platform.platform(),
        "processor": platform.processor(),
        "system": platform.system(),
    }


def compute_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute a SHA256 deterministic fingerprint of a DataFrame."""
    if df.empty:
        return "empty_dataset"
    content = "".join(df.astype(str).values.flatten())
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
