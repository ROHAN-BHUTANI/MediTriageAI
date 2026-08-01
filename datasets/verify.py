"""MediTriageAI Dataset Verification CLI.

Validates that every dataset registered in ADAPTER_REGISTRY is present,
properly formatted, and able to emit valid records. Exits with status code 1
if any active dataset is missing or unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path first
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from meditriage.builder.orchestrator import ADAPTER_REGISTRY
from meditriage.builder.config import Config

# Import helper from bootstrap
from datasets.bootstrap import get_expected_file, RAW


def log(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def verify_datasets() -> bool:
    """Verify all datasets in config.active_datasets.

    Returns:
        True if all active datasets pass verification, False otherwise.
    """
    config = Config.from_yaml(str(PROJECT_ROOT / "config" / "dataset_config.yaml"))
    
    log("\n========================================================")
    log("MediTriageAI Dataset Pre-Flight Verification Audit")
    log("========================================================\n")

    passed_all = True
    results = []

    for name in config.active_datasets:
        if name not in ADAPTER_REGISTRY:
            log(f"ERROR: Adapter {name} missing from ADAPTER_REGISTRY!")
            passed_all = False
            continue

        adapter_cls = ADAPTER_REGISTRY[name]
        adapter = adapter_cls()
        raw_dir = RAW / name
        exp_file = get_expected_file(name)

        file_exists = exp_file.exists()
        sample_rows = 0
        status = "FAILED"
        err_msg = ""

        if not file_exists:
            status = "MISSING_FILE"
            err_msg = f"Expected file {exp_file.name} not found."
            passed_all = False
        else:
            try:
                gen = adapter.ingest(str(raw_dir))
                first_chunk = next(gen, None)
                if first_chunk is not None:
                    sample_rows = len(first_chunk)
                    if sample_rows > 0:
                        status = "PASSED"
                    else:
                        status = "ZERO_ROWS"
                        err_msg = "Adapter chunk contains 0 rows."
                        passed_all = False
                else:
                    status = "ZERO_CHUNKS"
                    err_msg = "Adapter returned empty generator."
                    passed_all = False
            except Exception as e:
                status = "ERROR"
                err_msg = str(e)
                passed_all = False

        results.append({
            "Dataset": name,
            "Expected File": exp_file.name,
            "File Exists": "YES" if file_exists else "NO",
            "Sample Rows": sample_rows,
            "Status": status,
            "Error": err_msg,
        })

    df_results = pd.DataFrame(results)
    log(df_results.to_string(index=False))
    log("\n========================================================")

    if passed_all:
        log("RESULT: ALL DATASETS PASSED PRE-FLIGHT VERIFICATION [OK]")
        log("========================================================\n")
        return True
    else:
        log("RESULT: DATASET VERIFICATION FAILED [CRITICAL]")
        log("========================================================\n")
        return False


if __name__ == "__main__":
    success = verify_datasets()
    if not success:
        sys.exit(1)
    sys.exit(0)
