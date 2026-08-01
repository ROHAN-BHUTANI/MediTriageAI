"""MediTriageAI Dataset Bootstrap & Pre-Flight Audit Module.

Automated dataset acquisition, archive extraction, file layout verification,
and adapter readiness testing CLI for MediTriageAI.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from meditriage.builder.config import Config
from meditriage.builder.orchestrator import ADAPTER_REGISTRY

RAW = ROOT / "raw"
META = ROOT / "metadata"
LOGS = ROOT / "download_logs"

for d in [RAW, META, LOGS]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def compute_file_checksum(file_path: Path) -> str:
    """Compute SHA256 checksum of a single file."""
    import hashlib

    if not file_path.exists() or not file_path.is_file():
        return ""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract ZIP archive safely."""
    if not archive_path.exists():
        return False
    try:
        log(f"Extracting archive {archive_path.name} -> {extract_to}...")
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        log(f"Extracted {archive_path.name} successfully.")
        return True
    except Exception as e:
        log(f"Failed to extract archive {archive_path}: {e}")
        return False


def extract_all_archives() -> None:
    """Inspect raw directories and extract any compressed archives."""
    log("\n--- Checking and Extracting Data Archives ---")

    # 1. medqa_usmle: data_clean.zip
    medqa_zip = RAW / "medqa_usmle" / "data_clean.zip"
    medqa_target = RAW / "medqa_usmle" / "data_clean" / "data_clean" / "questions" / "US" / "US_qbank.jsonl"
    if medqa_zip.exists() and not medqa_target.exists():
        extract_archive(medqa_zip, RAW / "medqa_usmle" / "data_clean")

    # 2. nhamcs_ed: ed2019.zip, ed2020.zip, ed2021.zip
    for year in ["2019", "2020", "2021"]:
        zip_p = RAW / "nhamcs_ed" / f"ed{year}.zip"
        target_p = RAW / "nhamcs_ed" / f"ed{year}" / f"ed{year}"
        if zip_p.exists() and not target_p.exists():
            extract_archive(zip_p, RAW / "nhamcs_ed" / f"ed{year}")

    # 3. l3cube_code_mixed: code-mixed-nlp.zip
    l3_zip = RAW / "l3cube_code_mixed" / "code-mixed-nlp.zip"
    l3_target = RAW / "l3cube_code_mixed" / "code-mixed-nlp-main" / "L3Cube-HingLID" / "train.txt"
    if l3_zip.exists() and not l3_target.exists():
        extract_archive(l3_zip, RAW / "l3cube_code_mixed")


def get_expected_file(dataset_name: str) -> Path:
    """Infer expected primary file path for each adapter."""
    raw_dir = RAW / dataset_name
    if dataset_name == "pmc_patients":
        return raw_dir / "PMC-Patients.csv"
    elif dataset_name == "medqa_usmle":
        return raw_dir / "data_clean" / "data_clean" / "questions" / "US" / "US_qbank.jsonl"
    elif dataset_name == "neiss":
        return raw_dir / "neiss_all.parquet"
    elif dataset_name == "nhamcs_ed":
        return raw_dir / "ed2019" / "ed2019"
    elif dataset_name == "kaggle_medical_triage":
        return raw_dir / "medical_data.json" if (raw_dir / "medical_data.json").exists() else raw_dir / "triage.csv"
    elif dataset_name == "l3cube_code_mixed":
        return raw_dir / "code-mixed-nlp-main" / "L3Cube-HingLID" / "train.txt"
    elif dataset_name == "meddialog_en":
        p = list(raw_dir.rglob("*.parquet"))
        return p[0] if p else raw_dir / "dialog.jsonl"
    elif dataset_name == "mtsamples":
        return raw_dir / "mtsamples.csv" if (raw_dir / "mtsamples.csv").exists() else raw_dir / "train.jsonl"
    elif dataset_name == "symptom2disease":
        return raw_dir / "Symptom2Disease.csv"
    elif dataset_name == "chatdoctor_healthcaremagic":
        p = list(raw_dir.rglob("*.parquet"))
        return p[0] if p else raw_dir / "data" / "train.parquet"
    elif dataset_name == "chatdoctor_icliniq":
        p = list(raw_dir.rglob("*.parquet"))
        return p[0] if p else raw_dir / "data" / "train.parquet"
    elif dataset_name == "fedmml_ed_triage":
        return raw_dir / "fedmml_ed_triage_dataset.csv" if (raw_dir / "fedmml_ed_triage_dataset.csv").exists() else raw_dir / "data.csv"
    elif dataset_name == "medical_meadow_medqa":
        return raw_dir / "medical_meadow_medqa.json"
    return raw_dir


def bootstrap_and_audit():
    """Main bootstrap function."""
    log("\n========================================================")
    log("MediTriageAI Production Dataset Bootstrap & Verification")
    log("========================================================\n")

    # Step 1: Extract all archives first
    extract_all_archives()

    # Step 2: Download acquisition phase for missing datasets
    try:
        from datasets.download_hf import DATASET_SPECS, acquire_single_dataset
        log("Executing dataset download acquisition phase...")
        for spec in DATASET_SPECS:
            ds_name = spec[0]
            exp_p = get_expected_file(ds_name)
            if not exp_p.exists():
                acquire_single_dataset(spec)
    except Exception as e:
        log(f"Download acquisition failure: {e}")
        raise RuntimeError(f"Bootstrap failed during download acquisition: {e}")

    # Step 3: Re-verify extraction
    extract_all_archives()

    # Step 4: Audit all registered adapters
    log("\n--- Auditing Adapter Readiness & Generating Manifest ---")
    manifest = {}
    config = Config.from_yaml(str(PROJECT_ROOT / "config" / "dataset_config.yaml"))

    audit_summary = []

    for name in config.active_datasets:
        if name not in ADAPTER_REGISTRY:
            log(f"Warning: Adapter {name} not found in ADAPTER_REGISTRY!")
            continue

        adapter_cls = ADAPTER_REGISTRY[name]
        adapter = adapter_cls()
        raw_dir = RAW / name
        raw_dir.mkdir(parents=True, exist_ok=True)

        exp_file = get_expected_file(name)
        file_detected = exp_file.exists()
        
        # Test adapter ingestion via generator streaming
        emitted_rows = 0
        readiness = "NOT_READY"
        
        try:
            for chunk in adapter.ingest(str(raw_dir)):
                if chunk is not None and not chunk.empty:
                    emitted_rows += len(chunk)
            if emitted_rows > 0:
                readiness = "READY"
        except Exception as e:
            log(f"Ingestion error for {name}: {e}")

        size_bytes = exp_file.stat().st_size if file_detected and exp_file.is_file() else 0
        chksum = compute_file_checksum(exp_file) if file_detected and exp_file.is_file() else ""

        manifest[name] = {
            "dataset_name": name,
            "download_source": f"https://huggingface.co/datasets/{name}",
            "expected_file": str(exp_file.relative_to(PROJECT_ROOT)) if exp_file.is_relative_to(PROJECT_ROOT) else str(exp_file),
            "detected_file": str(exp_file.name) if file_detected else "MISSING",
            "file_size_bytes": size_bytes,
            "row_count": emitted_rows,
            "checksum_sha256": chksum,
            "adapter_readiness": readiness,
        }

        audit_summary.append({
            "Dataset": name,
            "Expected File": exp_file.name,
            "Detected": "YES" if file_detected else "NO",
            "Size (MB)": round(size_bytes / (1024 * 1024), 2),
            "Emitted Rows": emitted_rows,
            "Status": readiness,
        })

    # Save manifest
    manifest_path = META / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"\nWrote dataset manifest to: {manifest_path}\n")

    # Display audit summary
    df_audit = pd.DataFrame(audit_summary)
    log("========================================================")
    log("MediTriageAI Dataset Bootstrap Audit Summary")
    log("========================================================")
    log(df_audit.to_string(index=False))
    log("========================================================\n")


if __name__ == "__main__":
    bootstrap_and_audit()
