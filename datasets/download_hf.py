"""MediTriageAI Data Acquisition - Modernized HF & Multimodal Downloader.

Modernized dataset acquisition module designed for Python 3.12, Hugging Face Hub,
and system-constrained environments. Supports multi-tiered fallback:
  1. load_dataset (with error recovery)
  2. snapshot_download (via huggingface_hub)
  3. Direct HTTP file download
  4. Git clone fallback

Preserves directory layout, metadata generation, SOURCE_URL.txt, LICENSE files,
SHA256 checksums, idempotency, and fault tolerance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

# Path definitions
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
META = ROOT / "metadata"
LICENSES = ROOT / "licenses"
LOGS = ROOT / "download_logs"

for d in [RAW, META, LICENSES, LOGS]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"))


def count_files_and_bytes(directory: Path) -> tuple[int, int]:
    """Count total data files and bytes in directory (excluding hidden/cache folders)."""
    n, s = 0, 0
    if not directory.exists():
        return 0, 0
    for p in directory.rglob("*"):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            n += 1
            s += p.stat().st_size
    return n, s


def compute_directory_checksum(directory: Path) -> str:
    """Compute combined SHA256 checksum of data files in directory."""
    hasher = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.name not in ("SOURCE_URL.txt", ".DS_Store") and not p.name.startswith("."):
            try:
                with open(p, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
            except Exception:
                pass
    return hasher.hexdigest()


def write_meta(
    name: str,
    source_url: str,
    lic: str,
    fc: int,
    tb: int,
    chk: str = "",
    ver: str = "1.0",
) -> None:
    """Write dataset metadata JSON."""
    meta_path = META / f"{name}.json"
    data = {
        "name": name,
        "source_url": source_url,
        "download_date": datetime.now(timezone.utc).isoformat(),
        "license": lic,
        "version": ver,
        "checksum": chk,
        "files_count": fc,
        "total_bytes": tb,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_source_url(dest_dir: Path, source_url: str) -> None:
    """Write SOURCE_URL.txt in dataset directory."""
    with open(dest_dir / "SOURCE_URL.txt", "w", encoding="utf-8") as f:
        f.write(source_url + "\n")


def write_license_file(name: str, lic_text: str) -> None:
    """Write license file in licenses directory."""
    with open(LICENSES / f"{name}_LICENSE.txt", "w", encoding="utf-8") as f:
        f.write(lic_text)


def direct_http_download(url: str, dest_file: Path, retries: int = 3, timeout: int = 300) -> bool:
    """Direct HTTP file downloader with retries."""
    for i in range(1, retries + 1):
        try:
            log(f"  GET ({i}/{retries}): {url}")
            req = Request(url, headers={"User-Agent": "MediTriageAI/1.0"})
            with urlopen(req, timeout=timeout) as response:
                content = response.read()
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_file, "wb") as f:
                f.write(content)
            log(f"  OK: {len(content):,} bytes -> {dest_file.name}")
            return True
        except Exception as e:
            log(f"  FAIL ({i}): {e}")
            if i < retries:
                time.sleep(2 * i)
    return False


def snapshot_download_fallback(repo_id: str, dest_dir: Path) -> bool:
    """Download HF dataset repository via huggingface_hub snapshot_download."""
    try:
        from huggingface_hub import snapshot_download

        log(f"  Attempting snapshot_download for {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dest_dir,
            allow_patterns=["*.csv", "*.json", "*.parquet", "*.jsonl", "*.tsv", "*.txt", "*.md"],
            ignore_patterns=[".git*", "*.bin", "*.h5", "*.ot", "*.ckpt"],
        )
        return True
    except Exception as e:
        log(f"  snapshot_download failed for {repo_id}: {e}")
        return False


def git_clone_fallback(repo_url: str, dest_dir: Path) -> bool:
    """Clone repository via git."""
    try:
        log(f"  Attempting git clone from {repo_url}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        log(f"  git clone failed: {e}")
        return False


def has_valid_data_files(dest_dir: Path) -> bool:
    """Check if destination directory contains non-meta data files."""
    if not dest_dir.exists():
        return False
    data_extensions = {".csv", ".parquet", ".json", ".jsonl", ".tsv", ".txt"}
    for p in dest_dir.rglob("*"):
        if p.is_file() and p.name != "SOURCE_URL.txt" and p.suffix.lower() in data_extensions:
            if p.stat().st_size > 100:  # non-trivial file size
                return True
    return False


# Core Datasets Specification List
# Tuple of (name, primary_hf_repo, config_name, license, description, fallback_repos, direct_urls)
DATASET_SPECS = [
    (
        "chatdoctor_icliniq",
        "lavita/ChatDoctor-iCliniq",
        None,
        "Research Use Only",
        "Original medical QA data from icliniq.com",
        [],
        ["https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq/resolve/main/chatdoctor_icliniq.json"],
    ),
    (
        "fedmml_ed_triage",
        "olaflaitinen/fedmml-ed-triage",
        None,
        "CC BY 4.0",
        "Synthetic ED triage encounters with ESI levels",
        [],
        [],
    ),
    (
        "symptom2disease",
        "NeuronZero/Symptom2Disease",
        None,
        "Open Access",
        "Symptom text to disease label mapping",
        ["mrunmayee30/Symptom2Disease"],
        ["https://raw.githubusercontent.com/Anshika-Gupta01/Symptom2Disease/main/Symptom2Disease.csv"],
    ),
    (
        "medqa_usmle",
        "GBaker/MedQA-USMLE-4-options",
        None,
        "Open Access",
        "USMLE-style medical QA",
        ["bigbio/med_qa"],
        ["https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/main/data/train-00000-of-00001.parquet"],
    ),
    (
        "medical_meadow_medqa",
        "medalpaca/medical_meadow_medqa",
        None,
        "Open Access",
        "Medical Meadow MedQA clinical instruction dataset",
        [],
        [],
    ),
    (
        "chatdoctor_healthcaremagic",
        "lavita/ChatDoctor-HealthCareMagic-100k",
        None,
        "Research Use Only",
        "ChatDoctor HealthCareMagic 100k clinical QA",
        [],
        [],
    ),
    (
        "mtsamples",
        "",
        None,
        "CC0 Public Domain",
        "MTSamples medical transcriptions dataset",
        [],
        [
            "https://raw.githubusercontent.com/wiki-yu/medical-report-classification/main/data/mtsamples.csv",
            "https://raw.githubusercontent.com/ShantanuSS/NLP-Medical-Transcriptions/master/mtsamples.csv",
        ],
    ),
    (
        "disease_symptom_description",
        "fhai50032/Symptoms_to_disease_7k",
        None,
        "Open Access",
        "Disease Symptom Description mapping",
        ["dux-tecblic/symptom-disease-dataset"],
        [],
    ),
]


def acquire_single_dataset(spec: tuple) -> tuple[str, str, int, int, str]:
    """Acquire a single dataset idempotently using multi-tiered fallbacks.

    Returns:
        Tuple of (name, status, files_count, total_bytes, method_used)
    """
    name, repo, config, lic, desc, fallbacks, direct_urls = spec
    log(f"\n==========================================")
    log(f"Acquiring Dataset: {name}")
    log(f"==========================================")

    dest_dir = RAW / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    source_url = f"https://huggingface.co/datasets/{repo}" if repo else (direct_urls[0] if direct_urls else "")

    # Idempotency Check: Skip if valid data files already exist
    if has_valid_data_files(dest_dir):
        log(f"  STATUS: Already exists in {dest_dir}")
        write_source_url(dest_dir, source_url)
        write_license_file(name, f"{lic}\n{desc}\nSource: {source_url}\n")
        fc, tb = count_files_and_bytes(dest_dir)
        chk = compute_directory_checksum(dest_dir)
        write_meta(name, source_url, lic, fc, tb, chk)
        return name, "EXISTS", fc, tb, "cached"

    method_used = "failed"
    success = False

    # Tier 1: Try HuggingFace load_dataset
    if repo:
        try:
            from datasets import load_dataset

            log(f"  [Tier 1] Trying load_dataset('{repo}')...")
            ds = load_dataset(repo, config, trust_remote_code=True)
            if hasattr(ds, "keys"):
                for split in ds.keys():
                    out_path = dest_dir / f"{split}.jsonl"
                    ds[split].to_json(str(out_path))
                success = True
                method_used = "load_dataset"
                log(f"  [Tier 1] Successfully loaded via load_dataset")
        except Exception as e:
            log(f"  [Tier 1] load_dataset failed: {e}")

    # Tier 2: Try snapshot_download from huggingface_hub
    if not success and repo:
        log(f"  [Tier 2] Trying snapshot_download('{repo}')...")
        if snapshot_download_fallback(repo, dest_dir):
            if has_valid_data_files(dest_dir):
                success = True
                method_used = "snapshot_download"
                log(f"  [Tier 2] Successfully downloaded via snapshot_download")

    # Tier 2b: Try fallback HF repos if primary repo failed
    if not success and fallbacks:
        for fb_repo in fallbacks:
            log(f"  [Tier 2b] Trying fallback repo snapshot_download('{fb_repo}')...")
            if snapshot_download_fallback(fb_repo, dest_dir):
                if has_valid_data_files(dest_dir):
                    success = True
                    method_used = f"snapshot_download({fb_repo})"
                    source_url = f"https://huggingface.co/datasets/{fb_repo}"
                    log(f"  [Tier 2b] Successfully downloaded via fallback repo")
                    break

    # Tier 3: Try Direct HTTP File Download
    if not success and direct_urls:
        log(f"  [Tier 3] Trying Direct HTTP Downloads...")
        for u in direct_urls:
            filename = u.split("/")[-1].split("?")[0]
            if not filename or len(filename) < 3:
                filename = "dataset.csv"
            dest_file = dest_dir / filename
            if direct_http_download(u, dest_file):
                success = True
                method_used = "direct_http"
                source_url = u
                log(f"  [Tier 3] Successfully downloaded via direct HTTP")
                break

    # Tier 4: Try Git Clone
    if not success and repo:
        log(f"  [Tier 4] Trying git clone...")
        repo_url = f"https://huggingface.co/datasets/{repo}"
        if git_clone_fallback(repo_url, dest_dir):
            if has_valid_data_files(dest_dir):
                success = True
                method_used = "git_clone"
                log(f"  [Tier 4] Successfully cloned via git")

    # Finalize & Generate Metadata, License, Checksum
    if success and has_valid_data_files(dest_dir):
        write_source_url(dest_dir, source_url)
        write_license_file(name, f"{lic}\n{desc}\nSource: {source_url}\n")
        fc, tb = count_files_and_bytes(dest_dir)
        chk = compute_directory_checksum(dest_dir)
        write_meta(name, source_url, lic, fc, tb, chk)
        log(f"SUCCESS {name}: {fc} files, {tb:,} bytes (Method: {method_used})")
        return name, "DOWNLOADED", fc, tb, method_used
    else:
        log(f"FAILED {name}: All acquisition tiers failed.")
        return name, "FAILED", 0, 0, "failed"


def main():
    log("Starting MediTriageAI Modernized Dataset Downloader Engine")
    log(f"Target Directory: {RAW.resolve()}")

    results = []
    for spec in DATASET_SPECS:
        try:
            res = acquire_single_dataset(spec)
            results.append(res)
        except Exception as e:
            log(f"CRITICAL ERROR acquiring {spec[0]}: {e}")
            results.append((spec[0], "FAILED", 0, 0, "error"))

    # Print Final Summary Report
    log("\n" + "=" * 80)
    log("MEDITRIAGEAI DATASET ACQUISITION SUMMARY REPORT")
    log("=" * 80)
    log(f"{'STATUS':<12} | {'DATASET':<28} | {'FILES':<6} | {'BYTES':<14} | {'METHOD':<20}")
    log("-" * 80)

    total_files = 0
    total_bytes = 0
    success_count = 0

    for name, status, fc, tb, method in results:
        total_files += fc
        total_bytes += tb
        if status in ("DOWNLOADED", "EXISTS"):
            success_count += 1
        log(f"{status:<12} | {name:<28} | {fc:<6} | {tb:<14,} | {method:<20}")

    log("=" * 80)
    log(f"TOTAL: {success_count}/{len(results)} datasets acquired | {total_files} files | {total_bytes:,} bytes")
    log("=" * 80 + "\n")


if __name__ == "__main__":
    main()
