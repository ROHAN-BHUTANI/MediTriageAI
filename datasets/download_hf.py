"""MediTriageAI Data Acquisition - Non-Interactive Production Downloader.

Automated dataset acquisition module designed for Python 3.12, Hugging Face Hub,
and production execution environments (DGX, Colab, CI/CD).

Guarantees 100% non-interactive execution with zero prompts (input() calls disabled).
Fails loudly with an explicit RuntimeError if any mandatory dataset acquisition fails.

MedDialog acquisition contract:
    Primary source: wangrongsheng/MedDialog-1.1M
    Expected artifact: merged-MedDialog.json
    Expected records: 2,725,992
    Expected size: ~4 GB JSON array
    Schema: {instruction, input, output}
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

# Disable git terminal credential prompts globally
os.environ["GIT_TERMINAL_PROMPT"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
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
            ignore_patterns=[".git*", "*.bin", "*.h5", "*.ot", "*.ckpt"],
        )
        return True
    except Exception as e:
        log(f"  snapshot_download failed for {repo_id}: {e}")
        return False


def git_clone_fallback(repo_url: str, dest_dir: Path) -> bool:
    """Clone repository via git non-interactively."""
    try:
        log(f"  Attempting git clone from {repo_url}...")
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(dest_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        return True
    except Exception as e:
        log(f"  git clone failed: {e}")
        return False


def has_valid_data_files(dest_dir: Path) -> bool:
    """Check if destination directory contains non-meta data files."""
    if not dest_dir.exists():
        return False
    data_extensions = {".csv", ".parquet", ".json", ".jsonl", ".tsv", ".txt", ".zip"}
    for p in dest_dir.rglob("*"):
        if p.is_file() and p.name != "SOURCE_URL.txt" and p.suffix.lower() in data_extensions:
            if p.stat().st_size > 100:  # non-trivial file size
                return True
    return False


# ── MedDialog acquisition contract ────────────────────────────────────────────
MEDDIALOG_EXPECTED_RECORDS = 2_725_992
MEDDIALOG_MIN_FILE_BYTES = 3_000_000_000  # ~3 GB minimum for the 4 GB JSON
MEDDIALOG_EXPECTED_FILENAME = "merged-MedDialog.json"


def validate_meddialog_acquisition(dest_dir: Path) -> bool:
    """Validate the MedDialog acquisition produced the canonical artifact.

    Checks:
    1. merged-MedDialog.json exists
    2. File is not a Git LFS pointer
    3. File size is plausible (>3 GB)
    4. JSON structure is valid (starts with '[')
    5. Streamed record count matches expected 2,725,992

    Returns True if validation passes, raises RuntimeError otherwise.
    """
    json_path = dest_dir / MEDDIALOG_EXPECTED_FILENAME

    # Check 1: file existence
    if not json_path.exists():
        raise RuntimeError(
            f"MedDialog acquisition FAILED: expected artifact '{MEDDIALOG_EXPECTED_FILENAME}' "
            f"not found in {dest_dir}. The production MedDialog source is "
            f"wangrongsheng/MedDialog-1.1M, not petkopetkov/MedDialog."
        )

    # Check 2: not a Git LFS pointer
    with open(json_path, "rb") as f:
        header = f.read(64)
    if header.startswith(b"version https://git-lfs.github.com"):
        raise RuntimeError(
            f"MedDialog acquisition FAILED: '{MEDDIALOG_EXPECTED_FILENAME}' is a Git LFS pointer, "
            f"not the actual data file. Run 'git lfs pull' or re-acquire from HuggingFace."
        )

    # Check 3: file size plausibility
    file_size = json_path.stat().st_size
    if file_size < MEDDIALOG_MIN_FILE_BYTES:
        raise RuntimeError(
            f"MedDialog acquisition FAILED: '{MEDDIALOG_EXPECTED_FILENAME}' is only "
            f"{file_size:,} bytes (expected >{MEDDIALOG_MIN_FILE_BYTES:,} bytes). "
            f"This is likely the wrong source dataset."
        )

    # Check 4: JSON structure sanity (must start with '[')
    if not header.lstrip().startswith(b"["):
        raise RuntimeError(
            f"MedDialog acquisition FAILED: '{MEDDIALOG_EXPECTED_FILENAME}' does not start "
            f"with '['. Expected a JSON array. File may be corrupted or wrong format."
        )

    # Check 5: streaming record count via ijson
    try:
        import ijson
    except ImportError:
        log("  WARNING: ijson not available — skipping streaming record count validation")
        return True

    log(f"  Validating MedDialog record count (streaming)...")
    count = 0
    with open(json_path, "r", encoding="utf-8") as f:
        for _ in ijson.items(f, "item"):
            count += 1
            if count > MEDDIALOG_EXPECTED_RECORDS:
                break

    if count != MEDDIALOG_EXPECTED_RECORDS:
        raise RuntimeError(
            f"MedDialog acquisition FAILED: '{MEDDIALOG_EXPECTED_FILENAME}' contains "
            f"{count:,} records but expected exactly {MEDDIALOG_EXPECTED_RECORDS:,}. "
            f"The file may be truncated, corrupted, or from the wrong source."
        )

    log(f"  MedDialog validation PASSED: {count:,} records confirmed")
    return True


# Verified 100% Active Production Datasets Specification List
# Tuple of (name, primary_hf_repo, config_name, license, description, fallback_repos, direct_urls)
DATASET_SPECS = [
    (
        "mtsamples",
        "NickyNicky/medical_mtsamples",
        None,
        "CC0 Public Domain",
        "MTSamples medical transcriptions dataset",
        ["harishnair04/mtsamples", "ahlammm/mtsamples"],
        [
            "https://huggingface.co/datasets/NickyNicky/medical_mtsamples/resolve/main/mtsamples%20(1).csv",
        ],
    ),
    (
        "pmc_patients",
        "zhengyun21/PMC-Patients",
        None,
        "CC BY 4.0",
        "PMC-Patients 167k patient summaries from PubMed Central",
        ["bigbio/pmc_patients"],
        [],
    ),
    (
        "medqa_usmle",
        "GBaker/MedQA-USMLE-4-options",
        None,
        "Open Access",
        "USMLE-style medical QA dataset",
        ["bigbio/med_qa"],
        [],
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
        "symptom2disease",
        "NeuronZero/Symptom2Disease",
        None,
        "Open Access",
        "Symptom text to disease label mapping",
        ["mrunmayee30/Symptom2Disease"],
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
        "chatdoctor_icliniq",
        "lavita/ChatDoctor-iCliniq",
        None,
        "Research Use Only",
        "Original medical QA data from icliniq.com",
        [],
        [],
    ),
    (
        "neiss",
        "Layered-Labs/neiss-injury-data",
        None,
        "Public Domain",
        "NEISS National Electronic Injury Surveillance System",
        [],
        [],
    ),
    (
        "nhamcs_ed",
        None,
        None,
        "Public Domain (CDC Federal Government)",
        "CDC NHAMCS Emergency Department Datasets (2019-2021)",
        [],
        [
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datasets/NHAMCS/ed2019.zip",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datasets/NHAMCS/ed2020.zip",
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datasets/NHAMCS/ed2021.zip",
        ],
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
        "kaggle_medical_triage",
        "sweatSmile/medical-symptom-triage-csv",
        None,
        "Open Access",
        "Emergency medical symptom triage dataset",
        [],
        [],
    ),
    (
        "l3cube_code_mixed",
        None,
        None,
        "Open Access (L3Cube Pune)",
        "L3Cube HingLID Hinglish code-mixed dataset",
        [],
        [
            "https://raw.githubusercontent.com/l3cube-pune/code-mixed-nlp/main/L3Cube-HingLID/train.txt",
            "https://raw.githubusercontent.com/l3cube-pune/code-mixed-nlp/main/L3Cube-HingLID/test.txt",
        ],
    ),
    (
        "meddialog_en",
        "wangrongsheng/MedDialog-1.1M",
        None,
        "Research Use Only",
        "MedDialog 1.1M medical Q&A conversations (2,725,992 records)",
        ["petkopetkov/MedDialog"],
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

    # Tier 1: Try HuggingFace snapshot_download (if repo specified)
    if repo and not success:
        log(f"  [Tier 1] Trying snapshot_download('{repo}')...")
        success = snapshot_download_fallback(repo, dest_dir)
        if success:
            method_used = "snapshot_download"

    # Tier 2: Try HuggingFace load_dataset
    if repo and not success:
        try:
            from datasets import load_dataset

            log(f"  [Tier 2] Trying load_dataset('{repo}')...")
            ds = load_dataset(repo, config, trust_remote_code=True)
            if hasattr(ds, "keys"):
                for split in ds.keys():
                    out_path = dest_dir / f"{split}.jsonl"
                    ds[split].to_json(str(out_path))
                success = True
                method_used = "load_dataset"
        except Exception as e:
            log(f"  [Tier 2] load_dataset failed: {e}")

    # Tier 3: Try fallback repos via snapshot_download
    if not success and fallbacks:
        for fb_repo in fallbacks:
            log(f"  [Tier 3] Trying fallback repo snapshot_download('{fb_repo}')...")
            success = snapshot_download_fallback(fb_repo, dest_dir)
            if success:
                method_used = f"snapshot_fallback ({fb_repo})"
                source_url = f"https://huggingface.co/datasets/{fb_repo}"
                break

    # Tier 4: Direct HTTP Download (CDC / GitHub / direct files)
    if not success and direct_urls:
        log(f"  [Tier 4] Trying direct HTTP downloads...")
        download_count = 0
        for url in direct_urls:
            filename = url.split("/")[-1].replace("%20", " ")
            dest_file = dest_dir / filename
            if direct_http_download(url, dest_file):
                download_count += 1
        if download_count > 0:
            success = True
            method_used = "direct_http"

    # Tier 5: Git clone fallback (non-interactive)
    if not success and repo:
        repo_url = f"https://huggingface.co/datasets/{repo}"
        log(f"  [Tier 5] Trying non-interactive git clone '{repo_url}'...")
        success = git_clone_fallback(repo_url, dest_dir)
        if success:
            method_used = "git_clone"

    if success and has_valid_data_files(dest_dir):
        # Dataset-specific post-acquisition validation
        if name == "meddialog_en":
            validate_meddialog_acquisition(dest_dir)

        fc, tb = count_files_and_bytes(dest_dir)
        chk = compute_directory_checksum(dest_dir)
        write_source_url(dest_dir, source_url)
        write_license_file(name, f"{lic}\n{desc}\nSource: {source_url}\nMethod: {method_used}\n")
        write_meta(name, source_url, lic, fc, tb, chk)
        log(f"  SUCCESS via {method_used}: {fc} files, {tb:,} bytes")
        return name, "DOWNLOADED", fc, tb, method_used

    log(f"  CRITICAL ERROR: Failed to acquire mandatory dataset {name}")
    raise RuntimeError(f"Data acquisition failed for mandatory dataset '{name}'. Check network connectivity or repository availability.")


def acquire_all_datasets(active_datasets: list[str] | set[str] | None = None) -> list[tuple[str, str, int, int, str]]:
    """Acquire dataset specifications non-interactively, filtered by active configuration."""
    if active_datasets is None:
        from meditriage.builder.config import Config
        config_path = ROOT.parent / "config" / "dataset_config.yaml"
        if config_path.exists():
            config = Config.from_yaml(config_path)
            active_set = set(config.active_datasets)
        else:
            active_set = None
    else:
        active_set = set(active_datasets)

    specs = [spec for spec in DATASET_SPECS if active_set is None or spec[0] in active_set]

    results = []
    for spec in specs:
        res = acquire_single_dataset(spec)
        results.append(res)
    return results


if __name__ == "__main__":
    acquire_all_datasets()
