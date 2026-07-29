"""MediTriageAI Data Acquisition Engine.

Downloads all freely obtainable datasets relevant to clinical triage,
multilingual NLP, and code-mixed text processing.

Usage:
    python datasets/download_all.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import tarfile
try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    pass


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
META = ROOT / "metadata"
LICENSES = ROOT / "licenses"
LOGS = ROOT / "download_logs"

for d in [RAW, META, LICENSES, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS / f"acquisition_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"

def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode('ascii'))
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@dataclass
class DatasetResult:
    name: str
    status: str  # DOWNLOADED, SKIPPED, FAILED
    source_url: str
    license: str
    reason: str = ""
    files_count: int = 0
    total_bytes: int = 0
    checksum: str = ""
    download_date: str = ""
    version: str = ""
    local_path: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def count_files_in_dir(d: Path) -> tuple[int, int]:
    total_files = 0
    total_bytes = 0
    for p in d.rglob("*"):
        if p.is_file():
            total_files += 1
            total_bytes += p.stat().st_size
    return total_files, total_bytes


def safe_download(url: str, dest: Path, max_retries: int = 3, timeout: int = 120) -> bytes | None:
    """Download a URL to dest file. Returns bytes on success, None on failure."""
    headers = {"User-Agent": "MediTriageAI-DataEngine/1.0 (Research)"}
    for attempt in range(1, max_retries + 1):
        try:
            log(f"  Attempt {attempt}/{max_retries}: GET {url}")
            req = Request(url, headers=headers)
            resp = urlopen(req, timeout=timeout)
            data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            log(f"  Downloaded {len(data):,} bytes -> {dest.name}")
            return data
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            log(f"  Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
    return None


def write_metadata(name: str, result: DatasetResult) -> None:
    meta_path = META / f"{name}.json"
    meta = {
        "name": name,
        "source_url": result.source_url,
        "download_date": result.download_date,
        "license": result.license,
        "version": result.version,
        "checksum": result.checksum,
        "files_count": result.files_count,
        "total_bytes": result.total_bytes,
        "status": result.status,
        "local_path": result.local_path,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)



def safe_hf_download(repo_id: str, dest_dir: Path, allow_patterns=None) -> bool:
    try:
        log(f"  Downloading from HuggingFace Hub: {repo_id}")
        snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=allow_patterns, local_dir=str(dest_dir))
        return True
    except Exception as e:
        log(f"  HF Download Failed for {repo_id}: {e}")
        return False

def write_source_url(dataset_dir: Path, url: str) -> None:
    with open(dataset_dir / "SOURCE_URL.txt", "w", encoding="utf-8") as f:
        f.write(url + "\n")


# ==============================================================================
# Dataset 1: MTSamples (Kaggle - direct CSV from known mirrors)
# ==============================================================================
def download_mtsamples() -> DatasetResult:
    name = "mtsamples"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/NickyNicky/medical_mtsamples"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "CC0 Public Domain", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Try Kaggle CLI first
    log(f"[{name}] Attempting Kaggle CLI download...")
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "tboyle10/medicaltranscriptions",
             "-p", str(dest_dir), "--unzip"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            log(f"[{name}] Kaggle CLI download succeeded.")
            source_url = "https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions"
            success = True
        else:
            success = False
    except Exception as e:
        success = False
        
    if not success:
        log(f"[{name}] Kaggle CLI failed. Attempting HF mirror...")
        success = safe_hf_download("NickyNicky/medical_mtsamples", dest_dir)

    if not success:
        return DatasetResult(name, "FAILED", source_url, "CC0 Public Domain", "All download methods failed")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("CC0 1.0 Universal (CC0 1.0) Public Domain Dedication\n")
        f.write("Source: Kaggle / mtsamples.com\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "CC0 Public Domain", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "mtsamples.csv"

    data = safe_download(url, dest_file)
    if data is None:
        log(f"[{name}] Primary URL failed, trying alternative...")
        data = safe_download(alt_url, dest_file)

    if data is None:
        return DatasetResult(name, "FAILED", url, "CC0 Public Domain", "All download URLs failed")

    write_source_url(dest_dir, url)
    checksum = sha256_bytes(data)
    fc, tb = count_files_in_dir(dest_dir)

    # Write license
    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("CC0 1.0 Universal (CC0 1.0) Public Domain Dedication\n")
        f.write("Source: Kaggle / mtsamples.com\n")
        f.write("Contributor: Tara Boyle\n")

    return DatasetResult(name, "DOWNLOADED", url, "CC0 Public Domain", "",
                         fc, tb, checksum, datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))


# ==============================================================================
# Dataset 2: PMC-Patients (HuggingFace - direct JSON download)
# ==============================================================================
def download_pmc_patients() -> DatasetResult:
    name = "pmc_patients"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/zhengyun21/PMC-Patients"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "CC BY-NC-SA 4.0", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    success = safe_hf_download("zhengyun21/PMC-Patients", dest_dir)

    if not success:
        return DatasetResult(name, "FAILED", source_url, "CC BY-NC-SA 4.0", "All download URLs failed")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)\n")
        f.write("Source: https://huggingface.co/datasets/zhengyun21/PMC-Patients\n")
        f.write("Citation: Zhao et al. 2023, Scientific Data 10(1):909\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "CC BY-NC-SA 4.0", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "2.0", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "PMC-Patients.json"
    data = safe_download(url, dest_file, timeout=300)

    if data is None:
        # Try parquet version
        alt_url = "https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/data/train-00000-of-00001.parquet"
        log(f"[{name}] JSON failed, trying parquet...")
        dest_file_alt = dest_dir / "train-00000-of-00001.parquet"
        data = safe_download(alt_url, dest_file_alt, timeout=300)
        if data is None:
            return DatasetResult(name, "FAILED", url, "CC BY-NC-SA 4.0", "All download URLs failed")

    write_source_url(dest_dir, "https://huggingface.co/datasets/zhengyun21/PMC-Patients")
    checksum = sha256_bytes(data)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)\n")
        f.write("Source: https://huggingface.co/datasets/zhengyun21/PMC-Patients\n")
        f.write("Citation: Zhao et al. 2023, Scientific Data 10(1):909\n")

    return DatasetResult(name, "DOWNLOADED", url, "CC BY-NC-SA 4.0", "",
                         fc, tb, checksum, datetime.now(timezone.utc).isoformat(), "2.0", str(dest_dir))


# ==============================================================================
# Dataset 3: MedDialog English (Google Drive via GitHub mirror)
# ==============================================================================
def download_meddialog() -> DatasetResult:
    name = "meddialog_en"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/UCSD26/medical_dialog"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "Research Use", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    success = safe_hf_download("UCSD26/medical_dialog", dest_dir, allow_patterns="*en-train*")

    if not success:
        return DatasetResult(name, "FAILED", source_url, "Research Use",
                             "Could not download from HuggingFace.")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Research Use Only\n")
        f.write("Copyrights belong to icliniq.com and healthcaremagic.com\n")
        f.write("Source: https://github.com/UCSD-AI4H/Medical-Dialogue-System\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Research Use", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try HuggingFace parquet first
    dest_file = dest_dir / "en-train.parquet"
    data = safe_download(url, dest_file, timeout=300)

    if data is None:
        # Try alternative structure
        alt_url = "https://huggingface.co/datasets/UCSD26/medical_dialog/resolve/main/medical_dialog-en-train.parquet"
        data = safe_download(alt_url, dest_file, timeout=300)

    if data is None:
        return DatasetResult(name, "FAILED", source_url, "Research Use",
                             "HuggingFace parquet not directly accessible; requires manual download from Google Drive: "
                             "https://drive.google.com/drive/folders/1g29ssimdZ6JzTST6Y8g6h-ogUNReBtJD")

    write_source_url(dest_dir, source_url)
    checksum = sha256_bytes(data)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Research Use Only\n")
        f.write("Copyrights belong to icliniq.com and healthcaremagic.com\n")
        f.write("Source: https://github.com/UCSD-AI4H/Medical-Dialogue-System\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Research Use", "",
                         fc, tb, checksum, datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))


# ==============================================================================
# Dataset 4: ChatDoctor HealthCareMagic (HuggingFace)
# ==============================================================================
def download_chatdoctor_hcm() -> DatasetResult:
    name = "chatdoctor_healthcaremagic"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "Research Use", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    success = safe_hf_download("lavita/ChatDoctor-HealthCareMagic-100k", dest_dir)

    if not success:
        return DatasetResult(name, "FAILED", source_url, "Research Use",
                             "Could not download from HuggingFace; may require authentication or manual access")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Research Use Only\n")
        f.write("Source: https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k\n")
        f.write("Original data from healthcaremagic.com\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Research Use", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "chatdoctor-healthcaremagic.json"
    data = safe_download(url, dest_file, timeout=300)

    if data is None:
        # Try parquet
        alt_url = "https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k/resolve/main/data/train-00000-of-00001.parquet"
        dest_file = dest_dir / "train.parquet"
        data = safe_download(alt_url, dest_file, timeout=300)

    if data is None:
        return DatasetResult(name, "FAILED", source_url, "Research Use",
                             "Could not download from HuggingFace; may require authentication or manual access")

    write_source_url(dest_dir, source_url)
    checksum = sha256_bytes(data)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Research Use Only\n")
        f.write("Source: https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k\n")
        f.write("Original data from healthcaremagic.com\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Research Use", "",
                         fc, tb, checksum, datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))


# ==============================================================================
# Dataset 5: ChatDoctor iCliniq (HuggingFace)
# ==============================================================================
def download_chatdoctor_icliniq() -> DatasetResult:
    name = "chatdoctor_icliniq"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "Research Use", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    success = safe_hf_download("lavita/ChatDoctor-iCliniq", dest_dir)

    if not success:
        return DatasetResult(name, "FAILED", source_url, "Research Use",
                             "Could not download from HuggingFace; may require authentication or manual access")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Research Use Only\n")
        f.write("Source: https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq\n")
        f.write("Original data from icliniq.com\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Research Use", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "chatdoctor-icliniq.json"
    data = safe_download(url, dest_file, timeout=300)

    if data is None:
        alt_url = "https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq/resolve/main/data/train-00000-of-00001.parquet"
        dest_file = dest_dir / "train.parquet"
        data = safe_download(alt_url, dest_file, timeout=300)

    if data is None:
        return DatasetResult(name, "FAILED", source_url, "Research Use",
                             "Could not download from HuggingFace; may require authentication or manual access")

    write_source_url(dest_dir, source_url)
    checksum = sha256_bytes(data)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Research Use Only\n")
        f.write("Source: https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq\n")
        f.write("Original data from icliniq.com\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Research Use", "",
                         fc, tb, checksum, datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))


# ==============================================================================
# Dataset 6: FedMML ED Triage (HuggingFace - synthetic ESI triage)
# ==============================================================================
def download_fedmml_triage() -> DatasetResult:
    name = "fedmml_ed_triage"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "Open", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    success = safe_hf_download("olaflaitinen/fedmml-ed-triage", dest_dir)

    if not success:
        return DatasetResult(name, "FAILED", source_url, "Open", "Could not download from HuggingFace")
        
    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Open Access (Synthetic Data)\n")
        f.write(f"Source: {source_url}\n")
        f.write("~87,000 synthetic ED triage encounters with ESI levels\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Open (Synthetic)", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try parquet files
    for split in ["train", "test", "validation"]:
        url = f"https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage/resolve/main/data/{split}-00000-of-00001.parquet"
        dest_file = dest_dir / f"{split}.parquet"
        safe_download(url, dest_file, timeout=180)

    fc, tb = count_files_in_dir(dest_dir)
    if fc == 0:
        # Try CSV
        url = f"https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage/resolve/main/fedmml_ed_triage.csv"
        dest_file = dest_dir / "fedmml_ed_triage.csv"
        data = safe_download(url, dest_file, timeout=180)
        if data is None:
            return DatasetResult(name, "FAILED", source_url, "Open",
                                 "Could not download from HuggingFace")
        fc, tb = count_files_in_dir(dest_dir)

    write_source_url(dest_dir, source_url)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Open Access (Synthetic Data)\n")
        f.write(f"Source: {source_url}\n")
        f.write("~87,000 synthetic ED triage encounters with ESI levels\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Open (Synthetic)", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))


# ==============================================================================
# Dataset 7: Synthetic Medical Triage (Kaggle)
# ==============================================================================
def download_kaggle_triage() -> DatasetResult:
    name = "kaggle_medical_triage"
    dest_dir = RAW / name
    source_url = "https://www.kaggle.com/datasets/daniilkrasnoproshin/medical-triage-priority-dataset"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "Open", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try Kaggle CLI
    log(f"[{name}] Attempting Kaggle CLI download...")
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "daniilkrasnoproshin/medical-triage-priority-dataset",
             "-p", str(dest_dir), "--unzip"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            log(f"[{name}] Kaggle CLI download succeeded.")
            write_source_url(dest_dir, source_url)
            fc, tb = count_files_in_dir(dest_dir)
            with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
                f.write("Open Access (Synthetic Data)\n")
                f.write(f"Source: {source_url}\n")
            return DatasetResult(name, "DOWNLOADED", source_url, "Open", "",
                                 fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))
        else:
            log(f"[{name}] Kaggle CLI failed: {result.stderr}")
    except Exception as e:
        log(f"[{name}] Kaggle CLI not available: {e}")

    return DatasetResult(name, "FAILED", source_url, "Open",
                         "Kaggle CLI not available or authentication failed. "
                         "Download manually from: " + source_url)


# ==============================================================================
# Dataset 8: NHAMCS ED Data (CDC)
# ==============================================================================
def download_nhamcs() -> DatasetResult:
    name = "nhamcs_ed"
    dest_dir = RAW / name
    source_url = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS/"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "US Government Public Domain", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download the most recent ED public-use files (2020, 2021 are latest available)
    years_to_try = ["2021", "2020", "2019"]
    any_success = False

    for year in years_to_try:
        # CDC FTP naming conventions vary; try common patterns
        for fmt in [f"ed{year}.zip", f"ED{year}.zip", f"ed{year}pub.zip"]:
            url = f"https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS/{fmt}"
            dest_file = dest_dir / fmt
            data = safe_download(url, dest_file, max_retries=2, timeout=120)
            if data is not None:
                any_success = True
                # Try to extract
                try:
                    with zipfile.ZipFile(dest_file, 'r') as zf:
                        zf.extractall(dest_dir / f"ed{year}")
                    log(f"[{name}] Extracted {fmt}")
                except zipfile.BadZipFile:
                    log(f"[{name}] {fmt} is not a valid zip, keeping raw file")
                break

    if not any_success:
        # Document that NHAMCS data requires specific file paths from CDC
        return DatasetResult(name, "FAILED", source_url, "US Government Public Domain",
                             "CDC FTP paths not resolvable via automated download. "
                             "Manual download required from: https://www.cdc.gov/nchs/ahcd/datasets_documentation_related.htm "
                             "NHAMCS ended in 2022; data available 1992-2022.")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("US Government Public Domain\n")
        f.write("Source: CDC/NCHS National Hospital Ambulatory Medical Care Survey\n")
        f.write(f"Portal: {source_url}\n")
        f.write("Note: NHAMCS ended in 2022.\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "US Government Public Domain", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "2021", str(dest_dir))


# ==============================================================================
# Dataset 9: NEISS (CPSC)
# ==============================================================================
def download_neiss() -> DatasetResult:
    name = "neiss"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/Layered-Labs/neiss-injury-data"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "US Government Public Domain", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)

    success = safe_hf_download("Layered-Labs/neiss-injury-data", dest_dir)
    
    if not success:
        return DatasetResult(name, "FAILED", source_url, "US Government Public Domain",
                             "Could not download NEISS mirror from HuggingFace.")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("US Government Public Domain\n")
        f.write("Source: CPSC National Electronic Injury Surveillance System\n")
        f.write(f"Portal: {source_url}\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "US Government Public Domain", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "2023", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)

    # NEISS uses an interactive query system - try direct TSV links
    # Historical NEISS data available via CPSC
    any_success = False
    for year in [2023, 2022, 2021]:
        url = f"https://www.cpsc.gov/cgibin/NEISSQuery/Data/Archived/neiss{year}.tsv"
        dest_file = dest_dir / f"neiss{year}.tsv"
        data = safe_download(url, dest_file, max_retries=2, timeout=120)
        if data is not None:
            any_success = True

    if not any_success:
        return DatasetResult(name, "FAILED", source_url, "US Government Public Domain",
                             "NEISS data requires interactive query at cpsc.gov portal. "
                             "Bulk download not available via direct URL. "
                             "Manual download required from: https://www.cpsc.gov/cgibin/NEISSQuery/home.aspx")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("US Government Public Domain\n")
        f.write("Source: CPSC National Electronic Injury Surveillance System\n")
        f.write(f"Portal: {source_url}\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "US Government Public Domain", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "2023", str(dest_dir))


# ==============================================================================
# Dataset 10: L3Cube Code-Mixed NLP (GitHub)
# ==============================================================================
def download_l3cube() -> DatasetResult:
    name = "l3cube_code_mixed"
    dest_dir = RAW / name
    source_url = "https://github.com/l3cube-pune/code-mixed-nlp"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "MIT / CC BY 4.0", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download the repo as ZIP
    zip_url = "https://github.com/l3cube-pune/code-mixed-nlp/archive/refs/heads/main.zip"
    dest_file = dest_dir / "code-mixed-nlp.zip"
    data = safe_download(zip_url, dest_file, timeout=120)

    if data is not None:
        try:
            with zipfile.ZipFile(dest_file, 'r') as zf:
                zf.extractall(dest_dir)
            log(f"[{name}] Extracted GitHub repo archive")
        except zipfile.BadZipFile:
            log(f"[{name}] Bad zip file")
            return DatasetResult(name, "FAILED", source_url, "MIT / CC BY 4.0", "Downloaded zip was corrupted")
    else:
        return DatasetResult(name, "FAILED", source_url, "MIT / CC BY 4.0", "Could not download from GitHub")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("MIT License / CC BY 4.0\n")
        f.write("Source: https://github.com/l3cube-pune/code-mixed-nlp\n")
        f.write("L3Cube-HingCorpus: 52.93M Hinglish sentences, 1.04B tokens\n")
        f.write("Citation: Nayak & Joshi, LREC 2022\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "MIT / CC BY 4.0", "",
                         fc, tb, sha256_bytes(data), datetime.now(timezone.utc).isoformat(), "main", str(dest_dir))


# ==============================================================================
# Dataset 11: Medical Meadow (HuggingFace - medical QA)
# ==============================================================================
def download_medical_meadow() -> DatasetResult:
    name = "medical_meadow_medqa"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/medalpaca/medical_meadow_medqa"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "Open", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    success = safe_hf_download("medalpaca/medical_meadow_medqa", dest_dir)

    if not success:
        return DatasetResult(name, "FAILED", source_url, "Open",
                             "Could not download from HuggingFace")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Open Access\n")
        f.write(f"Source: {source_url}\n")
        f.write("Medical QA dataset for LLM fine-tuning\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Open", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/datasets/medalpaca/medical_meadow_medqa/resolve/main/data/train-00000-of-00001.parquet"
    dest_file = dest_dir / "train.parquet"
    data = safe_download(url, dest_file, timeout=180)

    if data is None:
        # Try JSON format
        alt_url = "https://huggingface.co/datasets/medalpaca/medical_meadow_medqa/resolve/main/medical_meadow_medqa.json"
        dest_file = dest_dir / "medical_meadow_medqa.json"
        data = safe_download(alt_url, dest_file, timeout=180)

    if data is None:
        return DatasetResult(name, "FAILED", source_url, "Open",
                             "Could not download from HuggingFace")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Open Access\n")
        f.write(f"Source: {source_url}\n")
        f.write("Medical QA dataset for LLM fine-tuning\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Open", "",
                         fc, tb, sha256_bytes(data), datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))


# ==============================================================================
# Dataset 12: Symptom2Disease (Kaggle/HuggingFace)
# ==============================================================================
def download_symptom2disease() -> DatasetResult:
    name = "symptom2disease"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/NeuronZero/Symptom2Disease"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "Open", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    
    success = safe_hf_download("NeuronZero/Symptom2Disease", dest_dir)

    if not success:
        return DatasetResult(name, "FAILED", source_url, "Open", "Could not download from HuggingFace")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Open Access\n")
        f.write(f"Source: {source_url}\n")
        f.write("Symptom text to disease label mapping\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Open", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    url = "https://huggingface.co/datasets/QuyenAnhDE/Symptom2Disease/resolve/main/data/train-00000-of-00001.parquet"
    dest_file = dest_dir / "train.parquet"
    data = safe_download(url, dest_file, timeout=120)

    if data is None:
        return DatasetResult(name, "FAILED", source_url, "Open", "Could not download from HuggingFace")

    # Try test split too
    test_url = "https://huggingface.co/datasets/QuyenAnhDE/Symptom2Disease/resolve/main/data/test-00000-of-00001.parquet"
    safe_download(test_url, dest_dir / "test.parquet", timeout=120)
    val_url = "https://huggingface.co/datasets/QuyenAnhDE/Symptom2Disease/resolve/main/data/validation-00000-of-00001.parquet"
    safe_download(val_url, dest_dir / "validation.parquet", timeout=120)

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Open Access\n")
        f.write(f"Source: {source_url}\n")
        f.write("Symptom text to disease label mapping\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Open", "",
                         fc, tb, sha256_bytes(data), datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))


# ==============================================================================
# Dataset 13: MedQA (USMLE-style medical QA)
# ==============================================================================
def download_medqa() -> DatasetResult:
    name = "medqa_usmle"
    dest_dir = RAW / name
    source_url = "https://huggingface.co/datasets/bigbio/med_qa"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        log(f"[{name}] Already exists, skipping.")
        fc, tb = count_files_in_dir(dest_dir)
        return DatasetResult(name, "SKIPPED", source_url, "Open", "Already downloaded", fc, tb,
                             download_date=datetime.now(timezone.utc).isoformat(), local_path=str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)
    success = safe_hf_download("bigbio/med_qa", dest_dir)

    if not success:
        return DatasetResult(name, "FAILED", source_url, "Open",
                             "Could not download from HuggingFace")

    write_source_url(dest_dir, source_url)
    fc, tb = count_files_in_dir(dest_dir)
    
    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Open Access\n")
        f.write(f"Source: {source_url}\n")
        f.write("USMLE-style medical QA from Jin et al., 2021\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Open", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Try GBQ/Huggingface parquet
    for split in ["train", "test", "validation"]:
        url = f"https://huggingface.co/datasets/bigbio/med_qa/resolve/main/data/med_qa_en_source/{split}-00000-of-00001.parquet"
        dest_file = dest_dir / f"{split}.parquet"
        safe_download(url, dest_file, timeout=180)

    fc, tb = count_files_in_dir(dest_dir)
    if fc == 0:
        return DatasetResult(name, "FAILED", source_url, "Open",
                             "Could not download from HuggingFace; may need `datasets` library")

    write_source_url(dest_dir, source_url)

    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f:
        f.write("Open Access\n")
        f.write(f"Source: {source_url}\n")
        f.write("USMLE-style medical QA from Jin et al., 2021\n")

    return DatasetResult(name, "DOWNLOADED", source_url, "Open", "",
                         fc, tb, "", datetime.now(timezone.utc).isoformat(), "1.0", str(dest_dir))


# ==============================================================================
# Main Execution
# ==============================================================================
def generate_inventory(results: list[DatasetResult]) -> None:
    """Generate DATASET_INVENTORY.md in the datasets directory."""
    downloaded = [r for r in results if r.status == "DOWNLOADED"]
    skipped = [r for r in results if r.status == "SKIPPED"]
    failed = [r for r in results if r.status == "FAILED"]

    inventory_path = ROOT / "DATASET_INVENTORY.md"
    with open(inventory_path, "w", encoding="utf-8") as f:
        f.write("# MediTriageAI — Dataset Inventory Report\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write("---\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write(f"| Status | Count |\n")
        f.write(f"|:---|:---:|\n")
        f.write(f"| ✅ Downloaded | {len(downloaded)} |\n")
        f.write(f"| ⏭️ Skipped (already present) | {len(skipped)} |\n")
        f.write(f"| ❌ Failed | {len(failed)} |\n")
        f.write(f"| **Total Attempted** | **{len(results)}** |\n\n")

        # Downloaded
        f.write("---\n\n## ✅ Successfully Downloaded Datasets\n\n")
        if downloaded:
            f.write("| Dataset | License | Files | Size | Source |\n")
            f.write("|:---|:---|:---:|:---:|:---|\n")
            for r in downloaded:
                size_str = f"{r.total_bytes:,} bytes" if r.total_bytes else "N/A"
                f.write(f"| **{r.name}** | {r.license} | {r.files_count} | {size_str} | [Link]({r.source_url}) |\n")
        else:
            f.write("*No new datasets downloaded in this run.*\n")
        f.write("\n")

        # Details for each downloaded dataset
        for r in downloaded:
            f.write(f"### {r.name}\n")
            f.write(f"- **Source:** {r.source_url}\n")
            f.write(f"- **License:** {r.license}\n")
            f.write(f"- **Download Date:** {r.download_date}\n")
            f.write(f"- **Local Path:** `{r.local_path}`\n")
            f.write(f"- **Files:** {r.files_count}\n")
            f.write(f"- **Total Size:** {r.total_bytes:,} bytes\n")
            if r.checksum:
                f.write(f"- **SHA-256:** `{r.checksum[:16]}...`\n")
            f.write("\n")

        # Skipped
        f.write("---\n\n## ⏭️ Skipped Datasets\n\n")
        if skipped:
            f.write("| Dataset | Reason |\n")
            f.write("|:---|:---|\n")
            for r in skipped:
                f.write(f"| {r.name} | {r.reason} |\n")
        else:
            f.write("*None.*\n")
        f.write("\n")

        # Failed
        f.write("---\n\n## ❌ Failed Downloads\n\n")
        if failed:
            for r in failed:
                f.write(f"### {r.name}\n")
                f.write(f"- **Source:** {r.source_url}\n")
                f.write(f"- **License:** {r.license}\n")
                f.write(f"- **Reason:** {r.reason}\n\n")
        else:
            f.write("*All downloads succeeded.*\n")
        f.write("\n")

        # Credentialed datasets
        f.write("---\n\n## 🔒 Credentialed Datasets (Require Manual Access)\n\n")
        f.write("The following datasets are highly relevant to MediTriageAI but require manual registration, "
                "institutional credentials, or data use agreements:\n\n")
        f.write("| Dataset | Source | Access Requirement | Relevance |\n")
        f.write("|:---|:---|:---|:---|\n")
        f.write("| **MIMIC-IV-ED** | PhysioNet | CITI training + DUA | Gold standard ED triage with ESI scores |\n")
        f.write("| **MIETIC** | PhysioNet | CITI training + DUA | MIMIC-IV triage instruction corpus for LLMs |\n")
        f.write("| **eICU** | PhysioNet | CITI training + DUA | Multi-center ICU data with acuity scores |\n")
        f.write("| **i2b2 NLP Challenges** | DBMI Portal | DUA + institutional affiliation | De-identified clinical notes |\n")
        f.write("| **n2c2 NLP Challenges** | DBMI Portal | DUA + institutional affiliation | Clinical NLP benchmarks |\n")
        f.write("| **UK Biobank** | UK Biobank | Institutional access | Large-scale health data |\n")
        f.write("| **CPRD** | CPRD/MHRA | Institutional access + fee | UK primary care data |\n")
        f.write("\n")

        f.write("---\n\n## 📋 Acquisition Log\n\n")
        f.write(f"Full acquisition log: `{LOG_FILE.relative_to(ROOT)}`\n")

    log(f"Inventory written to {inventory_path}")


def package_datasets(results: list[DatasetResult]) -> None:
    log("=" * 60)
    log("Packaging legally redistributable datasets into datasets_bundle.tar.gz...")
    
    non_redistributable = []
    bundle_path = ROOT.parent / "datasets_bundle.tar.gz"
    
    with tarfile.open(bundle_path, "w:gz") as tar:
        for r in results:
            if r.status in ("DOWNLOADED", "SKIPPED") and r.local_path:
                if "No Redistribution" in r.license:
                    non_redistributable.append(r.name)
                    log(f"  Skipping {r.name} from bundle (License: {r.license})")
                else:
                    log(f"  Adding {r.name} to bundle...")
                    tar.add(r.local_path, arcname=f"raw/{r.name}")
                    
        # Add metadata and licenses
        tar.add(META, arcname="metadata")
        tar.add(LICENSES, arcname="licenses")
        if (ROOT / "DATASET_INVENTORY.md").exists():
            tar.add(ROOT / "DATASET_INVENTORY.md", arcname="DATASET_INVENTORY.md")
        
    log(f"Datasets bundled successfully at {bundle_path}")
    if non_redistributable:
        log(f"Excluded due to license constraints: {', '.join(non_redistributable)}")

def main() -> None:
    log("=" * 60)
    log("MediTriageAI Data Acquisition Engine — Starting")
    log("=" * 60)

    results: list[DatasetResult] = []

    # Execute all download functions
    download_functions = [
        ("MTSamples", download_mtsamples),
        ("PMC-Patients", download_pmc_patients),
        ("MedDialog English", download_meddialog),
        ("ChatDoctor HealthCareMagic", download_chatdoctor_hcm),
        ("ChatDoctor iCliniq", download_chatdoctor_icliniq),
        ("FedMML ED Triage", download_fedmml_triage),
        ("Kaggle Medical Triage", download_kaggle_triage),
        ("NHAMCS ED (CDC)", download_nhamcs),
        ("NEISS (CPSC)", download_neiss),
        ("L3Cube Code-Mixed NLP", download_l3cube),
        ("Medical Meadow MedQA", download_medical_meadow),
        ("Symptom2Disease", download_symptom2disease),
        ("MedQA USMLE", download_medqa),
    ]

    for display_name, func in download_functions:
        log(f"\n{'=' * 40}")
        log(f"Processing: {display_name}")
        log(f"{'=' * 40}")
        try:
            result = func()
            results.append(result)
            write_metadata(result.name, result)
            log(f"  → {result.status}: {result.name} ({result.files_count} files, {result.total_bytes:,} bytes)")
        except Exception as e:
            log(f"  → EXCEPTION in {display_name}: {e}")
            log(traceback.format_exc())
            results.append(DatasetResult(display_name, "FAILED", "", "Unknown", f"Exception: {e}"))

    # Generate inventory
    log(f"\n{'=' * 60}")
    log("Generating inventory report...")
    generate_inventory(results)

    # Final summary
    downloaded = sum(1 for r in results if r.status == "DOWNLOADED")
    skipped = sum(1 for r in results if r.status == "SKIPPED")
    failed = sum(1 for r in results if r.status == "FAILED")
    log(f"\nFINAL SUMMARY: {downloaded} downloaded, {skipped} skipped, {failed} failed out of {len(results)} total")
    log("=" * 60)
    package_datasets(results)


if __name__ == "__main__":
    main()
