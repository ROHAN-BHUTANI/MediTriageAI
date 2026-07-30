"""MediTriageAI Data Acquisition - HF Datasets."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
META = ROOT / "metadata"
LICENSES = ROOT / "licenses"


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def count(d):
    n, s = 0, 0
    for p in d.rglob("*"):
        if p.is_file():
            n += 1
            s += p.stat().st_size
    return n, s


def write_meta(name, url, lic, fc, tb, chk="", ver="1.0"):
    with open(META / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": name,
                "source_url": url,
                "download_date": datetime.now(timezone.utc).isoformat(),
                "license": lic,
                "version": ver,
                "checksum": chk,
                "files_count": fc,
                "total_bytes": tb,
            },
            f,
            indent=2,
        )


def src(d, url):
    with open(d / "SOURCE_URL.txt", "w", encoding="utf-8") as f:
        f.write(url + "\n")


def lic_file(name, text):
    with open(LICENSES / f"{name}_LICENSE.txt", "w", encoding="utf-8") as f:
        f.write(text)


try:
    from datasets import load_dataset
except ImportError:
    log("HF datasets library not installed. Exiting.")
    sys.exit(1)

# List of datasets to download
# Tuple of (name, hf_repo, config_name, license, description)
hf_datasets = [
    (
        "chatdoctor_icliniq",
        "lavita/ChatDoctor-iCliniq",
        None,
        "Research Use Only",
        "Original data from icliniq.com",
    ),
    (
        "fedmml_ed_triage",
        "olaflaitinen/fedmml-ed-triage",
        None,
        "CC BY 4.0",
        "Synthetic ED triage encounters with ESI levels",
    ),
    (
        "symptom2disease",
        "QuyenAnhDE/Symptom2Disease",
        None,
        "Open Access",
        "Symptom text to disease label mapping",
    ),
    (
        "medqa_usmle",
        "bigbio/med_qa",
        "med_qa_en_source",
        "Open Access",
        "USMLE-style medical QA",
    ),
]

for name, repo, config, lic, desc in hf_datasets:
    log(f"\n== {name} ==")
    d = RAW / name
    d.mkdir(parents=True, exist_ok=True)
    if not any(d.iterdir()):
        try:
            log(f"Loading {repo} from Hugging Face...")
            ds = load_dataset(repo, config)
            for split in ds.keys():
                out_path = d / f"{split}.jsonl"
                log(f"Saving split {split} to {out_path}...")
                ds[split].to_json(out_path)

            url = f"https://huggingface.co/datasets/{repo}"
            src(d, url)
            lic_file(name, f"{lic}\n{desc}\nSource: {url}\n")
            fc, tb = count(d)
            write_meta(name, url, lic, fc, tb)
            log(f"DOWNLOADED {name} ({fc} files, {tb} bytes)")
        except Exception as e:
            log(f"FAILED {name}: {e}")
    else:
        log(f"EXISTS {name}")
