"""Generate DATASET_INVENTORY.md based on downloaded files in raw/."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
META = ROOT / "metadata"
INV_PATH = ROOT / "DATASET_INVENTORY.md"

def count(d):
    n, s = 0, 0
    if not d.exists(): return 0, 0
    for p in d.rglob("*"):
        if p.is_file(): n += 1; s += p.stat().st_size
    return n, s

# Define expected datasets and statuses
expected = {
    "mtsamples": {"desc": "Medical Transcriptions", "lic": "CC0"},
    "pmc_patients": {"desc": "PMC-Patients dataset", "lic": "CC BY-NC-SA 4.0"},
    "chatdoctor_healthcaremagic": {"desc": "ChatDoctor HealthCareMagic", "lic": "Research Use Only"},
    "chatdoctor_icliniq": {"desc": "ChatDoctor iCliniq", "lic": "Research Use Only"},
    "meddialog_en": {"desc": "MedDialog English", "lic": "Research Use Only"},
    "fedmml_ed_triage": {"desc": "FedMML ED Triage", "lic": "CC BY 4.0"},
    "nhamcs_ed": {"desc": "NHAMCS ED Data (CDC)", "lic": "US Government Public Domain"},
    "neiss": {"desc": "NEISS Data (CPSC)", "lic": "US Government Public Domain"},
    "l3cube_code_mixed": {"desc": "L3Cube Code-Mixed NLP", "lic": "MIT / CC BY 4.0"},
    "medical_meadow_medqa": {"desc": "Medical Meadow MedQA", "lic": "Open Access"},
    "symptom2disease": {"desc": "Symptom2Disease", "lic": "Open Access"},
    "medqa_usmle": {"desc": "MedQA USMLE", "lic": "Open Access"},
    "disease_symptom_description": {"desc": "Disease Symptom Description", "lic": "Open Access"}
}

downloaded = []
skipped = []
failed = []

for name, info in expected.items():
    d = RAW / name
    fc, tb = count(d)
    
    status = "FAILED"
    reason = "Could not download automatically."
    
    if fc > 0:
        # Check if it was skipped (already exists) or freshly downloaded
        if name == "meddialog_en" and any(p.name == "DOWNLOAD_INSTRUCTIONS.txt" for p in d.iterdir()):
            status = "SKIPPED"
            reason = "Requires manual download from Google Drive."
            skipped.append((name, info, reason))
        elif name == "neiss":
            status = "SKIPPED"
            reason = "Requires interactive query at cpsc.gov portal."
            skipped.append((name, info, reason))
        else:
            status = "DOWNLOADED"
            downloaded.append((name, info, fc, tb))
    else:
        failed.append((name, info, reason))

with open(INV_PATH, "w", encoding="utf-8") as f:
    f.write("# MediTriageAI — Dataset Inventory Report\n\n")
    f.write(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
    f.write("---\n\n")

    f.write("## Summary\n\n")
    f.write(f"| Status | Count |\n")
    f.write(f"|:---|:---:|\n")
    f.write(f"| ✅ Downloaded | {len(downloaded)} |\n")
    f.write(f"| ⏭️ Skipped (Manual Access) | {len(skipped)} |\n")
    f.write(f"| ❌ Failed | {len(failed)} |\n")
    f.write(f"| **Total Evaluated** | **{len(expected)}** |\n\n")

    f.write("---\n\n## ✅ Successfully Downloaded Datasets\n\n")
    if downloaded:
        f.write("| Dataset | License | Files | Size | Description |\n")
        f.write("|:---|:---|:---:|:---:|:---|\n")
        for name, info, fc, tb in downloaded:
            size_str = f"{tb:,} bytes"
            f.write(f"| **{name}** | {info['lic']} | {fc} | {size_str} | {info['desc']} |\n")
    else:
        f.write("*No datasets downloaded.*\n")
    f.write("\n")

    f.write("---\n\n## ⏭️ Skipped Datasets (Manual Required)\n\n")
    if skipped:
        f.write("| Dataset | Reason |\n")
        f.write("|:---|:---|\n")
        for name, info, reason in skipped:
            f.write(f"| {name} | {reason} |\n")
    else:
        f.write("*None.*\n")
    f.write("\n")

    f.write("---\n\n## ❌ Failed Downloads\n\n")
    if failed:
        for name, info, reason in failed:
            f.write(f"### {name}\n")
            f.write(f"- **Description:** {info['desc']}\n")
            f.write(f"- **License:** {info['lic']}\n")
            f.write(f"- **Reason:** {reason}\n\n")
    else:
        f.write("*None.*\n")
    f.write("\n")

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

print(f"Inventory written to {INV_PATH}")
