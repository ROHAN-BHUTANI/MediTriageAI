"""MediTriageAI Data Acquisition - Retry with corrected URLs."""
from __future__ import annotations
import hashlib, json, os, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
META = ROOT / "metadata"
LICENSES = ROOT / "licenses"
LOGS = ROOT / "download_logs"
for d in [RAW, META, LICENSES, LOGS]: d.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    try: print(line)
    except UnicodeEncodeError: print(line.encode('ascii','replace').decode('ascii'))

def dl(url, dest, retries=3, timeout=300):
    for i in range(1, retries+1):
        try:
            log(f"  GET ({i}/{retries}): {url}")
            req = Request(url, headers={"User-Agent":"MediTriageAI/1.0"})
            data = urlopen(req, timeout=timeout).read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f: f.write(data)
            log(f"  OK: {len(data):,} bytes -> {dest.name}")
            return data
        except Exception as e:
            log(f"  FAIL ({i}): {e}")
            if i < retries: time.sleep(2*i)
    return None

def sha256(data): return hashlib.sha256(data).hexdigest()

def count(d):
    n, s = 0, 0
    for p in d.rglob("*"):
        if p.is_file(): n += 1; s += p.stat().st_size
    return n, s

def write_meta(name, url, lic, fc, tb, chk="", ver="1.0"):
    with open(META / f"{name}.json", "w") as f:
        json.dump({"name":name, "source_url":url, "download_date":datetime.now(timezone.utc).isoformat(),
                    "license":lic, "version":ver, "checksum":chk, "files_count":fc, "total_bytes":tb}, f, indent=2)

def src(d, url):
    with open(d / "SOURCE_URL.txt", "w") as f: f.write(url+"\n")

def lic_file(name, text):
    with open(LICENSES / f"{name}_LICENSE.txt", "w") as f: f.write(text)

results = []

# ========== 1. MTSamples ==========
log("\n== MTSamples ==")
d = RAW / "mtsamples"; d.mkdir(parents=True, exist_ok=True)
if not (d / "mtsamples.csv").exists():
    urls = [
        "https://raw.githubusercontent.com/wiki-yu/medical-report-classification/main/data/mtsamples.csv",
        "https://raw.githubusercontent.com/ShantanuSS/NLP-Medical-Transcriptions/master/mtsamples.csv",
        "https://raw.githubusercontent.com/dsgiitr/Medical-NLP/master/data/mtsamples.csv",
    ]
    data = None
    for u in urls:
        data = dl(u, d / "mtsamples.csv")
        if data: break
    if data:
        src(d, urls[0]); lic_file("mtsamples", "CC0 Public Domain\nSource: Kaggle/mtsamples.com\n")
        fc, tb = count(d); write_meta("mtsamples", urls[0], "CC0", fc, tb, sha256(data))
        results.append(("mtsamples", "DOWNLOADED", fc, tb))
    else:
        results.append(("mtsamples", "FAILED", 0, 0))
else:
    fc, tb = count(d); results.append(("mtsamples", "EXISTS", fc, tb))
    log("  Already exists")

# ========== 2. PMC-Patients ==========
log("\n== PMC-Patients ==")
d = RAW / "pmc_patients"; d.mkdir(parents=True, exist_ok=True)
if not any(d.glob("*.csv")) and not any(d.glob("*.json")):
    # PMC-Patients uses Xet for large files; try the CSV
    url = "https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients.csv"
    data = dl(url, d / "PMC-Patients.csv", timeout=600)
    if not data:
        url2 = "https://huggingface.co/datasets/zhengyun21/PMC-Patients/resolve/main/PMC-Patients-V2.json"
        data = dl(url2, d / "PMC-Patients-V2.json", timeout=600)
    if data:
        src(d, "https://huggingface.co/datasets/zhengyun21/PMC-Patients")
        lic_file("pmc_patients", "CC BY-NC-SA 4.0\nZhao et al. 2023, Scientific Data\n")
        fc, tb = count(d); write_meta("pmc_patients", url, "CC BY-NC-SA 4.0", fc, tb, sha256(data))
        results.append(("pmc_patients", "DOWNLOADED", fc, tb))
    else:
        # Record as needing huggingface-hub library
        results.append(("pmc_patients", "FAILED-NEEDS-HF-HUB", 0, 0))
        log("  Requires huggingface_hub (Xet storage); manual download needed")
else:
    fc, tb = count(d); results.append(("pmc_patients", "EXISTS", fc, tb))

# ========== 3. ChatDoctor HealthCareMagic ==========
log("\n== ChatDoctor-HealthCareMagic ==")
d = RAW / "chatdoctor_healthcaremagic"; d.mkdir(parents=True, exist_ok=True)
if not any(d.iterdir()):
    url = "https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k/resolve/main/data/train-00000-of-00001-5e7cb295b9cff0bf.parquet"
    data = dl(url, d / "train.parquet", timeout=300)
    if data:
        src(d, "https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k")
        lic_file("chatdoctor_hcm", "Research Use Only\nOriginal: healthcaremagic.com\n")
        fc, tb = count(d); write_meta("chatdoctor_healthcaremagic", url, "Research Use", fc, tb, sha256(data))
        results.append(("chatdoctor_healthcaremagic", "DOWNLOADED", fc, tb))
    else:
        results.append(("chatdoctor_healthcaremagic", "FAILED", 0, 0))
else:
    fc, tb = count(d); results.append(("chatdoctor_healthcaremagic", "EXISTS", fc, tb))

# ========== 4. ChatDoctor iCliniq ==========
log("\n== ChatDoctor-iCliniq ==")
d = RAW / "chatdoctor_icliniq"; d.mkdir(parents=True, exist_ok=True)
if not any(d.iterdir()):
    # Try known parquet filename patterns
    urls = [
        "https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq/resolve/main/data/train-00000-of-00001.parquet",
        "https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq/resolve/main/chatdoctor_icliniq.json",
    ]
    data = None
    for u in urls:
        fn = u.split("/")[-1]
        data = dl(u, d / fn)
        if data: break
    if data:
        src(d, "https://huggingface.co/datasets/lavita/ChatDoctor-iCliniq")
        lic_file("chatdoctor_icliniq", "Research Use Only\nOriginal: icliniq.com\n")
        fc, tb = count(d); write_meta("chatdoctor_icliniq", u, "Research Use", fc, tb, sha256(data))
        results.append(("chatdoctor_icliniq", "DOWNLOADED", fc, tb))
    else:
        results.append(("chatdoctor_icliniq", "FAILED", 0, 0))
        log("  Not directly accessible; may need HF datasets library")
else:
    fc, tb = count(d); results.append(("chatdoctor_icliniq", "EXISTS", fc, tb))

# ========== 5. MedDialog (Google Drive hosted) ==========
log("\n== MedDialog EN ==")
d = RAW / "meddialog_en"; d.mkdir(parents=True, exist_ok=True)
if not any(d.iterdir()):
    # MedDialog requires Google Drive manual download
    # Record this as requiring manual intervention
    results.append(("meddialog_en", "REQUIRES-MANUAL", 0, 0))
    src(d, "https://github.com/UCSD-AI4H/Medical-Dialogue-System")
    with open(d / "DOWNLOAD_INSTRUCTIONS.txt", "w") as f:
        f.write("MedDialog English requires manual download from Google Drive:\n")
        f.write("https://drive.google.com/drive/folders/1g29ssimdZ6JzTST6Y8g6h-ogUNReBtJD\n\n")
        f.write("After downloading, place the files in this directory.\n")
        f.write("Then use: load_dataset('medical_dialog', name='en', data_dir='<path>')\n")
    log("  Requires manual Google Drive download; instructions saved")
else:
    fc, tb = count(d); results.append(("meddialog_en", "EXISTS", fc, tb))

# ========== 6. FedMML ED Triage ==========
log("\n== FedMML ED Triage ==")
d = RAW / "fedmml_ed_triage"; d.mkdir(parents=True, exist_ok=True)
if not any(p for p in d.iterdir() if p.suffix in ('.parquet','.csv')):
    # Try HF API to discover files
    api_url = "https://huggingface.co/api/datasets/olaflaitinen/fedmml-ed-triage/tree/main"
    try:
        req = Request(api_url, headers={"User-Agent":"MediTriageAI/1.0"})
        resp = urlopen(req, timeout=30)
        tree = json.loads(resp.read())
        parquets = [f["path"] for f in tree if f["path"].endswith(".parquet")]
        if parquets:
            any_ok = False
            for p in parquets:
                url = f"https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage/resolve/main/{p}"
                fname = p.replace("/", "_")
                data = dl(url, d / fname, timeout=300)
                if data: any_ok = True
            if any_ok:
                src(d, "https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage")
                lic_file("fedmml_ed_triage", "CC BY 4.0\nSynthetic ED triage data\n")
                fc, tb = count(d); write_meta("fedmml_ed_triage", api_url, "CC BY 4.0", fc, tb)
                results.append(("fedmml_ed_triage", "DOWNLOADED", fc, tb))
            else:
                results.append(("fedmml_ed_triage", "FAILED", 0, 0))
        else:
            log("  No parquet files found in repo tree")
            results.append(("fedmml_ed_triage", "FAILED-NO-FILES", 0, 0))
    except Exception as e:
        log(f"  API tree listing failed: {e}")
        results.append(("fedmml_ed_triage", "FAILED", 0, 0))
else:
    fc, tb = count(d); results.append(("fedmml_ed_triage", "EXISTS", fc, tb))

# ========== 7. MedQA (GBQ/Jin et al.) ==========
log("\n== MedQA ==")
d = RAW / "medqa_usmle"; d.mkdir(parents=True, exist_ok=True)
if not any(d.iterdir()):
    # Try the original GBQ GitHub release
    url = "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/main/data/train-00000-of-00001.parquet"
    data = dl(url, d / "train.parquet", timeout=300)
    if data:
        for split in ["test", "validation"]:
            u2 = url.replace("train", split)
            dl(u2, d / f"{split}.parquet", timeout=300)
        src(d, "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options")
        lic_file("medqa_usmle", "Open Access\nJin et al. 2021\n")
        fc, tb = count(d); write_meta("medqa_usmle", url, "Open", fc, tb, sha256(data))
        results.append(("medqa_usmle", "DOWNLOADED", fc, tb))
    else:
        results.append(("medqa_usmle", "FAILED", 0, 0))
else:
    fc, tb = count(d); results.append(("medqa_usmle", "EXISTS", fc, tb))

# ========== 8. Disease Symptom Description Dataset ==========
log("\n== Disease Symptom Description ==")
d = RAW / "disease_symptom_description"; d.mkdir(parents=True, exist_ok=True)
if not any(d.iterdir()):
    url = "https://raw.githubusercontent.com/itratrahman/disease_prediction/master/dataset.csv"
    data = dl(url, d / "dataset.csv")
    if data:
        src(d, url)
        lic_file("disease_symptom", "Open/Research\nGitHub: itratrahman/disease_prediction\n")
        fc, tb = count(d); write_meta("disease_symptom_description", url, "Open", fc, tb, sha256(data))
        results.append(("disease_symptom_description", "DOWNLOADED", fc, tb))
    else:
        results.append(("disease_symptom_description", "FAILED", 0, 0))
else:
    fc, tb = count(d); results.append(("disease_symptom_description", "EXISTS", fc, tb))

# ========== Print Summary ==========
log("\n" + "="*60)
log("RETRY ACQUISITION SUMMARY")
log("="*60)
for name, status, *rest in results:
    fc = rest[0] if rest else 0
    tb = rest[1] if len(rest) > 1 else 0
    log(f"  {status:25s} | {name} ({fc} files, {tb:,} bytes)")
log("="*60)
