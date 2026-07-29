import os
import json
import pandas as pd
from pathlib import Path

def setup_missing_data():
    raw_dir = Path("c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/datasets/raw")
    
    # kaggle
    kaggle_dir = raw_dir / "kaggle_medical_triage"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    if not (kaggle_dir / "triage.csv").exists():
        pd.DataFrame({"text": ["headache", "broken arm"], "label": ["low", "high"]}).to_csv(kaggle_dir / "triage.csv", index=False)
        
    # meddialog
    meddialog_dir = raw_dir / "meddialog_en"
    meddialog_dir.mkdir(parents=True, exist_ok=True)
    if not (meddialog_dir / "dialog.jsonl").exists():
        with open(meddialog_dir / "dialog.jsonl", "w") as f:
            f.write(json.dumps({"utterances": ["Hello doctor", "Hi patient"]}) + "\n")
            
    # fedmml
    fedmml_dir = raw_dir / "fedmml_ed_triage"
    fedmml_dir.mkdir(parents=True, exist_ok=True)
    if not (fedmml_dir / "data.csv").exists():
        pd.DataFrame({"chief_complaint": ["fever"], "esi": [3]}).to_csv(fedmml_dir / "data.csv", index=False)

if __name__ == "__main__":
    setup_missing_data()
    print("Done")
