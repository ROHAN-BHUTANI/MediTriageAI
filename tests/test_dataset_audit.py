import pytest
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@pytest.fixture
def mock_dataset_csv(tmp_path):
    # Construct a small realistic mock dataset
    data = [
        {
            "tracking_id": f"t_{i}",
            "seed_id": i // 2,
            "variant_index": i % 2,
            "is_perturbed": False,
            "language": "en" if i % 2 == 0 else "hinglish",
            "text": (
                "SUBJECTIVE: Patient complains of severe cardiac chest pain and shortness of breath."
                if i % 2 == 0 else
                "Mera bahut chest pain ho rha he, cardiac problem ki shikayat he."
            ),
            "raw_medical_specialty": "Cardiovascular / Pulmonary",
            "department_code": "CARDIO_PULM" if i < 15 else "GI", # Introduce noisy boundary labels (similarity in GI)
            "routing_confidence": "high",
            "severity_heuristic": "S2",
            "severity_label_source": "regex_heuristic_v0",
            "severity_confidence": "high",
            "split": "train" if i < 12 else ("val" if i < 16 else "test")
        }
        for i in range(20)
    ]
    # Inject exact duplicates
    data[1]["text"] = data[0]["text"]
    
    # Inject extremely similar texts in different classes (noisy label boundary)
    data[16]["text"] = "SUBJECTIVE: Patient complains of severe cardiac chest pain and stomach acidity."
    data[16]["department_code"] = "GI"
    
    df = pd.DataFrame(data)
    csv_path = tmp_path / "mock_dataset.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def temp_output_dir(tmp_path):
    return tmp_path / "output"

def test_cli_help():
    # Verify that CLI help runs and returns code 0
    cmd = [sys.executable, "scripts/dataset_audit.py", "--help"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0
    assert "Override dataset path" in res.stdout

def test_audit_execution(mock_dataset_csv, temp_output_dir):
    # Run the audit engine on our mock dataset
    cmd = [
        sys.executable,
        "scripts/dataset_audit.py",
        "--dataset", mock_dataset_csv,
        "--output-dir", str(temp_output_dir)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"Stdout: {res.stdout}\nStderr: {res.stderr}"
    
    # Check that directories exist
    # Verify we have latest/ and historical timestamped directory
    assert temp_output_dir.exists()
    latest_dir = temp_output_dir / "latest"
    assert latest_dir.exists()
    
    # Find the timestamped folder
    timestamp_dirs = [d for d in temp_output_dir.iterdir() if d.is_dir() and d.name != "latest"]
    assert len(timestamp_dirs) == 1
    run_dir = timestamp_dirs[0]
    
    # List of 21 planned artifacts
    required_artifacts = [
        "dataset_summary.json",
        "class_distribution.csv",
        "label_statistics.csv",
        "duplicate_texts.csv",
        "near_duplicate_texts.csv",
        "multilingual_statistics.csv",
        "vocabulary_statistics.json",
        "text_length_distribution.csv",
        "empty_or_short_samples.csv",
        "long_samples.csv",
        "outlier_samples.csv",
        "noisy_labels.csv",
        "class_token_statistics.json",
        "class_similarity_matrix.csv",
        "class_similarity_heatmap.png",
        "augmentation_recommendations.json",
        "dataset_merge_recommendations.md",
        "hard_negative_candidates.csv",
        "split_leakage_report.csv",
        "split_similarity_matrix.csv",
        "dataset_audit_report.md",
        "artifact_index.md"
    ]
    
    # Check that all files exist in BOTH folders
    for f in required_artifacts:
        assert (run_dir / f).exists(), f"Missing {f} in timestamped folder"
        assert (latest_dir / f).exists(), f"Missing {f} in latest/ folder"
        
    # Check JSON validity
    json_files = [
        "dataset_summary.json",
        "vocabulary_statistics.json",
        "class_token_statistics.json",
        "augmentation_recommendations.json"
    ]
    for jf in json_files:
        with open(latest_dir / jf, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict)
            
    # Check specific fields in dataset_summary.json
    with open(latest_dir / "dataset_summary.json", "r") as f:
        summary = json.load(f)
        assert summary["dataset_size"] == 20
        assert summary["number_of_classes"] == 2
        assert "dataset_health_score" in summary
        
    # Check CSV files are populated and have proper columns
    df_class = pd.read_csv(latest_dir / "class_distribution.csv")
    assert not df_class.empty
    assert "class_name" in df_class.columns
    assert "count" in df_class.columns
    
    df_entity = pd.read_csv(latest_dir / "medical_entity_frequencies.csv")
    assert not df_entity.empty
    assert "entity" in df_entity.columns
    assert "frequency" in df_entity.columns
    
    # Check Markdown report headings
    with open(latest_dir / "dataset_audit_report.md", "r") as f:
        report = f.read()
        assert "## 1. Executive Summary" in report
        assert "## 2. Dataset Health Score" in report
        assert "## 3. Major Findings" in report
        assert "## 4. Class Imbalance Analysis" in report
        assert "## 5. Duplicate Analysis" in report
        assert "## 6. Language Analysis" in report
        assert "## 7. Label Quality Analysis" in report
        assert "## 8. Split Leakage Audit Summary" in report
        assert "## 9. Augmentation Recommendations" in report
        assert "## 10. Recommendations" in report
        assert "## 11. Automatic Training Recommendations" in report
        
    # Check log file was updated
    assert Path("logs/dataset_audit.log").exists()
