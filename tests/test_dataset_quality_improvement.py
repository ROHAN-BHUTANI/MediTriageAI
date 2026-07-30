import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def mock_dataset_csv(tmp_path):
    # Construct a small dataset with duplicates, short text, missing labels, and normalizations needed
    data = [
        # Normal row
        {
            "tracking_id": "T1",
            "seed_id": 1,
            "variant_index": 0,
            "is_perturbed": False,
            "language": "en",
            "text": "SUBJECTIVE: Patient has severe abdominal pain and vomiting since yesterday morning.",
            "raw_medical_specialty": "Gastroenterology",
            "department_code": "GI",
            "routing_confidence": "high",
            "severity_heuristic": "S2",
            "severity_label_source": "regex_heuristic_v0",
            "severity_confidence": "high",
            "split": "train",
        },
        # Exact duplicate of T1
        {
            "tracking_id": "T2",
            "seed_id": 1,
            "variant_index": 1,
            "is_perturbed": True,
            "language": "en",
            "text": "SUBJECTIVE: Patient has severe abdominal pain and vomiting since yesterday morning.",
            "raw_medical_specialty": "Gastroenterology",
            "department_code": "GI",
            "routing_confidence": "high",
            "severity_heuristic": "S2",
            "severity_label_source": "regex_heuristic_v0",
            "severity_confidence": "high",
            "split": "train",
        },
        # Row needing whitespace collapse and unicode normalisation
        {
            "tracking_id": "T3",
            "seed_id": 2,
            "variant_index": 0,
            "is_perturbed": False,
            "language": "en",
            "text": "SUBJECTIVE:  Patient  presents with   cough  and  fever—high temp.   ",
            "raw_medical_specialty": "General Medicine",
            "department_code": "GEN_MED",
            "routing_confidence": "high",
            "severity_heuristic": "S3",
            "severity_label_source": "regex_heuristic_v0",
            "severity_confidence": "high",
            "split": "train",
        },
        # Short text complaint (to be removed)
        {
            "tracking_id": "T4",
            "seed_id": 3,
            "variant_index": 0,
            "is_perturbed": False,
            "language": "en",
            "text": "Short.",
            "raw_medical_specialty": "General Medicine",
            "department_code": "GEN_MED",
            "routing_confidence": "high",
            "severity_heuristic": "S5",
            "severity_label_source": "regex_heuristic_v0",
            "severity_confidence": "high",
            "split": "train",
        },
        # Missing label (to be removed)
        {
            "tracking_id": "T5",
            "seed_id": 4,
            "variant_index": 0,
            "is_perturbed": False,
            "language": "en",
            "text": "SUBJECTIVE: Patient has difficulty breathing, chest tightness, and rapid heart rate.",
            "raw_medical_specialty": "Cardiovascular / Pulmonary",
            "department_code": None,  # Missing specialist label
            "routing_confidence": "high",
            "severity_heuristic": "S2",
            "severity_label_source": "regex_heuristic_v0",
            "severity_confidence": "high",
            "split": "val",
        },
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_audit_dir(tmp_path):
    audit_path = tmp_path / "audit"
    audit_path.mkdir()

    # 1. duplicates csv
    dup_df = pd.DataFrame(
        [
            {
                "text": "SUBJECTIVE: Patient has severe abdominal pain and vomiting since yesterday morning.",
                "count": 2,
                "tracking_ids": "T1,T2",
                "classes": "GI,GI",
            }
        ]
    )
    dup_df.to_csv(audit_path / "duplicate_texts.csv", index=False)

    # 2. empty class distribution
    class_dist_df = pd.DataFrame(
        [
            {"class_name": "GI", "count": 2, "percentage": 40.0},
            {"class_name": "GEN_MED", "count": 2, "percentage": 40.0},
            {"class_name": "CARDIO_PULM", "count": 1, "percentage": 20.0},
        ]
    )
    class_dist_df.to_csv(audit_path / "class_distribution.csv", index=False)

    # 3. empty csv placeholders
    empty_df = pd.DataFrame()
    empty_df.to_csv(audit_path / "near_duplicate_texts.csv", index=False)
    empty_df.to_csv(audit_path / "noisy_labels.csv", index=False)
    empty_df.to_csv(audit_path / "hard_negative_candidates.csv", index=False)

    # 4. JSON configs
    with open(audit_path / "augmentation_recommendations.json", "w") as f:
        json.dump(
            {
                "GI": {
                    "recommended_augmentations": ["typo augmentation", "paraphrases"]
                },
                "GEN_MED": {"recommended_augmentations": ["ASR augmentation"]},
                "CARDIO_PULM": {
                    "recommended_augmentations": [
                        "more English samples",
                        "Hindi samples",
                        "Hinglish samples",
                    ]
                },
            },
            f,
        )

    with open(audit_path / "class_token_statistics.json", "w") as f:
        json.dump({}, f)

    with open(audit_path / "regex_coverage_statistics.json", "w") as f:
        json.dump({}, f)

    return str(audit_path)


@pytest.fixture
def temp_output_dir(tmp_path):
    return tmp_path / "improved_output"


def test_dry_run_mode(mock_dataset_csv, mock_audit_dir, temp_output_dir):
    cmd = [
        sys.executable,
        "scripts/dataset_quality_improvement.py",
        "--dataset",
        mock_dataset_csv,
        "--audit-dir",
        mock_audit_dir,
        "--output-dir",
        str(temp_output_dir),
        "--dry-run",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0

    # Dry run should write reports but NOT write dataset_improved.csv
    assert temp_output_dir.exists()
    assert not (temp_output_dir / "dataset_improved.csv").exists()
    assert (temp_output_dir / "dry_run_dataset_diff_report.md").exists()
    assert (temp_output_dir / "dry_run_change_log.csv").exists()
    assert (temp_output_dir / "dry_run_label_review_candidates.csv").exists()
    assert (temp_output_dir / "dry_run_improvement_manifest.json").exists()


def test_production_improvement_run(mock_dataset_csv, mock_audit_dir, temp_output_dir):
    cmd = [
        sys.executable,
        "scripts/dataset_quality_improvement.py",
        "--dataset",
        mock_dataset_csv,
        "--audit-dir",
        mock_audit_dir,
        "--output-dir",
        str(temp_output_dir),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"Stderr: {res.stderr}"

    # Verify cleaned dataset exists
    improved_csv = temp_output_dir / "dataset_improved.csv"
    assert improved_csv.exists()

    df_clean = pd.read_csv(improved_csv)
    # T2 (duplicate), T4 (short), and T5 (missing label) should be removed.
    # Out of 5 original rows, 3 removed -> 2 should remain (T1, T3).
    assert len(df_clean) == 2

    # Check that T3 was normalized: whitespaces collapsed and em-dash normalized
    t3_row = df_clean[df_clean["tracking_id"] == "T3"].iloc[0]
    # "SUBJECTIVE: Patient presents with cough and fever-high temp."
    assert (
        t3_row["text"] == "SUBJECTIVE: Patient presents with cough and fever-high temp."
    )

    # Verify other reports exist
    assert (temp_output_dir / "rollback_manifest.json").exists()
    assert (temp_output_dir / "dataset_lineage.json").exists()
    assert (temp_output_dir / "change_log.csv").exists()
    assert (temp_output_dir / "label_review_candidates.csv").exists()
    assert (temp_output_dir / "augmentation_plan.json").exists()
    assert (temp_output_dir / "dataset_merge_plan.md").exists()
    assert (temp_output_dir / "dataset_diff_report.md").exists()
    assert (temp_output_dir / "quality_improvement_report.md").exists()
    assert (temp_output_dir / "improvement_manifest.json").exists()

    # Verify rollback manifest structure
    with open(temp_output_dir / "rollback_manifest.json", "r") as f:
        rollback = json.load(f)
        assert len(rollback["deleted_records"]) == 3  # T2, T4, T5
        assert len(rollback["modified_records"]) == 1  # T3

    # Verify dataset lineage tracking counts
    with open(temp_output_dir / "dataset_lineage.json", "r") as f:
        lineage = json.load(f)
        assert lineage[0]["stage"] == "Input Dataset"
        assert lineage[0]["inflow_count"] == 5
        assert lineage[1]["stage"] == "Cleaning"
        assert lineage[1]["inflow_count"] == 5
        assert lineage[1]["outflow_count"] == 2
        assert lineage[2]["stage"] == "Normalization"
        assert lineage[2]["inflow_count"] == 2
        assert lineage[2]["outflow_count"] == 2
        assert lineage[3]["stage"] == "Label Review"
        assert lineage[3]["inflow_count"] == 2
        assert lineage[3]["outflow_count"] == 2
        assert lineage[4]["stage"] == "Export"
        assert lineage[4]["inflow_count"] == 2
        assert lineage[4]["outflow_count"] == 2

    # Verify quality improvement report exists and contains metrics
    with open(temp_output_dir / "quality_improvement_report.md", "r") as f:
        quality_rep = f.read()
        assert "## 1. Dataset Quality Score Metrics Breakdown" in quality_rep
        assert "Completeness" in quality_rep
        assert "Consistency" in quality_rep
        assert "Uniqueness" in quality_rep

    # Verify change log CSV content
    change_log_df = pd.read_csv(temp_output_dir / "change_log.csv")
    assert not change_log_df.empty
    assert "tracking_id" in change_log_df.columns
    assert "operation_applied" in change_log_df.columns

    # Cleaned rows count in change log
    assert (change_log_df["operation_applied"] == "REMOVE").sum() == 3
    assert (change_log_df["operation_applied"] == "CLEAN").sum() == 1
