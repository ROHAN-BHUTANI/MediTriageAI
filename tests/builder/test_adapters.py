import tempfile
from pathlib import Path

import pandas as pd
import pytest

from meditriage.builder.adapters.mtsamples import MTSamplesAdapter


def test_mtsamples_adapter_metadata():
    adapter = MTSamplesAdapter()
    assert adapter.dataset_source == "mtsamples"
    assert adapter.version == "1.1.0"


def test_mtsamples_adapter_ingest():
    adapter = MTSamplesAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake raw dataset directory
        raw_path = Path(tmpdir)
        csv_path = raw_path / "mtsamples (1).csv"

        # Write some fake data
        df = pd.DataFrame(
            {
                "Unnamed: 0": [0, 1, 2],
                "description": ["desc0", "desc1", "desc2"],
                "medical_specialty": [
                    " Allergy / Immunology",
                    " Bariatrics",
                    " Cardiovascular / Pulmonary",
                ],
                "sample_name": ["sample0", "sample1", "sample2"],
                "transcription": [
                    "text0",
                    "nan",
                    "text2",
                ],  # row 1 missing transcription, should fallback to desc
                "keywords": ["kw0", "kw1", "kw2"],
            }
        )
        df.set_index("Unnamed: 0", inplace=True)
        df.to_csv(csv_path)

        # Test ingestion with chunksize 2
        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))

        assert len(chunks) == 2  # 3 rows total -> chunks of 2 and 1

        first_chunk = chunks[0]
        assert len(first_chunk) == 2
        assert first_chunk.iloc[0]["raw_text"] == "text0"
        assert first_chunk.iloc[0]["department"] == "ENT_OPHTHALMO"
        assert first_chunk.iloc[0]["dataset_source"] == "mtsamples"

        # Fallback transcription test
        assert first_chunk.iloc[1]["raw_text"] == "desc1"
        assert first_chunk.iloc[1]["department"] == "GI"

        second_chunk = chunks[1]
        assert len(second_chunk) == 1
        assert second_chunk.iloc[0]["raw_text"] == "text2"
        assert second_chunk.iloc[0]["department"] == "CARDIO_PULM"


def test_mtsamples_adapter_ingest_empty_skip():
    adapter = MTSamplesAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir)
        csv_path = raw_path / "mtsamples (1).csv"

        df = pd.DataFrame(
            {
                "Unnamed: 0": [0],
                "description": ["nan"],
                "medical_specialty": ["nan"],
                "transcription": ["nan"],
            }
        )
        df.set_index("Unnamed: 0", inplace=True)
        df.to_csv(csv_path)

        chunks = list(adapter.ingest(str(raw_path)))
        assert len(chunks) == 0  # The only row is empty and should be skipped


from meditriage.builder.adapters.pmc_patients import PMCPatientsAdapter


def test_pmc_patients_adapter_metadata():
    adapter = PMCPatientsAdapter()
    assert adapter.dataset_source == "pmc_patients"
    assert adapter.version == "1.1.0"


def test_pmc_patients_adapter_ingest():
    adapter = PMCPatientsAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir)
        csv_path = raw_path / "PMC-Patients.csv"

        df = pd.DataFrame({"patient": ["text0", "nan", "", "text3"]})
        df.to_csv(csv_path, index=False)

        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))

        # We expect 2 chunks, because chunk 1 (text0, nan) -> 1 record, chunk 2 (, text3) -> 1 record
        assert len(chunks) == 2

        first_chunk = chunks[0]
        assert len(first_chunk) == 1
        assert first_chunk.iloc[0]["raw_text"] == "text0"

        second_chunk = chunks[1]
        assert len(second_chunk) == 1
        assert second_chunk.iloc[0]["raw_text"] == "text3"


from meditriage.builder.adapters.medqa_usmle import MedqaUsmleAdapter


def test_medqa_usmle_adapter_metadata():
    adapter = MedqaUsmleAdapter()
    assert adapter.dataset_source == "medqa_usmle"
    assert adapter.version == "1.0.0"


def test_medqa_usmle_adapter_ingest():
    adapter = MedqaUsmleAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir)
        jsonl_dir = raw_path / "data_clean" / "data_clean" / "questions" / "US"
        jsonl_dir.mkdir(parents=True)
        jsonl_path = jsonl_dir / "US_qbank.jsonl"

        df = pd.DataFrame({"question": ["text0", "nan", "", "text3"]})
        df.to_json(jsonl_path, orient="records", lines=True)

        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))

        assert len(chunks) == 2

        first_chunk = chunks[0]
        assert len(first_chunk) == 1
        assert first_chunk.iloc[0]["raw_text"] == "text0"

        second_chunk = chunks[1]
        assert len(second_chunk) == 1
        assert second_chunk.iloc[0]["raw_text"] == "text3"


import json

from meditriage.builder.adapters.medical_meadow_medqa import MedicalMeadowMedqaAdapter


def test_medical_meadow_medqa_adapter_metadata():
    adapter = MedicalMeadowMedqaAdapter()
    assert adapter.dataset_source == "medical_meadow_medqa"
    assert adapter.version == "1.0.0"


def test_medical_meadow_medqa_adapter_ingest():
    adapter = MedicalMeadowMedqaAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir)
        json_path = raw_path / "medical_meadow_medqa.json"

        data = [
            {"input": "text0", "instruction": "ignore"},
            {"input": "", "instruction": "inst1"},
            {"input": "", "instruction": ""},
            {"input": "text3", "instruction": "inst3"},
        ]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))

        assert len(chunks) == 2

        first_chunk = chunks[0]
        assert len(first_chunk) == 2
        assert first_chunk.iloc[0]["raw_text"] == "text0"
        assert first_chunk.iloc[1]["raw_text"] == "inst1"

        second_chunk = chunks[1]
        assert len(second_chunk) == 1
        assert second_chunk.iloc[0]["raw_text"] == "text3"


from meditriage.builder.adapters.chatdoctor_healthcaremagic import (
    ChatDoctorHealthcareMagicAdapter,
)
from meditriage.builder.adapters.chatdoctor_icliniq import ChatDoctorIcliniqAdapter
from meditriage.builder.adapters.fedmml_ed_triage import FedmmlEdTriageAdapter
from meditriage.builder.adapters.kaggle_medical_triage import KaggleMedicalTriageAdapter
from meditriage.builder.adapters.l3cube_code_mixed import L3CubeCodeMixedAdapter
from meditriage.builder.adapters.meddialog_en import MeddialogEnAdapter
from meditriage.builder.adapters.neiss import NeissAdapter
from meditriage.builder.adapters.nhamcs_ed import NhamcsEdAdapter
from meditriage.builder.adapters.symptom2disease import Symptom2DiseaseAdapter


@pytest.fixture
def symptom2disease_adapter():
    return Symptom2DiseaseAdapter()


def test_symptom2disease_adapter_metadata(symptom2disease_adapter):
    assert symptom2disease_adapter.dataset_source == "symptom2disease"
    assert symptom2disease_adapter.version == "1.0.0"


def test_symptom2disease_adapter_ingest():
    adapter = Symptom2DiseaseAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir)
        csv_path = raw_path / "Symptom2Disease.csv"

        df = pd.DataFrame(
            {
                "text": ["text0", "nan", "", "text3", "text4"],
                "label": ["Psoriasis", "disease", "", "disease2", "Typhoid"],
            }
        )
        df.to_csv(csv_path, index=False)

        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))
        df_out = pd.concat(chunks, ignore_index=True)

        assert len(df_out) == 3

        assert df_out.iloc[0]["raw_text"] == "text0"
        assert df_out.iloc[0]["department"] == "ENT_OPHTHALMO"

        assert df_out.iloc[1]["raw_text"] == "text3"
        assert df_out.iloc[1]["department"] == "GEN_MED"  # default for unmapped

        assert df_out.iloc[2]["raw_text"] == "text4"
        assert df_out.iloc[2]["department"] == "GEN_MED"


def test_chatdoctor_healthcaremagic_adapter_metadata():
    adapter = ChatDoctorHealthcareMagicAdapter()
    assert adapter.dataset_source == "chatdoctor_healthcaremagic"
    assert adapter.version == "1.0.0"


def test_chatdoctor_healthcaremagic_adapter_ingest():
    adapter = ChatDoctorHealthcareMagicAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir)
        data_dir = raw_path / "data"
        data_dir.mkdir()
        parquet_path = data_dir / "test.parquet"

        df = pd.DataFrame({"input": ["text0", "nan", "", "text3"]})
        df.to_parquet(parquet_path)

        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))

        assert len(chunks) == 2

        first_chunk = chunks[0]
        assert len(first_chunk) == 1
        assert first_chunk.iloc[0]["raw_text"] == "text0"

        second_chunk = chunks[1]
        assert len(second_chunk) == 1
        assert second_chunk.iloc[0]["raw_text"] == "text3"


def test_chatdoctor_icliniq_adapter_metadata():
    adapter = ChatDoctorIcliniqAdapter()
    assert adapter.dataset_source == "chatdoctor_icliniq"
    assert adapter.version == "1.0.0"


def test_chatdoctor_icliniq_adapter_ingest():
    adapter = ChatDoctorIcliniqAdapter()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir)
        data_dir = raw_path / "data"
        data_dir.mkdir()
        parquet_path = data_dir / "test.parquet"

        df = pd.DataFrame({"input": ["text0", "nan", "", "text3"]})
        df.to_parquet(parquet_path)

        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))

        assert len(chunks) == 2

        first_chunk = chunks[0]
        assert len(first_chunk) == 1
        assert first_chunk.iloc[0]["raw_text"] == "text0"

        second_chunk = chunks[1]
        assert len(second_chunk) == 1
        assert second_chunk.iloc[0]["raw_text"] == "text3"


def test_neiss_adapter_metadata():
    adapter = NeissAdapter()
    assert adapter.dataset_source == "neiss"
    assert adapter.version == "1.0.0"


def test_neiss_adapter(tmp_path):
    adapter = NeissAdapter()

    df = pd.DataFrame(
        {
            "Narrative_1": ["Patient fell down stairs", "Cut finger with knife"],
            "Age": [45, 12],
            "Sex": [1, 2],
            "Race": [1, 0],
        }
    )
    df.to_parquet(tmp_path / "neiss_all.parquet")

    results = list(adapter.ingest(str(tmp_path)))
    assert len(results) == 1
    res = results[0]
    assert len(res) == 2
    assert "triage_level" in res.columns
    assert "department" in res.columns


def test_nhamcs_adapter(tmp_path):
    adapter = NhamcsEdAdapter()

    year_dir = tmp_path / "ed2021"
    year_dir.mkdir()

    line = " " * 3000
    line_chars = list(line)

    import json
    import os

    dict_path = os.path.join(
        os.path.dirname(adapter.__module__.replace(".", "/")), "nhamcs_dict.json"
    )
    with open(dict_path, "r") as f:
        cols = json.load(f)["2021"]

    for col in cols:
        name = col["name"]
        start = col["start"]
        col["length"]
        if name == "AGE":
            line_chars[start : start + 3] = list("045")
        elif name == "SEX":
            line_chars[start : start + 1] = list("1")
        elif name == "IMMEDR":
            line_chars[start : start + 2] = list("03")
        elif name == "RFV1":
            line_chars[start : start + 5] = list("12345")

    with open(year_dir / "ed2021", "w") as f:
        f.write("".join(line_chars) + "\n")

    results = list(adapter.ingest(str(tmp_path)))
    assert len(results) == 1
    res = results[0]
    assert len(res) == 1
    assert "Age: 045" in res.iloc[0]["raw_text"]
    assert "Reason for Visit 1 (Code): 12345" in res.iloc[0]["raw_text"]
    assert res.iloc[0]["triage_level"] == "3"


def test_fedmml_ed_triage_adapter_metadata():
    adapter = FedmmlEdTriageAdapter()
    assert adapter.dataset_source == "fedmml_ed_triage"
    assert adapter.version == "1.0"


def test_fedmml_ed_triage_adapter_ingest(tmp_path):
    adapter = FedmmlEdTriageAdapter()

    df = pd.DataFrame(
        {
            "chief_complaint": ["Chest pain", "Arm pain"],
            "clinical_notes": ["Patient looks sick", "Fell down"],
            "age": [55, 12],
            "sex": ["M", "M"],
            "systolic_bp": [120, 110],
            "diastolic_bp": [80, 70],
            "esi_level": [2, 4],
        }
    )

    df.to_csv(tmp_path / "fedmml_ed_triage_dataset.csv", index=False)

    results = list(adapter.ingest(str(tmp_path)))
    assert len(results) == 1
    res = results[0]
    assert len(res) == 2
    assert "Chest pain" in res.iloc[0]["raw_text"]
    assert "120/80" in res.iloc[0]["raw_text"]
    assert res.iloc[0]["triage_level"] == 2
    assert res.iloc[1]["triage_level"] == 4


def test_kaggle_medical_triage_adapter_metadata():
    adapter = KaggleMedicalTriageAdapter()
    assert adapter.dataset_source == "kaggle_medical_triage"
    assert adapter.version == "2.0"


def test_kaggle_medical_triage_adapter_ingest(tmp_path):
    import json

    adapter = KaggleMedicalTriageAdapter()

    sample = [
        {
            "id": 1,
            "input_text": "Chest pain",
            "symptoms": ["chest pain", "sweating"],
            "urgency_level": 1,
            "urgency_label": "ACIL",
            "reasoning": "Possible myocardial infarction",
            "response": "Call emergency services immediately",
        },
        {
            "id": 2,
            "input_text": "Mild headache",
            "symptoms": ["headache"],
            "urgency_level": 4,
            "urgency_label": "NORMAL",
            "reasoning": "Likely tension headache",
            "response": "Outpatient evaluation",
        },
    ]

    with open(tmp_path / "medical_data.json", "w", encoding="utf-8") as f:
        json.dump(sample, f)

    results = list(adapter.ingest(str(tmp_path)))

    assert len(results) == 1

    df = results[0]

    assert len(df) == 2

    assert "Chest pain" in df.iloc[0]["raw_text"]
    assert df.iloc[0]["triage_level"] == 1
    assert df.iloc[1]["triage_level"] == 4
    assert df.iloc[0]["language"] == "tr"
    assert df.iloc[0]["raw_severity"] == "ACIL"


def test_kaggle_medical_triage_adapter_ingest_multi_parquet(tmp_path):
    adapter = KaggleMedicalTriageAdapter()

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Train shard with 1112 records
    train_records = [
        {
            "id": f"train_{i}",
            "symptom_description": f"Train complaint {i}",
            "symptoms": ["fever"],
            "urgency_level": "Urgent",
            "primary_specialty": "ED",
        }
        for i in range(1112)
    ]

    # Validation shard with 278 records
    val_records = [
        {
            "id": f"val_{i}",
            "symptom_description": f"Val complaint {i}",
            "symptoms": ["cough"],
            "urgency_level": "Routine",
            "primary_specialty": "CARDIO_PULM",
        }
        for i in range(278)
    ]

    pd.DataFrame(train_records).to_parquet(data_dir / "train-00000-of-00001.parquet")
    pd.DataFrame(val_records).to_parquet(data_dir / "validation-00000-of-00001.parquet")

    results = list(adapter.ingest(str(tmp_path)))
    assert len(results) >= 1

    df = pd.concat(results, ignore_index=True)
    assert len(df) == 1390
    assert len(df[df["raw_text"].str.contains("Train complaint")]) == 1112
    assert len(df[df["raw_text"].str.contains("Val complaint")]) == 278


def test_l3cube_code_mixed_adapter_metadata():
    adapter = L3CubeCodeMixedAdapter()
    assert adapter.dataset_source == "l3cube_code_mixed"
    assert adapter.version == "1.0"


def test_l3cube_code_mixed_adapter_ingest(tmp_path):
    adapter = L3CubeCodeMixedAdapter()

    base_dir = tmp_path / "code-mixed-nlp-main" / "L3Cube-HingLID"
    base_dir.mkdir(parents=True)

    with open(base_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("sanatan\tHI\n0809\tHI\ntiding\tEN\n\naap\tHI\nne\tHI\nkaha\tHI\n\n")

    results = list(adapter.ingest(str(tmp_path)))
    assert len(results) == 1
    res = results[0]
    assert len(res) == 2
    assert res.iloc[0]["raw_text"] == "sanatan 0809 tiding"
    assert res.iloc[1]["raw_text"] == "aap ne kaha"
    assert res.iloc[0]["language"] == "hi-en"
    assert res.iloc[0]["triage_level"] is None


def test_meddialog_en_adapter_metadata():
    adapter = MeddialogEnAdapter()
    assert adapter.dataset_source == "meddialog_en"
    assert adapter.version == "2.0"


def test_meddialog_en_adapter_ingest(tmp_path):
    adapter = MeddialogEnAdapter()

    with open(tmp_path / "dialog.jsonl", "w", encoding="utf-8") as f:
        f.write('{"utterances": ["Hello doctor", "Hi patient"]}\n')
        f.write('{"utterances": ["Chest pain"]}\n')
        f.write("\n")

    results = list(adapter.ingest(str(tmp_path)))
    assert len(results) == 1
    res = results[0]
    assert len(res) == 2
    assert res.iloc[0]["raw_text"] == "Hello doctor\nHi patient"
    assert res.iloc[1]["raw_text"] == "Chest pain"
    assert res.iloc[0]["language"] == "en"
    assert res.iloc[0]["triage_level"] is None


def test_meddialog_en_adapter_ingest_json_schema(tmp_path):
    """Test C: Verify the adapter ingests merged-MedDialog.json (instruction/input/output schema)."""
    import json as _json

    adapter = MeddialogEnAdapter()

    records = [
        {"instruction": "Patient has chest pain", "input": "", "output": "Possible cardiac issue"},
        {"instruction": "Headache for 3 days", "input": "No nausea", "output": "Monitor symptoms"},
        {"instruction": "Knee injury from fall", "input": "Swelling observed", "output": "X-ray recommended"},
    ]

    json_path = tmp_path / "merged-MedDialog.json"
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(records, f)

    results = list(adapter.ingest(str(tmp_path)))
    assert len(results) == 1
    res = results[0]
    assert len(res) == 3

    # Verify instruction/output concatenation (input="" is skipped)
    assert "Patient has chest pain" in res.iloc[0]["raw_text"]
    assert "Possible cardiac issue" in res.iloc[0]["raw_text"]

    # Record with non-empty input should include all three parts
    assert "No nausea" in res.iloc[1]["raw_text"]
    assert "Swelling observed" in res.iloc[2]["raw_text"]

    # All should be meddialog_en source
    assert all(res["dataset_source"] == "meddialog_en")


def test_meddialog_en_adapter_malformed_json_raises(tmp_path):
    """Test E: Verify the adapter raises RuntimeError for malformed/invalid JSON, not silent empty result."""
    adapter = MeddialogEnAdapter()

    # Write a file that looks like a JSON but is actually an LFS pointer
    lfs_content = "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1234\n"
    json_path = tmp_path / "broken.json"
    json_path.write_text(lfs_content, encoding="utf-8")

    with pytest.raises(RuntimeError, match="MedDialog JSON ingestion failed"):
        list(adapter.ingest(str(tmp_path)))


def test_meddialog_en_adapter_json_precedence_over_parquet(tmp_path):
    """Test F: Verify JSON path takes precedence over parquet when both exist."""
    import json as _json

    adapter = MeddialogEnAdapter()

    # Create a JSON file with 5 records
    json_records = [
        {"instruction": f"Symptom {i}", "input": "", "output": f"Diagnosis {i}"}
        for i in range(5)
    ]
    json_path = tmp_path / "merged-MedDialog.json"
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(json_records, f)

    # Create a parquet file with 3 different records
    import pandas as pd
    pq_data = pd.DataFrame({
        "description": ["desc1", "desc2", "desc3"],
        "utterances": [["utt1"], ["utt2"], ["utt3"]],
    })
    pq_dir = tmp_path / "data"
    pq_dir.mkdir()
    pq_data.to_parquet(pq_dir / "train.parquet", index=False)

    # Adapter should use JSON (5 records), NOT parquet (3 records)
    results = list(adapter.ingest(str(tmp_path)))
    total = sum(len(chunk) for chunk in results)
    assert total == 5, f"Expected 5 records from JSON, got {total} (parquet fallback was used)"
