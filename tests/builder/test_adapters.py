import pytest
import pandas as pd
import tempfile
from pathlib import Path
import os
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
        df = pd.DataFrame({
            "Unnamed: 0": [0, 1, 2],
            "description": ["desc0", "desc1", "desc2"],
            "medical_specialty": [" Allergy / Immunology", " Bariatrics", " Cardiovascular / Pulmonary"],
            "sample_name": ["sample0", "sample1", "sample2"],
            "transcription": ["text0", "nan", "text2"], # row 1 missing transcription, should fallback to desc
            "keywords": ["kw0", "kw1", "kw2"]
        })
        df.set_index("Unnamed: 0", inplace=True)
        df.to_csv(csv_path)
        
        # Test ingestion with chunksize 2
        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))
        
        assert len(chunks) == 2 # 3 rows total -> chunks of 2 and 1
        
        first_chunk = chunks[0]
        assert len(first_chunk) == 2
        assert first_chunk.iloc[0]["raw_text"] == "text0"
        assert first_chunk.iloc[0]["raw_medical_specialty"] == "Allergy / Immunology"
        assert first_chunk.iloc[0]["dataset_source"] == "mtsamples"
        
        # Fallback transcription test
        assert first_chunk.iloc[1]["raw_text"] == "desc1"
        assert first_chunk.iloc[1]["raw_medical_specialty"] == "Bariatrics"
        
        second_chunk = chunks[1]
        assert len(second_chunk) == 1
        assert second_chunk.iloc[0]["raw_text"] == "text2"
        assert second_chunk.iloc[0]["raw_medical_specialty"] == "Cardiovascular / Pulmonary"

def test_mtsamples_adapter_ingest_empty_skip():
    adapter = MTSamplesAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir)
        csv_path = raw_path / "mtsamples (1).csv"
        
        df = pd.DataFrame({
            "Unnamed: 0": [0],
            "description": ["nan"],
            "medical_specialty": ["nan"],
            "transcription": ["nan"],
        })
        df.set_index("Unnamed: 0", inplace=True)
        df.to_csv(csv_path)
        
        chunks = list(adapter.ingest(str(raw_path)))
        assert len(chunks) == 0 # The only row is empty and should be skipped

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
        
        df = pd.DataFrame({
            "patient": ["text0", "nan", "", "text3"]
        })
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
        
        df = pd.DataFrame({
            "question": ["text0", "nan", "", "text3"]
        })
        df.to_json(jsonl_path, orient="records", lines=True)
        
        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))
        
        assert len(chunks) == 2
        
        first_chunk = chunks[0]
        assert len(first_chunk) == 1
        assert first_chunk.iloc[0]["raw_text"] == "text0"
        
        second_chunk = chunks[1]
        assert len(second_chunk) == 1
        assert second_chunk.iloc[0]["raw_text"] == "text3"

from meditriage.builder.adapters.medical_meadow_medqa import MedicalMeadowMedqaAdapter
import json

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
            {"input": "text3", "instruction": "inst3"}
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

from meditriage.builder.adapters.symptom2disease import Symptom2DiseaseAdapter

def test_symptom2disease_adapter_metadata():
    adapter = Symptom2DiseaseAdapter()
    assert adapter.dataset_source == "symptom2disease"
    assert adapter.version == "1.0.0"

def test_symptom2disease_adapter_ingest():
    adapter = Symptom2DiseaseAdapter()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir)
        csv_path = raw_path / "Symptom2Disease.csv"
        
        df = pd.DataFrame({
            "text": ["text0", "nan", "", "text3"],
            "label": ["Psoriasis", "disease", "", "disease2"]
        })
        df.to_csv(csv_path, index=False)
        
        chunks = list(adapter.ingest(str(raw_path), chunk_size=2))
        
        assert len(chunks) == 2
        
        first_chunk = chunks[0]
        assert len(first_chunk) == 1
        assert first_chunk.iloc[0]["raw_text"] == "text0"
        assert first_chunk.iloc[0]["raw_medical_specialty"] == "Psoriasis"
        
        second_chunk = chunks[1]
        assert len(second_chunk) == 1
        assert second_chunk.iloc[0]["raw_text"] == "text3"
        assert second_chunk.iloc[0]["raw_medical_specialty"] == "disease2"
