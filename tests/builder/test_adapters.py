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
