import time
from meditriage.builder.adapters.kaggle_medical_triage import KaggleMedicalTriageAdapter

def validate():
    adapter = KaggleMedicalTriageAdapter()
    total_records = 0
    chunks = 0
    
    print(f"Validating {adapter.dataset_source} against real dataset...", flush=True)
    start = time.time()
    
    for df in adapter.ingest("datasets/raw/kaggle_medical_triage", chunk_size=50000):
        chunks += 1
        total_records += len(df)
        
        assert "raw_text" in df.columns, "Missing raw_text"
        assert "triage_level" in df.columns, "Missing triage_level"
        assert "department" in df.columns, "Missing department"
        assert "language" in df.columns, "Missing language"
        assert (df["dataset_source"] == "kaggle_medical_triage").all(), "Wrong source"
        assert (df["language"] == "en").all(), "Language should be en"
        
        if chunks % 5 == 0:
            print(f"Processed {chunks} chunks, {total_records} records...", flush=True)
            
    end = time.time()
    print(f"Validation complete! Ingested {total_records} records across {chunks} chunks in {end - start:.2f}s.", flush=True)
    
if __name__ == "__main__":
    validate()
