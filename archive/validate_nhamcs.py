import time
from meditriage.builder.adapters.nhamcs_ed import NhamcsEdAdapter

def validate():
    adapter = NhamcsEdAdapter()
    total_records = 0
    chunks = 0
    
    print(f"Validating {adapter.dataset_source} against real dataset...", flush=True)
    start = time.time()
    
    for df in adapter.ingest("datasets/raw/nhamcs_ed", chunk_size=50000):
        chunks += 1
        total_records += len(df)
        
        # Validation checks
        assert "raw_text" in df.columns, "Missing raw_text"
        assert "triage_level" in df.columns, "Missing triage_level"
        assert "department" in df.columns, "Missing department"
        assert "language" in df.columns, "Missing language"
        assert (df["dataset_source"] == "nhamcs_ed").all(), "Wrong source"
        assert (df["language"] == "en").all(), "Language should be en"
        
        if chunks % 5 == 0:
            print(f"Processed {chunks} chunks, {total_records} records...", flush=True)
            
    end = time.time()
    print(f"Validation complete! Ingested {total_records} records across {chunks} chunks in {end - start:.2f}s.", flush=True)
    
if __name__ == "__main__":
    validate()
