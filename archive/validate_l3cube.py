import time
from meditriage.builder.adapters.l3cube_code_mixed import L3CubeCodeMixedAdapter

def validate():
    adapter = L3CubeCodeMixedAdapter()
    total_records = 0
    chunks = 0
    
    print(f"Validating {adapter.dataset_source} against real dataset...", flush=True)
    start = time.time()
    
    for df in adapter.ingest("datasets/raw/l3cube_code_mixed", chunk_size=50000):
        chunks += 1
        total_records += len(df)
        
        assert "raw_text" in df.columns, "Missing raw_text"
        assert "triage_level" in df.columns, "Missing triage_level"
        assert "department" in df.columns, "Missing department"
        assert "language" in df.columns, "Missing language"
        assert (df["dataset_source"] == "l3cube_code_mixed").all(), "Wrong source"
        assert (df["language"] == "hi-en").all(), "Language should be hi-en"
        
        if chunks % 5 == 0:
            print(f"Processed {chunks} chunks, {total_records} records...", flush=True)
            
    end = time.time()
    print(f"Validation complete! Ingested {total_records} records across {chunks} chunks in {end - start:.2f}s.", flush=True)
    
if __name__ == "__main__":
    validate()
