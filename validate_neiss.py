from meditriage.builder.adapters.neiss import NeissAdapter
import time

def validate():
    adapter = NeissAdapter()
    total_records = 0
    chunks = 0
    
    print(f"Validating {adapter.dataset_source} against real dataset...", flush=True)
    start = time.time()
    
    for df in adapter.ingest("datasets/raw/neiss", chunk_size=100000):
        total_records += len(df)
        chunks += 1
        
        # Invariants check
        assert not df["raw_text"].isnull().any(), "Found null raw_text"
        assert not (df["raw_text"] == "").any(), "Found empty raw_text"
        assert (df["dataset_source"] == "neiss").all(), "Wrong source"
        
        if chunks % 5 == 0:
            print(f"Processed {chunks} chunks, {total_records} records...", flush=True)
            
    end = time.time()
    print(f"Validation complete! Ingested {total_records} records across {chunks} chunks in {end - start:.2f}s.", flush=True)
    
if __name__ == "__main__":
    validate()
