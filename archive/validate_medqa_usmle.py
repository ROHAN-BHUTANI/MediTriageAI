from meditriage.builder.adapters.medqa_usmle import MedqaUsmleAdapter
import time

def validate():
    adapter = MedqaUsmleAdapter()
    total_records = 0
    chunks = 0
    
    print(f"Validating {adapter.dataset_source} against real dataset...")
    start = time.time()
    
    for df in adapter.ingest("datasets/raw/medqa_usmle", chunk_size=1000):
        total_records += len(df)
        chunks += 1
        
        # Invariants check
        assert not df["raw_text"].isnull().any(), "Found null raw_text"
        assert not (df["raw_text"] == "").any(), "Found empty raw_text"
        assert (df["dataset_source"] == "medqa_usmle").all(), "Wrong source"
        
        if chunks % 5 == 0:
            print(f"Processed {chunks} chunks, {total_records} records...")
            
    end = time.time()
    print(f"Validation complete! Ingested {total_records} records across {chunks} chunks in {end - start:.2f}s.")
    
if __name__ == "__main__":
    validate()
