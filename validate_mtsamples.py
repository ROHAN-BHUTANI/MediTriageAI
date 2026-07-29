from meditriage.builder.adapters.mtsamples import MTSamplesAdapter

def validate():
    adapter = MTSamplesAdapter()
    total_records = 0
    chunks = 0
    
    print(f"Validating {adapter.dataset_source} against real dataset...")
    
    for df in adapter.ingest("datasets/raw/mtsamples", chunk_size=1000):
        total_records += len(df)
        chunks += 1
        
        # Invariants check
        assert not df["raw_text"].isnull().any(), "Found null raw_text"
        assert not (df["raw_text"] == "").any(), "Found empty raw_text"
        assert (df["dataset_source"] == "mtsamples").all(), "Wrong source"
        
    print(f"Validation complete! Ingested {total_records} records across {chunks} chunks.")
    
if __name__ == "__main__":
    validate()
