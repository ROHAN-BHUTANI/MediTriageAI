import time
import pandas as pd
from pathlib import Path

from .config import Config
from .adapters.mtsamples import MTSamplesAdapter
from .adapters.pmc_patients import PMCPatientsAdapter
from .stages.normalize import apply_normalization
from .stages.deduplicate import apply_deduplication
from .stages.augment import apply_augmentation
from .stages.split import apply_split
from .stages.validate import validate_dataframe
from .utils.reporting import write_duplicate_report, write_statistics, write_manifest

ADAPTER_REGISTRY = {
    "mtsamples": MTSamplesAdapter,
    "pmc_patients": PMCPatientsAdapter
}

class Builder:
    def __init__(self, config: Config, base_dir: Path):
        self.config = config
        self.base_dir = base_dir
        self.raw_dir = base_dir / "datasets" / "raw"
        self.out_dir = base_dir / "meditriage" / "data"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "processed").mkdir(parents=True, exist_ok=True)
        
    def build(self, force: bool = False) -> None:
        if (self.out_dir / "processed" / "dataset.csv").exists() and not force:
            raise FileExistsError("Dataset exists. Use --force to overwrite.")
            
        start_time = time.time()
        print("Starting dataset build...")
        
        # 1. Ingest
        dfs = []
        adapters = {}
        for source in self.config.active_datasets:
            if source in ADAPTER_REGISTRY:
                adapter = ADAPTER_REGISTRY[source]()
                adapters[source] = adapter
                print(f"Ingesting {source}...")
                df = adapter.ingest(str(self.raw_dir / source))
                dfs.append(df)
            else:
                print(f"Warning: Adapter for {source} not found.")
                
        if not dfs:
            print("No data ingested.")
            return
            
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Ingested {len(combined_df)} raw records.")
        
        # 2. Normalize
        print("Applying normalization...")
        combined_df = apply_normalization(combined_df)
        
        # 3. Deduplicate
        print("Applying deduplication...")
        combined_df, dropped = apply_deduplication(combined_df, self.config.deduplication)
        write_duplicate_report(str(self.out_dir / "processed" / "duplicate_report.txt"), dropped)
        print(f"Dropped {len(dropped)} duplicates.")
        
        # 4. Augment
        print("Applying augmentation...")
        combined_df = apply_augmentation(combined_df, self.config.augmentation)
        print(f"Total rows after augmentation: {len(combined_df)}")
        
        # 5. Split
        print("Applying splits...")
        combined_df = apply_split(combined_df, self.config.splits)
        
        # 6. Validate
        print("Validating schema and invariants...")
        validate_dataframe(combined_df, require_split=True)
        
        # 7. Write out
        print("Writing artifacts...")
        out_csv = self.out_dir / "processed" / "dataset.csv"
        combined_df.to_csv(out_csv, index=False)
        
        write_statistics(combined_df, str(self.out_dir / "processed" / "dataset_statistics.json"))
        write_manifest(self.config, combined_df, adapters, start_time, self.out_dir)
        
        print("Build complete!")
