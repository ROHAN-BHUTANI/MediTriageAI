import time
import pandas as pd
from pathlib import Path
from importlib import import_module
import inspect

from .config import Config
from .stages.normalize import apply_normalization
from .stages.deduplicate import apply_deduplication
from .stages.augment import apply_augmentation
from .stages.split import apply_split
from .stages.validate import validate_dataframe
from .utils.reporting import write_duplicate_report, write_statistics, write_manifest
from .adapters.base import BaseAdapter

class Builder:
    def __init__(self, config: Config, base_dir: Path):
        self.config = config
        self.base_dir = base_dir
        self.raw_dir = base_dir / "datasets" / "raw"
        self.out_dir = base_dir / "meditriage" / "data"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "processed").mkdir(parents=True, exist_ok=True)
        
        # Dynamically load adapters
        self.registry = {}
        try:
            adapters_module = import_module("meditriage.builder.adapters")
            for name, obj in inspect.getmembers(adapters_module):
                if inspect.isclass(obj) and issubclass(obj, BaseAdapter) and obj is not BaseAdapter:
                    try:
                        inst = obj()
                        self.registry[inst.dataset_source] = obj
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: Failed to load adapters dynamically: {e}")

    def build(self, force: bool = False) -> None:
        if (self.out_dir / "processed" / "dataset.csv").exists() and not force:
            raise FileExistsError("Dataset exists. Use --force to overwrite.")
            
        start_time = time.time()
        print("Starting dataset build...")
        
        dfs = []
        adapters = {}
        for source in self.config.active_datasets:
            if source in self.registry:
                adapter = self.registry[source]()
                adapters[source] = adapter
                print(f"Ingesting {source}...")
                df = adapter.ingest(str(self.raw_dir / source))
                if not df.empty:
                    dfs.append(df)
            else:
                print(f"Warning: Adapter for {source} not found.")
                
        if not dfs:
            print("No data ingested.")
            return
            
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Ingested {len(combined_df)} raw records.")
        
        print("Applying normalization...")
        combined_df = apply_normalization(combined_df)
        
        print("Applying deduplication...")
        combined_df, dropped = apply_deduplication(combined_df, self.config.deduplication)
        write_duplicate_report(str(self.out_dir / "processed" / "duplicate_report.txt"), dropped)
        print(f"Dropped {len(dropped)} duplicates.")
        
        print("Applying augmentation...")
        combined_df = apply_augmentation(combined_df, self.config.augmentation)
        print(f"Total rows after augmentation: {len(combined_df)}")
        
        print("Applying splits...")
        combined_df = apply_split(combined_df, self.config.splits)
        
        print("Validating schema and invariants...")
        validate_dataframe(combined_df, require_split=True)
        
        print("Writing artifacts...")
        out_csv = self.out_dir / "processed" / "dataset.csv"
        combined_df.to_csv(out_csv, index=False)
        
        write_statistics(combined_df, str(self.out_dir / "processed" / "dataset_statistics.json"))
        write_manifest(self.config, combined_df, adapters, start_time, self.out_dir)
        
        print("Build complete!")
