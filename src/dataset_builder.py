from typing import List, Dict, Any, Iterator
from pathlib import Path
import pandas as pd
from src.dataset_registry import get_enabled_adapters, load_config
from src.dataset_adapters import UnifiedPatientRecord
from src.model import SEVERITY_LABELS, SPECIALIST_CLASSES

def build_unified_records(split: str) -> Iterator[Dict[str, Any]]:
    """
    Yields in-memory dictionary records from all enabled dataset adapters.
    Records are converted to the format expected by the model.
    """
    adapters = get_enabled_adapters()
    for adapter in adapters:
        # Check if the raw directory exists; if not, we can't yield records
        if not adapter.data_path.exists():
            continue
            
        for record in adapter.iter_records():
            if record.split == split:
                try:
                    spec_id = SPECIALIST_CLASSES.index(record.specialist_label) if record.specialist_label in SPECIALIST_CLASSES else SPECIALIST_CLASSES.index("GEN_MED")
                except ValueError:
                    spec_id = SPECIALIST_CLASSES.index("GEN_MED")
                    
                try:
                    sev_id = SEVERITY_LABELS.index(record.severity_label) if record.severity_label in SEVERITY_LABELS else SEVERITY_LABELS.index(2) # Default moderate
                except ValueError:
                    sev_id = SEVERITY_LABELS.index(2)
                    
                yield {
                    "text": record.complaint_text,
                    "label_specialist_id": spec_id,
                    "label_severity_id": sev_id,
                    "source_dataset": record.source_dataset,
                    "language": record.language,
                }

def optionally_cache_unified_dataset(split: str, records: List[Dict[str, Any]]):
    """Caches the built dataset to CSV if configured to do so."""
    config = load_config()
    if config.get("cache_unified_dataset", False):
        cache_dir = Path("data/processed/unified")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"dataset_unified_{split}.csv"
        df = pd.DataFrame(records)
        df.to_csv(cache_path, index=False)
