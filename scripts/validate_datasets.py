import sys
import os
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset_registry import get_enabled_adapters

def main():
    print("Starting Dataset Adapters Validation...")
    adapters = get_enabled_adapters()
    if not adapters:
        print("No adapters enabled in config.")
        sys.exit(0)
        
    all_passed = True
    
    for adapter in adapters:
        name = adapter.__class__.__name__
        print(f"\n--- Validating {name} ---")
        
        if not adapter.data_path.exists():
            print(f"[{name}] WARNING: Path {adapter.data_path} does not exist. Skipping.")
            continue
            
        record_count = 0
        null_issues = 0
        schema_issues = 0
        splits_seen = set()
        
        try:
            for record in adapter.iter_records():
                record_count += 1
                
                # Check schema / fields
                if not record.complaint_text:
                    null_issues += 1
                if not record.source_dataset:
                    schema_issues += 1
                if not record.split:
                    schema_issues += 1
                
                splits_seen.add(record.split)
                
                # Lightweight check: max 1000 items per adapter just to ensure it parses correctly
                if record_count >= 1000:
                    break
                    
            print(f"[{name}] OK - Parsed {record_count} records (capped at 1000 for validation).")
            print(f"[{name}] Splits found: {splits_seen}")
            if null_issues > 0 or schema_issues > 0:
                print(f"[{name}] FAILED - Null text: {null_issues}, Schema Issues: {schema_issues}")
                all_passed = False
                
        except Exception as e:
            print(f"[{name}] FAILED to parse records. Error: {e}")
            all_passed = False

    if all_passed:
        print("\nAll datasets passed lightweight validation!")
    else:
        print("\nValidation failed for one or more datasets.")
        sys.exit(1)

if __name__ == "__main__":
    main()
