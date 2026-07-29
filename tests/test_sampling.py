import pytest
import pandas as pd
from src.sampling import create_stratified_subset

def test_stratified_sampling():
    import random
    from src.dataset import SPECIALIST_CLASSES, SEVERITY_LABELS
    
    # Generate synthetic dataframe with 13 classes
    data = []
    for cls in SPECIALIST_CLASSES:
        for sev in SEVERITY_LABELS:
            # Over-represent the first class to test ceiling limit logic
            num_samples = 500 if cls == SPECIALIST_CLASSES[0] else 50
            for _ in range(num_samples):
                data.append({'raw_text': 'test data', 'department': cls, 'triage_level': sev})
    
    df = pd.DataFrame(data)
    
    from src.schema import validate_and_translate_schema
    df = validate_and_translate_schema(df)
    
    # 1. Stratifies by department
    subset = create_stratified_subset(df, 1000, label_col='department', secondary_col='triage_level')
    
    # Check length
    assert len(subset) == 1000
    
    counts = subset['department'].value_counts()
    
    # (a) all 13 department classes appear in any subset of size >= 1000
    assert len(counts) == 13, f"Not all classes appeared! Found {len(counts)}"
    
    # (b) no single class exceeds reasonable ceiling unless it's genuinely that large
    full_counts = df['department'].value_counts(normalize=True)
    subset_counts = counts / len(subset)
    
    for cls, prop in subset_counts.items():
        if prop > 0.40:
            assert full_counts[cls] > 0.40, f"Class {cls} exceeds 40% in subset but not in full dataset."

def test_sampling_small():
    df = pd.DataFrame({'department': ['A', 'A', 'B', 'B', 'B', 'C'], 'triage_level': [1,2,1,2,1,2]})
    subset = create_stratified_subset(df, 3, label_col='department')
    assert len(subset) == 3
