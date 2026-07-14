import pytest
import pandas as pd
from src.sampling import create_stratified_subset

def test_stratified_sampling():
    df = pd.read_csv('meditriage/data/processed/dataset.csv')
    df = df.dropna(subset=['text'])
    
    # 1. Stratifies by department_code
    sev_col = 'severity_heuristic' if 'severity_heuristic' in df.columns else 'severity_label'
    subset = create_stratified_subset(df, 1000, label_col='department_code', secondary_col=sev_col)
    
    # Check length
    assert len(subset) == 1000
    
    counts = subset['department_code'].value_counts()
    
    # (a) all 13 department_code classes appear in any subset of size >= 1000
    assert len(counts) == 13, f"Not all classes appeared! Found {len(counts)}"
    
    # (b) no single class exceeds reasonable ceiling unless it's genuinely that large
    full_counts = df['department_code'].value_counts(normalize=True)
    subset_counts = counts / len(subset)
    
    for cls, prop in subset_counts.items():
        if prop > 0.40:
            assert full_counts[cls] > 0.40, f"Class {cls} exceeds 40% in subset but not in full dataset."

def test_sampling_small():
    df = pd.DataFrame({'department_code': ['A', 'A', 'B', 'B', 'B', 'C'], 'severity': [1,2,1,2,1,2]})
    subset = create_stratified_subset(df, 3, label_col='department_code')
    assert len(subset) == 3
