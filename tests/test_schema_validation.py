import pytest
import pandas as pd
from src.schema import validate_and_translate_schema
from src.dataset import load_split_rows
from unittest.mock import patch, MagicMock

def test_stratified_sampling_missing_values(tmp_path):
    """
    Regression test to ensure stratified sampling never receives null labels
    and schema validation drops invalid rows gracefully.
    """
    # Create dataset with nulls in required columns
    df = pd.DataFrame({
        "raw_text": ["text1", "text2", None, "text4", "text5"],
        "department": ["DEP1", "DEP2", "DEP1", None, "DEP2"],
        "triage_level": ["S1", "S2", "S3", "S1", None],
        "split": ["train", "train", "train", "train", "train"]
    })
    
    # Save to a temporary parquet file
    dataset_path = tmp_path / "test_dataset.parquet"
    df.to_parquet(dataset_path)
    
    # Call load_split_rows, which should process schema validation
    # Note: We need to patch SPECIALIST_CLASSES and SEVERITY_LABELS to match our dummy data
    # since load_split_rows does strict mapping
    with patch("src.dataset.SPECIALIST_CLASSES", ["DEP1", "DEP2", "DEP3"]), \
         patch("src.dataset.SEVERITY_LABELS", ["S1", "S2", "S3"]):
        
        with pytest.warns(UserWarning, match="Schema validation dropped 3 rows"):
            rows = load_split_rows(dataset_path, split="train", max_rows=100)
            
    # Out of 5 rows, 3 have nulls. We should only get 2 back.
    assert len(rows) == 2
    assert rows[0]["text"] == "text1"
    assert rows[1]["text"] == "text2"
