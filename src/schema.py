"""Centralized schema translation and validation layer for datasets."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "raw_text",
    "department",
    "triage_level",
}

def validate_and_translate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates that the dataframe meets the required production schema.
    Provides backwards compatibility for legacy schemas by translating 
    column names where appropriate.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        pd.DataFrame: A new dataframe matching the canonical schema.
        
    Raises:
        ValueError: If required columns are missing and cannot be translated.
    """
    # Create a copy to avoid mutating the original dataframe
    df = df.copy()
    
    # Translate legacy 'text' -> 'raw_text'
    if "text" in df.columns and "raw_text" not in df.columns:
        df = df.rename(columns={"text": "raw_text"})
        
    # Translate legacy 'department_code' -> 'department'
    if "department_code" in df.columns and "department" not in df.columns:
        df = df.rename(columns={"department_code": "department"})
        
    # Translate legacy 'severity_heuristic' or 'severity_label' -> 'triage_level'
    if "triage_level" not in df.columns:
        if "severity_heuristic" in df.columns:
            df = df.rename(columns={"severity_heuristic": "triage_level"})
        elif "severity_label" in df.columns:
            df = df.rename(columns={"severity_label": "triage_level"})
            
    # Fail fast if canonical schema is not satisfied
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required canonical columns: {sorted(missing_cols)}. "
            f"Found columns: {df.columns.tolist()}"
        )
        
    return df
