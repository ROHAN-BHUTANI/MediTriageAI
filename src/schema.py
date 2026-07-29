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
        
    # Audit and drop nulls
    initial_rows = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    
    # Map legacy class names to canonical classes
    if "department" in df.columns:
        df["department"] = df["department"].replace({"Emergency": "ED"})
        
    # Drop rows that do not match known classes
    try:
        from src.dataset import SPECIALIST_CLASSES, SEVERITY_LABELS
        df = df[df["department"].isin(SPECIALIST_CLASSES)]
        if not df.empty:
            # If triage level is numeric or string representation of float (1.0, 2.0), map to S1, S2
            if df["triage_level"].dtype in ['float64', 'int64', 'float32', 'int32'] or str(df["triage_level"].iloc[0]).replace('.', '').isdigit():
                df["triage_level"] = "S" + df["triage_level"].astype(float).astype(int).astype(str)
            
            # Filter remaining
            df = df[df["triage_level"].isin(SEVERITY_LABELS)]
    except ImportError:
        pass
        
    dropped_rows = initial_rows - len(df)
    
    if dropped_rows > 0:
        import warnings
        warnings.warn(
            f"Schema validation dropped {dropped_rows:,} rows containing null values "
            f"or unrecognized labels in required columns {sorted(list(REQUIRED_COLUMNS))}."
        )
        
    return df
