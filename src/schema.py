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
        
    # Translate legacy 'department_code' or 'specialist_label' -> 'department'
    if "department" not in df.columns:
        if "department_code" in df.columns:
            df = df.rename(columns={"department_code": "department"})
        elif "specialist_label" in df.columns:
            df = df.rename(columns={"specialist_label": "department"})
        
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
        import warnings

        # Validate required columns
        required_cols = ['raw_text']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' is missing.")
        if "split" not in df.columns:
            df["split"] = "train"
                
        # Check for complete lack of supervision
        initial_len = len(df)
        
        # Text must not be null
        valid_text = df['raw_text'].notna()
        
        # We must have AT LEAST ONE valid label (department or triage)
        has_dept = df['department'].notna()
        valid_dept = df['department'].isin(SPECIALIST_CLASSES) & has_dept
        
        has_triage = df['triage_level'].notna()
        
        # Fix numeric triage mapping
        if not df.empty and has_triage.any():
            first_valid = df.loc[has_triage, "triage_level"].iloc[0]
            if df["triage_level"].dtype in ['float64', 'int64', 'float32', 'int32'] or str(first_valid).replace('.', '').isdigit():
                # Only apply to valid triage rows to avoid NaN issues
                df["triage_level"] = df["triage_level"].astype(str)
                has_triage = df["triage_level"].notna() & (df["triage_level"] != "nan")
                df.loc[has_triage, "triage_level"] = "S" + pd.to_numeric(df.loc[has_triage, "triage_level"], errors='coerce').fillna(0).astype(int).astype(str)

        valid_triage = df['triage_level'].isin(SEVERITY_LABELS) & has_triage
        
        valid_label_mask = valid_dept | valid_triage
        
        df = df[valid_text & valid_label_mask].copy()
        
        dropped_count = initial_len - len(df)
        if dropped_count > 0:
            warnings.warn(f"Schema validation dropped {dropped_count} rows lacking raw_text or missing both specialist and severity supervision.")
            
        # Standardize columns
        # Non-valid labels should be converted to None (or pad) for masked loss later
        df.loc[~valid_dept, 'department'] = None
        df.loc[~valid_triage, 'triage_level'] = None
        
    except ImportError:
        pass
        
    return df
