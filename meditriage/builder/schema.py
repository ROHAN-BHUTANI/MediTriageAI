import pandas as pd

REQUIRED_COLUMNS = {
    "tracking_id": str,
    "seed_id": str,
    "dataset_source": str,
    "raw_text": str,
    "raw_medical_specialty": str, 
    "raw_severity": str, 
    "language": str,
    "text": str,
    "department": str,
    "routing_confidence": str,
    "triage_level": str,
    "severity_label_source": str,
    "is_perturbed": bool,
    "variant_index": int,
    "split": str
}

def validate_schema(df: pd.DataFrame, require_split: bool = False) -> None:
    missing = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed: Missing columns {missing}")
        
    strict_non_null = [
        "tracking_id", "seed_id", "dataset_source", 
        "raw_text", "language", "text", 
        "routing_confidence", "severity_label_source",
        "is_perturbed", "variant_index"
    ]
    if require_split:
        strict_non_null.append("split")
        
    for col in strict_non_null:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            raise ValueError(f"Schema validation failed: Column '{col}' contains {null_count} null values.")
            
    valid_confidences = {"high", "low"}
    invalid_conf = set(df["routing_confidence"].dropna().unique()) - valid_confidences
    if invalid_conf:
        raise ValueError(f"Schema validation failed: Invalid routing_confidence values {invalid_conf}")
        
    valid_severities = {"S1", "S2", "S3", "S4", "S5", "UNKNOWN"}
    invalid_sev = set(df["triage_level"].dropna().unique()) - valid_severities
    if invalid_sev:
        raise ValueError(f"Schema validation failed: Invalid triage_level values {invalid_sev}")
        
    if require_split:
        valid_splits = {"train", "val", "test"}
        invalid_splits = set(df["split"].dropna().unique()) - valid_splits
        if invalid_splits:
            raise ValueError(f"Schema validation failed: Invalid split values {invalid_splits}")
