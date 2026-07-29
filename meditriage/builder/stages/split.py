import hashlib
import pandas as pd

def make_tracking_id(seed_id: str, variant_idx: int) -> str:
    return f"{seed_id}::{variant_idx}"

def _hash_to_float(s: str) -> float:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xffffffff

def assign_split(seed_id: str, splits: dict[str, float]) -> str:
    # splits = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    val = _hash_to_float(seed_id)
    if val < splits.get('train', 0.8):
        return 'train'
    elif val < splits.get('train', 0.8) + splits.get('val', 0.1):
        return 'val'
    else:
        return 'test'

def apply_split(df: pd.DataFrame, splits: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    if len(df) == 0:
        return df
    
    # Vectorized assignment
    df['split'] = df['seed_id'].apply(lambda s: assign_split(s, splits))
    return df
