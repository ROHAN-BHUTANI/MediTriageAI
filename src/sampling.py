import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

def create_stratified_subset(df: pd.DataFrame, n: int, label_col: str, seed: int = 42, secondary_col: str = None, min_guarantee: int = 5) -> pd.DataFrame:
    """
    Creates a stratified subset of a dataframe with a minimum representation guarantee.
    """
    if n >= len(df):
        return df.copy()
        
    try:
        subset, _ = train_test_split(df, train_size=n, stratify=df[label_col], random_state=seed)
    except ValueError:
        warnings.warn("Classes too small to strictly stratify, falling back to random sample.")
        subset = df.sample(n=n, random_state=seed)

    # Minimum guarantee injection logic
    # First, track what we need
    guaranteed_indices = set()
    
    # Check primary column
    for val in df[label_col].unique():
        current_count = (subset[label_col] == val).sum()
        if current_count < min_guarantee:
            shortfall = min_guarantee - current_count
            pool = df[(df[label_col] == val) & (~df.index.isin(subset.index))]
            sampled = pool.sample(n=min(shortfall, len(pool)), random_state=seed)
            guaranteed_indices.update(sampled.index.tolist())
            
    # Check secondary column
    if secondary_col and secondary_col in df.columns:
        for val in df[secondary_col].unique():
            current_count = (subset[secondary_col] == val).sum()
            # Note: We must also count the ones we just added in guaranteed_indices
            added_count = (df.loc[list(guaranteed_indices), secondary_col] == val).sum() if guaranteed_indices else 0
            if current_count + added_count < min_guarantee:
                shortfall = min_guarantee - (current_count + added_count)
                pool = df[(df[secondary_col] == val) & (~df.index.isin(subset.index)) & (~df.index.isin(guaranteed_indices))]
                sampled = pool.sample(n=min(shortfall, len(pool)), random_state=seed)
                guaranteed_indices.update(sampled.index.tolist())
                
    if guaranteed_indices:
        # We need to swap out some existing majority-class rows to make room
        # Identify rows we can drop (don't drop minority classes we barely have enough of)
        num_to_drop = len(guaranteed_indices)
        
        # Calculate safety margins so we don't accidentally drop something below the guarantee
        subset_to_keep = []
        subset_to_drop = []
        
        label_counts = subset[label_col].value_counts().to_dict()
        if secondary_col:
            sec_counts = subset[secondary_col].value_counts().to_dict()
            
        for idx, row in subset.iterrows():
            l_val = row[label_col]
            s_val = row[secondary_col] if secondary_col else None
            
            can_drop = True
            if label_counts[l_val] <= min_guarantee:
                can_drop = False
            if secondary_col and sec_counts[s_val] <= min_guarantee:
                can_drop = False
                
            if can_drop:
                subset_to_drop.append(idx)
                # update virtual counts
                label_counts[l_val] -= 1
                if secondary_col:
                    sec_counts[s_val] -= 1
            else:
                subset_to_keep.append(idx)
                
        # If we can't find enough safe rows to drop, just drop randomly from what we marked droppable, 
        # or if still not enough, drop purely randomly (rare edge case)
        if len(subset_to_drop) >= num_to_drop:
            # We randomly select which of the droppable rows to actually drop
            import random
            random.seed(seed)
            actually_dropped = set(random.sample(subset_to_drop, num_to_drop))
            final_subset_indices = list(set(subset.index) - actually_dropped) + list(guaranteed_indices)
        else:
            # Fallback: just drop the first `num_to_drop` rows from the largest classes
            final_subset_indices = list(subset.index)[num_to_drop:] + list(guaranteed_indices)
            
        subset = df.loc[final_subset_indices]
        
    return subset
