import pandas as pd


def apply_deduplication(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list]:
    if len(df) == 0:
        return df, []

    priority_order = config.get("priority_order", [])

    # Map dataset_source to priority score (lower is better)
    priority_map = {source: i for i, source in enumerate(priority_order)}

    # Assign a priority score (default to high number if not in priority order)
    df["_priority"] = df["dataset_source"].map(lambda x: priority_map.get(x, 999))

    # Sort by text and priority
    df = df.sort_values(["text", "_priority"])

    # Find duplicates
    duplicates_mask = df.duplicated(subset=["text"], keep="first")
    duplicates = df[duplicates_mask]
    dropped_seeds = duplicates["seed_id"].tolist()

    # Keep first
    df = df[~duplicates_mask].copy()
    df = df.drop(columns=["_priority"])

    return df, dropped_seeds
