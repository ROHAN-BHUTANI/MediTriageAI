import pandas as pd

from ..schema import validate_schema


def validate_dataframe(df: pd.DataFrame, require_split: bool = False) -> None:
    validate_schema(df, require_split=require_split)

    # Check leakage
    if require_split:
        leakage_check = df.groupby("seed_id")["split"].nunique()
        leaked = leakage_check[leakage_check > 1]
        if not leaked.empty:
            raise ValueError(
                f"LEAKAGE DETECTED: seed_ids {leaked.index.tolist()} span multiple splits!"
            )
