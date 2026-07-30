import pandas as pd

from ..utils.hash import hash_to_int


def perturb_text(text: str, seed: int, rate: float) -> str:
    # Minimal perturbation for test purposes
    if rate > 0 and len(text) > 0 and seed % 2 == 0:
        return text.replace("a", "aa")
    return text


def apply_augmentation(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if len(df) == 0:
        return df

    hinglish_config = config.get("hinglish", {})
    enabled_for = set(hinglish_config.get("enabled_for", []))
    variants = hinglish_config.get("variants_per_seed", 0)
    rate = hinglish_config.get("substitution_rate", 0.5)

    if not enabled_for or variants == 0:
        return df

    new_rows = []
    for _, row in df.iterrows():
        new_rows.append(row.to_dict())  # keep original

        if row["dataset_source"] in enabled_for:
            for v in range(1, variants + 1):
                new_row = row.to_dict()
                new_row["variant_index"] = v
                new_row["is_perturbed"] = True
                new_row["language"] = "hinglish"
                new_row["tracking_id"] = f"{row['seed_id']}::{v}"

                # Perturb
                seed = hash_to_int(new_row["tracking_id"])
                new_row["text"] = perturb_text(row["text"], seed, rate)

                new_rows.append(new_row)

    return pd.DataFrame(new_rows)
