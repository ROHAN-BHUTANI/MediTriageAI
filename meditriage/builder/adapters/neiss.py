from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .base import BaseAdapter


class NeissAdapter(BaseAdapter):
    """
    Adapter for the NEISS dataset.

    Mapping Strategy:
    - `Narrative_1` -> `raw_text`
    - Stream parquet using pyarrow.
    """

    @property
    def dataset_source(self) -> str:
        return "neiss"

    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        parquet_path = Path(raw_path) / "neiss_all.parquet"
        if not parquet_path.exists():
            return

        parquet_file = pq.ParquetFile(parquet_path)

        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            chunk_df = batch.to_pandas()

            # Vectorized operations
            # Filter valid text
            chunk_df["Narrative_1"] = chunk_df["Narrative_1"].astype(str).str.strip()
            valid_mask = (chunk_df["Narrative_1"] != "") & (
                chunk_df["Narrative_1"].str.lower() != "nan"
            )
            valid_df = chunk_df[valid_mask].copy()

            if len(valid_df) == 0:
                continue

            narrative_lower = valid_df["Narrative_1"].str.lower()

            narrative_lower = valid_df["Narrative_1"].str.lower()
            diag_code = pd.to_numeric(valid_df["Diagnosis"], errors="coerce") if "Diagnosis" in valid_df.columns else pd.Series(float("nan"), index=valid_df.index)
            body_code = pd.to_numeric(valid_df["Body_Part"], errors="coerce") if "Body_Part" in valid_df.columns else pd.Series(float("nan"), index=valid_df.index)

            # Deterministic Hierarchy:
            # 1. Diagnosis numeric code mapping
            # 55=Dislocation, 57=Fracture, 64=Strain/Sprain -> ORTHO
            # 52=Concussion, 61=Nerve Damage -> NEURO
            # 65=Anoxia, 67=Electric Shock, 68=Drowning -> CARDIO_PULM
            # 66=Poisoning/Ingestion -> GI
            # 54=Dental, 58/59=Laceration -> ENT_OPHTHALMO
            # 50=Amputation, 63=Puncture -> SURGERY
            department = pd.Series("GEN_MED", index=valid_df.index, dtype=object)

            ortho_diag = diag_code.isin([55, 57, 64])
            neuro_diag = diag_code.isin([52, 61])
            cardio_diag = diag_code.isin([65, 67, 68])
            gi_diag = diag_code.isin([66])
            ent_diag = diag_code.isin([54, 58, 59])
            surg_diag = diag_code.isin([50, 63])

            department.loc[ortho_diag] = "ORTHO"
            department.loc[neuro_diag] = "NEURO"
            department.loc[cardio_diag] = "CARDIO_PULM"
            department.loc[gi_diag] = "GI"
            department.loc[ent_diag] = "ENT_OPHTHALMO"
            department.loc[surg_diag] = "SURGERY"

            # 2. Body_Part numeric code mapping (for unassigned or general diagnoses)
            body_ent = body_code.isin([76, 77])  # Face, Eyeball
            body_neuro = body_code.isin([75])  # Head
            body_cardio = body_code.isin([31])  # Upper Trunk / Chest
            body_ortho = body_code.isin([30, 34, 35, 36, 37])  # Shoulder, Wrist, Knee, Lower Leg, Ankle
            body_uro = body_code.isin([33, 38])  # Lower Trunk, Pubic Region

            unmapped_mask = department == "GEN_MED"
            department.loc[unmapped_mask & body_ent] = "ENT_OPHTHALMO"
            department.loc[unmapped_mask & body_neuro] = "NEURO"
            department.loc[unmapped_mask & body_cardio] = "CARDIO_PULM"
            department.loc[unmapped_mask & body_ortho] = "ORTHO"
            department.loc[unmapped_mask & body_uro] = "RENAL_URO"

            # 3. Narrative text regex rules (override/refine unmapped or general cases)
            derm_mask = narrative_lower.str.contains("laceration|cut|burn|rash|skin|abrasion", regex=True)
            ortho_mask = narrative_lower.str.contains("fracture|sprain|strain|bone|joint|knee|shoulder|ankle|wrist|hip|dislocation", regex=True)
            neuro_mask = narrative_lower.str.contains("head injury|concussion|headache|dizziness|seizure|loss of consciousness", regex=True)
            cardio_mask = narrative_lower.str.contains("chest pain|shortness of breath|asthma|heart|lung|breathing", regex=True)
            eye_mask = narrative_lower.str.contains("eye|cornea|vision|ear|nose|throat|swallowed", regex=True)

            unmapped_mask = department == "GEN_MED"
            department.loc[unmapped_mask & ortho_mask] = "ORTHO"
            department.loc[unmapped_mask & neuro_mask] = "NEURO"
            department.loc[unmapped_mask & cardio_mask] = "CARDIO_PULM"
            department.loc[unmapped_mask & eye_mask] = "ENT_OPHTHALMO"
            department.loc[unmapped_mask & derm_mask] = "ENT_OPHTHALMO"

            # 4. Age < 18 -> PEDS priority override
            is_pediatric = pd.to_numeric(valid_df["Age"], errors="coerce") < 18
            department.loc[is_pediatric] = "PEDS"

            # Create standard dataframe
            out_df = pd.DataFrame(
                {
                    "dataset_source": self.dataset_source,
                    "raw_text": valid_df["Narrative_1"],
                    "department": department,
                    "triage_level": None,
                    "language": "en",
                }
            )

            yield out_df
