from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .base import BaseAdapter


class ChatDoctorHealthcareMagicAdapter(BaseAdapter):
    """
    Adapter for the ChatDoctor HealthcareMagic dataset.

    Mapping Strategy:
    - `input` -> `raw_text`
    - Stream parquet using pyarrow.
    """

    @property
    def dataset_source(self) -> str:
        return "chatdoctor_healthcaremagic"

    @property
    def version(self) -> str:
        return "1.0.0"

    def ingest(self, raw_path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        data_dir = Path(raw_path) / "data"
        if not data_dir.exists():
            return

        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            return

        for pfile in parquet_files:
            parquet_file = pq.ParquetFile(pfile)

            # Use iter_batches with batch_size=chunk_size
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas()

                # Vectorized operations
                chunk_df["input"] = chunk_df["input"].astype(str).str.strip()
                valid_mask = (chunk_df["input"] != "") & (
                    chunk_df["input"].str.lower() != "nan"
                )
                valid_df = chunk_df[valid_mask]

                if len(valid_df) == 0:
                    continue

                # Deterministic specialty routing (prefer doctor output over patient query)
                p_text = valid_df["input"].astype(str).str.lower()
                d_text = valid_df.get("output", pd.Series("", index=valid_df.index)).fillna("").astype(str).str.lower()
                comb_text = d_text + " " + p_text

                department = pd.Series("GEN_MED", index=valid_df.index, dtype=object)

                peds_m = d_text.str.contains("pediatric|pediatrician|baby|infant|toddler|child", regex=True)
                obgyn_m = comb_text.str.contains("gynecolog|obstetric|pregnancy|pregnant|period|menstrual|vagina|ovary|uterus|pap smear", regex=True)
                neuro_m = comb_text.str.contains("neurolog|neurosurg|vertigo|seizure|epilepsy|migraine|bppv|numbness|paralysis|stroke", regex=True)
                cardio_m = comb_text.str.contains("cardiolog|pulmonolog|cardiac|heart|hypertension|chest pain|asthma|bronchitis|pneumonia|shortness of breath", regex=True)
                ortho_m = comb_text.str.contains("orthoped|fracture|sprain|joint|knee|bone|spine|back pain|arthritis|ligament|tendon", regex=True)
                gi_m = comb_text.str.contains("gastroenterolog|gastrointestinal|acidity|gerd|diarrhea|constipation|liver|gallbladder|appendix|ulcer|bowel", regex=True)
                uro_m = comb_text.str.contains("urolog|nephrolog|kidney|urinary|uti|prostate|bladder|testes|testicle|scrotum", regex=True)
                ent_m = comb_text.str.contains("dermatolog|ophthalmolog|otolaryngolog|eye|ear|throat|skin|rash|acne|sinus|tonsil|vision", regex=True)
                psych_m = comb_text.str.contains("psychiatr|psycholog|anxiety|depression|panic|bipolar|mental health", regex=True)
                onco_m = comb_text.str.contains("oncolog|hematolog|cancer|tumor|biopsy|chemotherapy|leukemia|lymphoma", regex=True)
                surg_m = comb_text.str.contains("surgeon|surgery|operation|post-operative|incision", regex=True)
                peds_p = p_text.str.contains("pediatric|pediatrician|baby|infant|toddler|child", regex=True)

                department.loc[surg_m] = "SURGERY"
                department.loc[onco_m] = "ONCOLOGY_HEME"
                department.loc[psych_m] = "PSYCH"
                department.loc[ent_m] = "ENT_OPHTHALMO"
                department.loc[uro_m] = "RENAL_URO"
                department.loc[gi_m] = "GI"
                department.loc[ortho_m] = "ORTHO"
                department.loc[cardio_m] = "CARDIO_PULM"
                department.loc[neuro_m] = "NEURO"
                department.loc[obgyn_m] = "OBGYN"
                department.loc[peds_m | peds_p] = "PEDS"

                out_df = pd.DataFrame(
                    {
                        "dataset_source": self.dataset_source,
                        "raw_text": valid_df["input"],
                        "department": department,
                        "triage_level": None,
                        "language": "en",
                    }
                )
                out_df["id"] = [
                    f"chatdoctor_healthcaremagic::{pfile.name}::{i}"
                    for i in range(len(valid_df))
                ]

                yield out_df
