from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from .base import BaseAdapter


class PMCPatientsAdapter(BaseAdapter):
    """
    Adapter for the PMC Patients dataset.

    Mapping Strategy:
    - `patient` -> `raw_text`
    - Drop records with empty or NaN patient strings.
    - Yield chunks using chunksize to handle the 544MB file.
    """

    @property
    def dataset_source(self) -> str:
        return "pmc_patients"

    @property
    def version(self) -> str:
        return "1.1.0"

    def ingest(self, raw_path: str, chunk_size: int = 100000) -> Iterator[pd.DataFrame]:
        csv_path = Path(raw_path) / "PMC-Patients.csv"
        if not csv_path.exists():
            return

        for chunk_idx, chunk_df in enumerate(
            pd.read_csv(csv_path, chunksize=chunk_size)
        ):
            # Vectorized operations
            chunk_df["patient"] = chunk_df["patient"].fillna("").astype(str).str.strip()
            valid_mask = (chunk_df["patient"] != "") & (
                chunk_df["patient"].str.lower() != "nan"
            )
            valid_df = chunk_df[valid_mask]

            if len(valid_df) == 0:
                continue

            # Controlled keyword taxonomy mapping to SPECIALIST_CLASSES
            title_text = valid_df.get("title", pd.Series("", index=valid_df.index)).fillna("").astype(str)
            patient_text = valid_df["patient"].astype(str)
            combined_text = (title_text + " " + patient_text).str.lower()

            department = pd.Series("GEN_MED", index=valid_df.index, dtype=object)

            # Age < 18 or pediatric terms -> PEDS
            age_vals = pd.to_numeric(valid_df.get("age", pd.Series(None, index=valid_df.index)).astype(str).str.extract(r"^(\d+)", expand=False), errors="coerce")
            is_pediatric = (age_vals < 18) | combined_text.str.contains("pediatric|neonatal|infant|newborn", regex=True)

            # Taxonomies
            onco_mask = combined_text.str.contains("oncology|carcinoma|leukemia|lymphoma|metastatic|chemotherapy|melanoma|sarcoma|tumor|cancer|blastoma", regex=True)
            cardio_mask = combined_text.str.contains("cardiac|cardio|myocardial|heart|coronary|infarction|arrhythmia|pulmonary|pneumonia|embolism|respiratory failure|hypertension|aortic", regex=True)
            neuro_mask = combined_text.str.contains("neurolog|neurosurger|stroke|seizure|epilepsy|brain|cerebral|aneurysm|encephalopathy|dementia|parkinson|meningitis", regex=True)
            ortho_mask = combined_text.str.contains("orthoped|fracture|dislocation|femur|tibia|joint|knee|hip replacement|spine|spinal cord|arthroplasty|scoliosis", regex=True)
            gi_mask = combined_text.str.contains("gastro|colon|liver|hepatic|pancreatic|bowel|gastric|esophag|appendicitis|cirrhosis|cholecystitis|crohn", regex=True)
            obgyn_mask = combined_text.str.contains("gynecolog|obstetric|pregnancy|pregnant|ovarian|uterine|cervical cancer|placenta|ectopic|endometriosis|caesarean", regex=True)
            uro_mask = combined_text.str.contains("nephrolog|urolog|kidney|renal|dialysis|bladder|prostate|glomerulonephritis|nephrolithiasis", regex=True)
            ent_mask = combined_text.str.contains("ophthalmo|dermatolog|otolaryngolog|cornea|retinal|cataract|glaucoma|psoriasis|eczema|sinusitis|tonsillitis", regex=True)
            psych_mask = combined_text.str.contains("psychiatr|schizophrenia|bipolar|depression|suicidal|psychosis|anorexia", regex=True)
            surg_mask = combined_text.str.contains("surgeon|surgery|resection|transplantation|laparotom|mastectomy|graft|postoperative", regex=True)

            department.loc[surg_mask] = "SURGERY"
            department.loc[psych_mask] = "PSYCH"
            department.loc[ent_mask] = "ENT_OPHTHALMO"
            department.loc[uro_mask] = "RENAL_URO"
            department.loc[obgyn_mask] = "OBGYN"
            department.loc[gi_mask] = "GI"
            department.loc[ortho_mask] = "ORTHO"
            department.loc[neuro_mask] = "NEURO"
            department.loc[cardio_mask] = "CARDIO_PULM"
            department.loc[onco_mask] = "ONCOLOGY_HEME"
            department.loc[is_pediatric] = "PEDS"

            out_df = pd.DataFrame(
                {
                    "dataset_source": self.dataset_source,
                    "raw_text": valid_df["patient"],
                    "department": department,
                    "triage_level": None,
                    "language": "en",
                }
            )
            out_df["id"] = [
                f"pmc_patients::{chunk_idx}::{i}" for i in range(len(valid_df))
            ]

            yield out_df
