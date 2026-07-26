import csv
import re
from pathlib import Path
from typing import List, Dict

# Simple medical entity list for demonstration; in production this would be a comprehensive ontology
MEDICAL_ENTITIES = {
    "pain",
    "fever",
    "cough",
    "headache",
    "nausea",
    "vomiting",
    "dyspnea",
    "chest pain",
    "shortness of breath",
    "diarrhea",
    "fatigue",
    "dizziness",
}

ANATOMICAL_LOCATIONS = {
    "head",
    "chest",
    "abdomen",
    "leg",
    "arm",
    "back",
    "neck",
    "shoulder",
    "wrist",
    "knee",
}

DURATION_PATTERNS = [
    r"\b\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    r"\b(acute|chronic|subacute)\b",
]

class ClinicalSafetyValidator:
    """Validate that clinical semantics are preserved after augmentation.

    The validator checks five core aspects of a clinical note:
    1. Chief complaint remains present.
    2. Symptom polarity (positive/negative) is unchanged.
    3. Anatomical location mentions are unchanged.
    4. Duration semantics are preserved.
    5. Medical entities are preserved.

    It produces a CSV report with one row per record indicating pass/fail and reasons.
    """

    def __init__(self, original_records: List[Dict[str, str]], augmented_records: List[Dict[str, str]]):
        self.original = original_records
        self.augmented = augmented_records
        self.report_path = Path("data/processed/enriched/clinical_validation_report.csv")
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _extract_entities(text: str) -> set:
        lowered = text.lower()
        return {entity for entity in MEDICAL_ENTITIES if entity in lowered}

    @staticmethod
    def _extract_locations(text: str) -> set:
        lowered = text.lower()
        return {loc for loc in ANATOMICAL_LOCATIONS if loc in lowered}

    @staticmethod
    def _extract_durations(text: str) -> List[str]:
        durations = []
        for pat in DURATION_PATTERNS:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            durations.extend(matches)
        return durations

    @staticmethod
    def _extract_polarity(text: str) -> str:
        lowered = text.lower()
        if re.search(r"\b(no|denies|without|absent)\b", lowered):
            return "negative"
        return "positive"

    def validate(self) -> None:
        with self.report_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["record_id", "passed", "reasons"])
            writer.writeheader()
            for orig, aug in zip(self.original, self.augmented):
                reasons = []
                if orig.get("chief_complaint") != aug.get("chief_complaint"):
                    reasons.append("Chief complaint changed")
                if self._extract_polarity(orig.get("text", "")) != self._extract_polarity(aug.get("text", "")):
                    reasons.append("Symptom polarity changed")
                if self._extract_locations(orig.get("text", "")) != self._extract_locations(aug.get("text", "")):
                    reasons.append("Anatomical location changed")
                if set(self._extract_durations(orig.get("text", ""))) != set(self._extract_durations(aug.get("text", ""))):
                    reasons.append("Duration semantics changed")
                if self._extract_entities(orig.get("text", "")) != self._extract_entities(aug.get("text", "")):
                    reasons.append("Medical entities changed")
                passed = len(reasons) == 0
                writer.writerow({"record_id": aug.get("id", ""), "passed": passed, "reasons": "; ".join(reasons)})

if __name__ == "__main__":
    import pandas as pd
    original_path = Path("data/processed/improved/dataset_improved.csv")
    augmented_path = Path("data/processed/enriched/synthetic_samples.csv")
    if not original_path.is_file() or not augmented_path.is_file():
        raise FileNotFoundError("Required dataset files not found.")
    orig_df = pd.read_csv(original_path).to_dict(orient="records")
    aug_df = pd.read_csv(augmented_path).to_dict(orient="records")
    validator = ClinicalSafetyValidator(orig_df, aug_df)
    validator.validate()
    print(f"Clinical safety validation report written to {validator.report_path}")
