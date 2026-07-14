"""Deterministic specialty routing map for MediTriageAI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Department:
    code: str
    name: str


DEPARTMENTS: dict[str, Department] = {
    "ED": Department("ED", "Emergency Medicine"),
    "CARDIO_PULM": Department("CARDIO_PULM", "Cardiovascular & Pulmonary"),
    "GI": Department("GI", "Gastroenterology"),
    "NEURO": Department("NEURO", "Neurology & Neurosurgery"),
    "ORTHO": Department("ORTHO", "Orthopedics & Physical Medicine"),
    "SURGERY": Department("SURGERY", "General & Specialty Surgery"),
    "OBGYN": Department("OBGYN", "Obstetrics & Gynecology"),
    "PEDS": Department("PEDS", "Pediatrics"),
    "PSYCH": Department("PSYCH", "Psychiatry & Mental Health"),
    "ONCOLOGY_HEME": Department("ONCOLOGY_HEME", "Oncology & Hematology"),
    "RENAL_URO": Department("RENAL_URO", "Nephrology & Urology"),
    "ENT_OPHTHALMO": Department("ENT_OPHTHALMO", "ENT, Ophthalmology & Dermatology"),
    "GEN_MED": Department("GEN_MED", "General / Internal Medicine (catch-all)"),
}

RAW_TO_DEPARTMENT: dict[str, str] = {
    "Emergency Room Reports": "ED",
    "Cardiovascular / Pulmonary": "CARDIO_PULM",
    "Sleep Medicine": "CARDIO_PULM",
    "Gastroenterology": "GI",
    "Bariatrics": "GI",
    "Diets and Nutritions": "GI",
    "Neurology": "NEURO",
    "Neurosurgery": "NEURO",
    "Orthopedic": "ORTHO",
    "Physical Medicine - Rehab": "ORTHO",
    "Podiatry": "ORTHO",
    "Chiropractic": "ORTHO",
    "Surgery": "SURGERY",
    "Cosmetic / Plastic Surgery": "SURGERY",
    "Obstetrics / Gynecology": "OBGYN",
    "Pediatrics - Neonatal": "PEDS",
    "Psychiatry / Psychology": "PSYCH",
    "Hematology - Oncology": "ONCOLOGY_HEME",
    "Nephrology": "RENAL_URO",
    "Urology": "RENAL_URO",
    "ENT - Otolaryngology": "ENT_OPHTHALMO",
    "Ophthalmology": "ENT_OPHTHALMO",
    "Dermatology": "ENT_OPHTHALMO",
    "Allergy / Immunology": "ENT_OPHTHALMO",
    "General Medicine": "GEN_MED",
    "Consult - History and Phy.": "GEN_MED",
    "Endocrinology": "GEN_MED",
    "Rheumatology": "GEN_MED",
    "Pain Management": "GEN_MED",
    "IME-QME-Work Comp etc.": "GEN_MED",
    "Radiology": "GEN_MED",
    "SOAP / Chart / Progress Notes": "GEN_MED",
    "Discharge Summary": "GEN_MED",
    "Office Notes": "GEN_MED",
    "Letters": "GEN_MED",
    "Lab Medicine - Pathology": "GEN_MED",
    "Autopsy": "GEN_MED",
    "Hospice - Palliative Care": "GEN_MED",
    "Speech - Language": "GEN_MED",
    "Dentistry": "GEN_MED",
}

_LOW_CONFIDENCE_RAW_LABELS: frozenset[str] = frozenset(
    {"SOAP / Chart / Progress Notes", "Discharge Summary", "Office Notes", "Letters", "Lab Medicine - Pathology", "Autopsy", "Hospice - Palliative Care", "Speech - Language", "Dentistry"}
)


def map_specialty(raw_specialty: str) -> tuple[str, str]:
    key = raw_specialty.strip()
    if key not in RAW_TO_DEPARTMENT:
        raise KeyError(f"Unrecognized raw medical_specialty value: {raw_specialty!r}.")
    department_code = RAW_TO_DEPARTMENT[key]
    confidence = "low" if key in _LOW_CONFIDENCE_RAW_LABELS else "high"
    return department_code, confidence


def all_raw_labels_mapped(raw_labels: list[str]) -> bool:
    return all(label.strip() in RAW_TO_DEPARTMENT for label in raw_labels)
