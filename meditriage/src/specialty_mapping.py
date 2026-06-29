"""
specialty_mapping.py
---------------------
Deterministic, auditable mapping from MTSamples' 40 raw `medical_specialty`
values to the 13-department condensed routing schema defined in
docs/01_clinical_taxonomy.md (Section 3).

This mapping is intentionally a plain dict, not a learned model — it must be
trivially reviewable by a clinician without running any code. Every raw
specialty maps to exactly one department code, and every department carries
a `routing_confidence` of "high" (content directly implies the department)
or "low" (catch-all / document-type artifact, content-based specialty is not
actually determinable from the label alone).

This module has NO side effects: importing it does not read or write data.
"""

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

# Raw MTSamples `medical_specialty` string (as it appears in the CSV, including
# leading/trailing whitespace quirks from the source scrape) -> department code.
# Built from the 40 categories observed in mtsamples.csv (n=4999 rows).
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
    # --- GEN_MED catch-all: genuine general medicine content ---
    "General Medicine": "GEN_MED",
    "Consult - History and Phy.": "GEN_MED",
    "Endocrinology": "GEN_MED",
    "Rheumatology": "GEN_MED",
    "Pain Management": "GEN_MED",
    "IME-QME-Work Comp etc.": "GEN_MED",
    "Radiology": "GEN_MED",
    # --- GEN_MED catch-all: document-type artifacts, NOT a content specialty ---
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

# Routing confidence: "high" if the raw label directly and unambiguously implies
# clinical content for that department; "low" if it is a document-type artifact
# or a heterogeneous catch-all where true specialty cannot be inferred from the
# label alone. See docs/01_clinical_taxonomy.md Section 3 design notes.
_LOW_CONFIDENCE_RAW_LABELS: frozenset[str] = frozenset(
    {
        "SOAP / Chart / Progress Notes",
        "Discharge Summary",
        "Office Notes",
        "Letters",
        "Lab Medicine - Pathology",
        "Autopsy",
        "Hospice - Palliative Care",
        "Speech - Language",
        "Dentistry",
    }
)


def map_specialty(raw_specialty: str) -> tuple[str, str]:
    """
    Map a raw MTSamples `medical_specialty` string to a (department_code,
    routing_confidence) tuple.

    Parameters
    ----------
    raw_specialty:
        The raw value from the MTSamples `medical_specialty` column. Leading/
        trailing whitespace is stripped before lookup (the source CSV has
        inconsistent padding).

    Returns
    -------
    (department_code, routing_confidence)
        department_code is one of the 13 keys in DEPARTMENTS.
        routing_confidence is "high" or "low".

    Raises
    ------
    KeyError
        If `raw_specialty` (after stripping) is not a recognized MTSamples
        category. This is intentional: silent fallback to a default department
        would hide data-quality problems. Callers ingesting new/unexpected
        data should catch this and route to a quarantine list for manual
        review rather than guessing.
    """
    key = raw_specialty.strip()
    if key not in RAW_TO_DEPARTMENT:
        raise KeyError(
            f"Unrecognized raw medical_specialty value: {raw_specialty!r}. "
            "This value is not in the documented 40-category MTSamples "
            "schema (docs/01_clinical_taxonomy.md Section 3). Do not guess "
            "a department for it — route to quarantine for manual review."
        )
    department_code = RAW_TO_DEPARTMENT[key]
    confidence = "low" if key in _LOW_CONFIDENCE_RAW_LABELS else "high"
    return department_code, confidence


def all_raw_labels_mapped(raw_labels: list[str]) -> bool:
    """
    Sanity-check helper: returns True iff every label in `raw_labels` (after
    stripping whitespace) has a mapping. Intended to be called once against
    the full set of distinct values in a fresh MTSamples export, so a schema
    drift (e.g. a new specialty category added upstream) fails loudly instead
    of silently mis-routing rows.
    """
    return all(label.strip() in RAW_TO_DEPARTMENT for label in raw_labels)


if __name__ == "__main__":
    # Self-check: 40 raw labels in, 13 departments out, no orphans.
    print(f"Departments defined: {len(DEPARTMENTS)}")
    print(f"Raw labels mapped:    {len(RAW_TO_DEPARTMENT)}")
    dept_counts: dict[str, int] = {}
    for dept_code in RAW_TO_DEPARTMENT.values():
        dept_counts[dept_code] = dept_counts.get(dept_code, 0) + 1
    for code, count in sorted(dept_counts.items()):
        print(f"  {code:15s} <- {count} raw labels")
