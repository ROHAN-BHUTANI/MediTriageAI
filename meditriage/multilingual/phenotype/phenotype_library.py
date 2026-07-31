"""Clinical Phenotype Knowledge Base Library.

Defines extensible phenotype definitions across 8 clinical specialties:
  - Cardiology
  - Neurology
  - Respiratory
  - Orthopedics
  - Pediatrics
  - ENT (Otolaryngology)
  - Emergency Medicine
  - General Medicine

Each phenotype specifies core symptoms, optional manifestations, clinical wording,
patient wording, triage phrasing, and explicit contraindicated symptoms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhenotypeDefinition:
    """Definition of a clinical disease phenotype."""

    phenotype_id: str
    name: str
    specialty: str
    department_mapping: list[str]
    core_symptoms: list[str]
    optional_symptoms: list[str]
    supporting_symptoms: list[str]
    rare_manifestations: list[str]
    patient_wording: list[str]
    clinical_wording: list[str]
    triage_wording: list[str]
    contraindicated_symptoms: list[str] = field(default_factory=list)


class PhenotypeLibrary:
    """Knowledge base repository of clinical phenotypes."""

    def __init__(self):
        self._phenotypes: dict[str, PhenotypeDefinition] = {}
        self._initialize_library()

    def _initialize_library(self) -> None:
        # ── 1. CARDIOLOGY ───────────────────────────────────────────────────
        self._register(
            PhenotypeDefinition(
                phenotype_id="CARD_ACS",
                name="Acute Coronary Syndrome / Myocardial Ischemia",
                specialty="Cardiology",
                department_mapping=["CARDIO_PULM", "Cardiology", "Emergency Medicine"],
                core_symptoms=["chest pain", "chest tightness", "substernal pressure", "retrosternal discomfort"],
                optional_symptoms=["radiation to left arm", "radiation to jaw", "diaphoresis", "shortness of breath", "nausea"],
                supporting_symptoms=["sweating", "left arm heaviness", "exertional chest discomfort", "dyspnea"],
                rare_manifestations=["epigastric burning", "isolated arm heaviness"],
                patient_wording=[
                    "Chest tightness with diaphoresis.",
                    "Pressure-like substernal discomfort with nausea.",
                    "Severe squeezing chest pain and breathlessness.",
                    "Retrosternal pain with sweating.",
                    "Pain extending into jaw and left shoulder.",
                    "Chest discomfort after exertion.",
                    "Burning substernal discomfort with dyspnea.",
                    "Left arm heaviness with chest pressure.",
                ],
                clinical_wording=[
                    "Patient reports acute substernal chest pressure with diaphoresis and radiation to left shoulder.",
                    "Complaint of exertional retrosternal tightness associated with dyspnea and nausea.",
                ],
                triage_wording=[
                    "[CARDIO TRIAGE - S2] Pt c/o substernal CP radiating to left arm w/ diaphoresis. Vitals monitored.",
                ],
                contraindicated_symptoms=["bilateral facial paralysis", "petechial rash", "joint swelling"],
            )
        )

        self._register(
            PhenotypeDefinition(
                phenotype_id="CARD_HF",
                name="Acute Heart Failure Exacerbation",
                specialty="Cardiology",
                department_mapping=["CARDIO_PULM", "Cardiology"],
                core_symptoms=["shortness of breath", "orthopnea", "pedal edema", "paroxysmal nocturnal dyspnea"],
                optional_symptoms=["fatigue", "jugular venous distension", "crackles", "weight gain"],
                supporting_symptoms=["bilateral leg swelling", "breathlessness on lying flat"],
                rare_manifestations=["right upper quadrant tenderness"],
                patient_wording=[
                    "Shortness of breath getting worse when lying flat.",
                    "Bilateral leg swelling with severe breathlessness on exertion.",
                    "Waking up gasping for air at night with swollen ankles.",
                ],
                clinical_wording=[
                    "Patient evaluates with acute orthopnea, bilateral lower extremity edema, and exertional dyspnea.",
                ],
                triage_wording=[
                    "[CARDIO TRIAGE - S2] Pt c/o progressive dyspnea and pedal edema x 3 days.",
                ],
                contraindicated_symptoms=["migraine", "focal weakness", "otitis media"],
            )
        )

        # ── 2. NEUROLOGY ────────────────────────────────────────────────────
        self._register(
            PhenotypeDefinition(
                phenotype_id="NEURO_MIGRAINE",
                name="Migraine / Acute Vascular Headache",
                specialty="Neurology",
                department_mapping=["NEURO", "Neurology"],
                core_symptoms=["headache", "throbbing head pain", "unilateral headache", "severe sirdard"],
                optional_symptoms=["photophobia", "phonophobia", "nausea", "visual aura", "vomiting"],
                supporting_symptoms=["sensitivity to light", "pulsatile headache"],
                rare_manifestations=["transient scintillating scotoma"],
                patient_wording=[
                    "Throbbing unilateral headache with light sensitivity and nausea.",
                    "Severe pulsating head pain with vomiting and blurred vision aura.",
                    "Intense one-sided headache exacerbated by bright light.",
                ],
                clinical_wording=[
                    "Patient presents with severe throbbing unilateral headache accompanied by photophobia and nausea.",
                ],
                triage_wording=[
                    "[NEURO TRIAGE - S3] Pt c/o acute unilateral throbbing headache w/ photophobia.",
                ],
                contraindicated_symptoms=["chest pain radiating to arm", "pregnancy", "diarrhea"],
            )
        )

        self._register(
            PhenotypeDefinition(
                phenotype_id="NEURO_CVA",
                name="Cerebrovascular Accident / TIA",
                specialty="Neurology",
                department_mapping=["NEURO", "Neurology", "Emergency Medicine"],
                core_symptoms=["focal weakness", "facial droop", "slurred speech", "hemiparesis"],
                optional_symptoms=["numbness", "ataxia", "confusion", "dizziness", "aphasia"],
                supporting_symptoms=["arm weakness", "difficulty speaking", "sudden loss of balance"],
                rare_manifestations=["isolated dysarthria"],
                patient_wording=[
                    "Sudden right-sided arm weakness and difficulty speaking clearly.",
                    "Facial droop on left side with slurred speech starting 1 hour ago.",
                    "Sudden numbness in right arm and leg with loss of balance.",
                ],
                clinical_wording=[
                    "Stroke Code Activation: Acute onset right hemiparesis and expressive aphasia.",
                ],
                triage_wording=[
                    "[STROKE TRIAGE - S1] Sudden onset facial droop and right arm weakness. LKM < 2 hours.",
                ],
                contraindicated_symptoms=["wheezing", "pedal edema", "ear discharge"],
            )
        )

        # ── 3. RESPIRATORY ──────────────────────────────────────────────────
        self._register(
            PhenotypeDefinition(
                phenotype_id="RESP_ASTHMA",
                name="Asthma / COPD Exacerbation",
                specialty="Respiratory",
                department_mapping=["CARDIO_PULM", "Respiratory"],
                core_symptoms=["shortness of breath", "wheezing", "chest tightness", "cough"],
                optional_symptoms=["use of accessory muscles", "tachypnea", "sputum production"],
                supporting_symptoms=["breathlessness", "whistling sound when breathing"],
                rare_manifestations=["silent chest"],
                patient_wording=[
                    "Severe wheezing and chest tightness not responding to inhaler.",
                    "Shortness of breath with whistling sound in chest.",
                    "Coughing and breathlessness worsening at night.",
                ],
                clinical_wording=[
                    "Patient evaluates with acute bronchospasm, bilateral expiratory wheezing, and dyspnea.",
                ],
                triage_wording=[
                    "[RESP TRIAGE - S2] Acute asthma exacerbation w/ expiratory wheezing and distress.",
                ],
                contraindicated_symptoms=["joint dislocation", "hematuria"],
            )
        )

        # ── 4. ORTHOPEDICS ──────────────────────────────────────────────────
        self._register(
            PhenotypeDefinition(
                phenotype_id="ORTHO_FRACTURE",
                name="Acute Fracture / Musculoskeletal Trauma",
                specialty="Orthopedics",
                department_mapping=["ORTHO", "Orthopedics"],
                core_symptoms=["severe pain", "swelling", "inability to bear weight", "deformity"],
                optional_symptoms=["bruising", "tenderness", "limited range of motion", "crepitus"],
                supporting_symptoms=["chot", "toot", "inability to bend limb"],
                rare_manifestations=["compartment pressure tenderness"],
                patient_wording=[
                    "Fell down stairs with severe wrist pain, swelling, and deformity.",
                    "Inability to bear weight on right ankle after twisting injury.",
                    "Severe localized bone pain and swelling following direct trauma.",
                ],
                clinical_wording=[
                    "Patient presenting post-fall with focal tenderness, swelling, and restricted range of motion.",
                ],
                triage_wording=[
                    "[ORTHO TRIAGE - S3] Closed extremity trauma w/ localized edema and point tenderness.",
                ],
                contraindicated_symptoms=["hemoptysis", "jaundice", "diplopia"],
            )
        )

        # ── 5. PEDIATRICS ───────────────────────────────────────────────────
        self._register(
            PhenotypeDefinition(
                phenotype_id="PEDS_FEVER",
                name="Pediatric Febrile Illness",
                specialty="Pediatrics",
                department_mapping=["PEDS", "Pediatrics"],
                core_symptoms=["fever", "high temperature", "irritability", "lethargy"],
                optional_symptoms=["poor feeding", "cough", "rhinorrhea", "vomiting", "chills"],
                supporting_symptoms=["High-grade fever", "Low-grade fever", "Fever with chills",
                                     "Persistent fever", "Intermittent fever", "Fever associated with body ache",
                                     "Fever and malaise", "Fever with rigors"],
                rare_manifestations=["febrile seizure"],
                patient_wording=[
                    "High-grade fever of 103F in 2-year-old child with poor feeding and lethargy.",
                    "Persistent fever with chills and vomiting in 4-year-old.",
                    "Fever associated with body ache and irritability since yesterday.",
                    "Fever with rigors and refusal to take fluids.",
                ],
                clinical_wording=[
                    "Pediatric patient presenting with pyrexia of 102.8F, malaise, and mild dehydration signs.",
                ],
                triage_wording=[
                    "[PEDS TRIAGE - S3] Febrile pediatric patient w/ temperature 103F and lethargy.",
                ],
                contraindicated_symptoms=["angina", "exertional dyspnea", "pregnancy"],
            )
        )

        # ── 6. ENT ──────────────────────────────────────────────────────────
        self._register(
            PhenotypeDefinition(
                phenotype_id="ENT_OTITIS",
                name="Acute Otitis Media / Otalgia",
                specialty="ENT",
                department_mapping=["ENT", "General Medicine"],
                core_symptoms=["ear pain", "otalgia", "ear fullness", "hearing difficulty"],
                optional_symptoms=["fever", "ear discharge", "otorrhea", "tinnitus"],
                supporting_symptoms=["sharp pain in ear", "blocked ear feeling"],
                rare_manifestations=["mastoid tenderness"],
                patient_wording=[
                    "Severe ear pain with feeling of fullness and mild fever.",
                    "Sharp otalgia with yellowish ear discharge.",
                    "Persistent throbbing pain in right ear with reduced hearing.",
                ],
                clinical_wording=[
                    "Patient evaluates with acute right otalgia, tympanic membrane erythema, and otorrhea.",
                ],
                triage_wording=[
                    "[ENT TRIAGE - S4] Pt c/o right ear pain and fullness x 2 days.",
                ],
                contraindicated_symptoms=["chest pressure", "paralysis", "hemiparesis"],
            )
        )

        # ── 7. EMERGENCY MEDICINE ───────────────────────────────────────────
        self._register(
            PhenotypeDefinition(
                phenotype_id="EMERG_ABDOMEN",
                name="Acute Abdomen / Appendicitis",
                specialty="Emergency Medicine",
                department_mapping=["Emergency Medicine", "GENERAL", "General Medicine"],
                core_symptoms=["acute abdominal pain", "right lower quadrant pain", "rebound tenderness"],
                optional_symptoms=["nausea", "vomiting", "fever", "anorexia", "guarding"],
                supporting_symptoms=["pet mein tez dard", "pain starting near navel and moving to RLQ"],
                rare_manifestations=["obturator sign positive"],
                patient_wording=[
                    "Severe right lower quadrant abdominal pain with nausea and vomiting.",
                    "Abdominal pain starting around navel and shifting to right lower side with fever.",
                    "Intense abdominal tenderness with inability to walk straight.",
                ],
                clinical_wording=[
                    "Patient exhibits RLQ abdominal tenderness, localized guarding, and low-grade pyrexia.",
                ],
                triage_wording=[
                    "[EMERG TRIAGE - S2] Acute RLQ abdominal pain w/ rebound tenderness and emesis.",
                ],
                contraindicated_symptoms=["stridor", "diplopia", "otitis"],
            )
        )

        # ── 8. GENERAL MEDICINE ──────────────────────────────────────────────
        self._register(
            PhenotypeDefinition(
                phenotype_id="GEN_GASTRO",
                name="Acute Gastroenteritis",
                specialty="General Medicine",
                department_mapping=["General Medicine", "GENERAL"],
                core_symptoms=["watery diarrhea", "vomiting", "abdominal cramps", "nausea"],
                optional_symptoms=["fever", "dehydration", "weakness", "tenesmus"],
                supporting_symptoms=["pet dard", "dast", "loose motions"],
                rare_manifestations=["electrolyte muscle cramps"],
                patient_wording=[
                    "Multiple episodes of watery diarrhea and vomiting with abdominal cramps.",
                    "Loose motions and nausea associated with generalized weakness.",
                    "Abdominal cramping and frequent diarrhea after eating outside food.",
                ],
                clinical_wording=[
                    "Patient presents with acute gastroenteritis, frequent unformed stools, and mild volume depletion.",
                ],
                triage_wording=[
                    "[GEN TRIAGE - S3] Acute gastroenteritis w/ nausea, emesis, and loose motions x 24h.",
                ],
                contraindicated_symptoms=["chest tightness radiating to arm", "hemiplegia"],
            )
        )

    def _register(self, p: PhenotypeDefinition) -> None:
        self._phenotypes[p.phenotype_id] = p

    def get_all_phenotypes(self) -> list[PhenotypeDefinition]:
        return list(self._phenotypes.values())

    def get_phenotype_by_id(self, p_id: str) -> PhenotypeDefinition | None:
        return self._phenotypes.get(p_id)

    def match_phenotype(self, text: str, department: str | None = None) -> PhenotypeDefinition | None:
        """Find the best matching phenotype definition for a clinical text and department."""
        text_lower = text.lower()
        best_match = None
        best_score = 0

        for p in self._phenotypes.values():
            score = 0
            # Department match bonus
            if department and any(d.lower() == department.lower() for d in p.department_mapping):
                score += 3

            # Core symptom matches
            for core in p.core_symptoms:
                if core in text_lower:
                    score += 5

            # Optional symptom matches
            for opt in p.optional_symptoms + p.supporting_symptoms:
                if opt in text_lower:
                    score += 2

            if score > best_score and score >= 5:
                best_score = score
                best_match = p

        # Fallback to first phenotype matching department if no exact text match
        if not best_match and department:
            for p in self._phenotypes.values():
                if any(d.lower() == department.lower() for d in p.department_mapping):
                    return p

        return best_match or self._phenotypes["CARD_ACS"]
