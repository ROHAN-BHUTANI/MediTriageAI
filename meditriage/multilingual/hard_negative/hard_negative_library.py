"""Differential Diagnosis Knowledge Base Library for Hard Negative Generation.

Maps true disease presentations to clinically plausible differential diagnoses,
defining shared symptoms, distinguishing features, red flags, expected target department,
and acuity triage levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DifferentialDiagnosis:
    """Differential diagnosis candidate entry."""

    diff_id: str
    name: str
    target_department: str
    target_triage_level: str
    shared_symptoms: list[str]
    distinguishing_symptoms: list[str]
    red_flags: list[str]
    patient_wording: list[str]
    clinical_wording: list[str]


@dataclass
class TruePhenotypeDifferentialMapping:
    """Mapping from a true presentation to its differential diagnosis set."""

    phenotype_key: str
    primary_condition: str
    source_department: str
    differentials: list[DifferentialDiagnosis]


class DifferentialDiagnosisLibrary:
    """Knowledge base repository of differential diagnosis mappings."""

    def __init__(self):
        self._mappings: dict[str, TruePhenotypeDifferentialMapping] = {}
        self._initialize_library()

    def _initialize_library(self) -> None:
        # ── 1. CARDIOLOGY: Acute Myocardial Infarction ──────────────────────────
        self._register(
            TruePhenotypeDifferentialMapping(
                phenotype_key="ACS_AMI",
                primary_condition="Acute Myocardial Infarction",
                source_department="CARDIO_PULM",
                differentials=[
                    DifferentialDiagnosis(
                        diff_id="DIFF_GERD",
                        name="Gastroesophageal Reflux Disease (GERD)",
                        target_department="GENERAL",
                        target_triage_level="S4",
                        shared_symptoms=["substernal burning", "chest discomfort"],
                        distinguishing_symptoms=["acid reflux", "worsening postprandial", "relief with antacids"],
                        red_flags=["no radiation to arm", "no diaphoresis"],
                        patient_wording=[
                            "Substernal chest burning after heavy meal, relieved by antacid liquid.",
                            "Chest discomfort worsening when lying down flat after dinner.",
                        ],
                        clinical_wording=[
                            "Patient reports postprandial retrosternal burning pain responsive to antacids; EKG normal.",
                        ],
                    ),
                    DifferentialDiagnosis(
                        diff_id="DIFF_COSTOCHONDRITIS",
                        name="Costochondritis / Musculoskeletal Chest Pain",
                        target_department="ORTHO",
                        target_triage_level="S4",
                        shared_symptoms=["chest pain", "localized chest discomfort"],
                        distinguishing_symptoms=["reproducible on palpation", "sharp pain with deep inspiration", "focal wall tenderness"],
                        red_flags=["no dyspnea", "reproducible wall tenderness"],
                        patient_wording=[
                            "Sharp chest pain that gets worse when pressing directly on parasternal rib joints.",
                            "Chest wall discomfort exacerbated by twisting torso and deep breath.",
                        ],
                        clinical_wording=[
                            "Focal chest wall pain sharply reproducible upon palpation of costochondral junctions.",
                        ],
                    ),
                    DifferentialDiagnosis(
                        diff_id="DIFF_PANIC_ATTACK",
                        name="Panic Disorder / Acute Hyperventilation",
                        target_department="NEURO",
                        target_triage_level="S3",
                        shared_symptoms=["chest tightness", "shortness of breath", "palpitations"],
                        distinguishing_symptoms=["perioral tingling", "carpopedal spasm", "acute anxiety trigger", "tachypnea"],
                        red_flags=["perioral numbness", "normal cardiac troponin"],
                        patient_wording=[
                            "Sudden chest tightness with rapid breathing and tingling in fingers during anxiety episode.",
                            "Palpitations and feeling of doom with hyperventilation after acute stress.",
                        ],
                        clinical_wording=[
                            "Acute anxiety episode presenting with hyperventilation, perioral paresthesias, and chest tightness.",
                        ],
                    ),
                ],
            )
        )

        # ── 2. NEUROLOGY: Stroke / Cerebrovascular Accident ─────────────────────
        self._register(
            TruePhenotypeDifferentialMapping(
                phenotype_key="NEURO_STROKE",
                primary_condition="Cerebrovascular Accident (Stroke)",
                source_department="NEURO",
                differentials=[
                    DifferentialDiagnosis(
                        diff_id="DIFF_BELLS_PALSY",
                        name="Bell's Palsy (Idiopathic Facial Paralysis)",
                        target_department="NEURO",
                        target_triage_level="S3",
                        shared_symptoms=["facial droop", "inability to smile"],
                        distinguishing_symptoms=["forehead muscle weakness", "inability to wrinkle forehead", "sparing of limbs"],
                        red_flags=["no extremity weakness", "isolated cranial nerve VII"],
                        patient_wording=[
                            "Left facial droop involving forehead with inability to close eye, but normal arm strength.",
                            "Sudden unilateral facial weakness including eyebrow drooping without leg or arm weakness.",
                        ],
                        clinical_wording=[
                            "Isolated peripheral facial nerve palsy involving upper and lower face; limb power 5/5 bilaterally.",
                        ],
                    ),
                    DifferentialDiagnosis(
                        diff_id="DIFF_HYPOGLYCEMIA",
                        name="Hypoglycemia Mocking Stroke",
                        target_department="GENERAL",
                        target_triage_level="S2",
                        shared_symptoms=["confusion", "slurred speech", "weakness"],
                        distinguishing_symptoms=["blood glucose < 50 mg/dL", "diaphoresis", "rapid reversal after dextrose"],
                        red_flags=["diabetic on insulin", "resolved with glucose"],
                        patient_wording=[
                            "Sudden confusion and slurred speech in diabetic patient, fingerstick glucose 42 mg/dL.",
                            "Sweating and trembling with disorientation after missing lunch.",
                        ],
                        clinical_wording=[
                            "Neuroglycopenic presentation with altered mental status and slurred speech; CBG 45 mg/dL.",
                        ],
                    ),
                    DifferentialDiagnosis(
                        diff_id="DIFF_MIGRAINE_AURA",
                        name="Migraine with Visual Aura",
                        target_department="NEURO",
                        target_triage_level="S3",
                        shared_symptoms=["visual disturbance", "numbness", "headache"],
                        distinguishing_symptoms=["scintillating scotoma", "marching paresthesia", "throbbing headache follows"],
                        red_flags=["gradual visual aura march", "history of recurring migraines"],
                        patient_wording=[
                            "Gradual shimmering visual lights followed by right hand tingling and unilateral head pain.",
                            "Zig-zag visual lines lasting 20 minutes followed by throbbing headache.",
                        ],
                        clinical_wording=[
                            "Classic visual scotoma aura marching over 15 minutes followed by unilateral pulsatile otalgia/headache.",
                        ],
                    ),
                ],
            )
        )

        # ── 3. RESPIRATORY: Asthma Exacerbation ─────────────────────────────────
        self._register(
            TruePhenotypeDifferentialMapping(
                phenotype_key="RESP_ASTHMA",
                primary_condition="Asthma Exacerbation",
                source_department="CARDIO_PULM",
                differentials=[
                    DifferentialDiagnosis(
                        diff_id="DIFF_COPD_EXAC",
                        name="COPD Acute Exacerbation",
                        target_department="CARDIO_PULM",
                        target_triage_level="S2",
                        shared_symptoms=["shortness of breath", "wheezing", "cough"],
                        distinguishing_symptoms=["chronic smoking history", "increased purulent sputum", "barrel chest"],
                        red_flags=["heavy smoking history", "baseline productive cough"],
                        patient_wording=[
                            "Worsening breathlessness with thick yellow sputum in 65-year-old long-term smoker.",
                            "Increased shortness of breath and chronic cough exacerbation after cold exposure.",
                        ],
                        clinical_wording=[
                            "Acute-on-chronic COPD exacerbation with purulent sputum production and wheezing in long-term smoker.",
                        ],
                    ),
                    DifferentialDiagnosis(
                        diff_id="DIFF_PULM_EDEMA",
                        name="Acute Pulmonary Edema / Cardiac Asthma",
                        target_department="CARDIO_PULM",
                        target_triage_level="S1",
                        shared_symptoms=["shortness of breath", "wheezing", "chest tightness"],
                        distinguishing_symptoms=["pink frothy sputum", "bilateral lung crackles", "hypertension"],
                        red_flags=["pink frothy sputum", "pedal edema", "elevated JVP"],
                        patient_wording=[
                            "Sudden severe breathlessness coughing up pink frothy fluid with high blood pressure.",
                            "Severe orthopnea with bubbling sound in chest and leg swelling.",
                        ],
                        clinical_wording=[
                            "Acute cardiogenic pulmonary edema presenting with pink frothy sputum, bilateral basilar crackles, and HTN emergency.",
                        ],
                    ),
                ],
            )
        )

        # ── 4. ORTHOPEDICS: Bone Fracture ────────────────────────────────────────
        self._register(
            TruePhenotypeDifferentialMapping(
                phenotype_key="ORTHO_FRACTURE",
                primary_condition="Bone Fracture",
                source_department="ORTHO",
                differentials=[
                    DifferentialDiagnosis(
                        diff_id="DIFF_CONTUSION",
                        name="Severe Musculoskeletal Contusion",
                        target_department="ORTHO",
                        target_triage_level="S4",
                        shared_symptoms=["swelling", "localized pain", "bruising"],
                        distinguishing_symptoms=["intact bone alignment on X-ray", "able to bear partial weight", "soft tissue echymosis"],
                        red_flags=["no bony crepitus", "no deformity"],
                        patient_wording=[
                            "Blunt blow to thigh with painful bruising and swelling, able to walk slowly.",
                            "Direct impact trauma to forearm with localized soft tissue bruise but intact mobility.",
                        ],
                        clinical_wording=[
                            "Extremity soft tissue contusion with ecchymosis; intact bony alignment and motion.",
                        ],
                    ),
                ],
            )
        )

        # ── 5. PEDIATRICS: Pediatric Febrile Illness ─────────────────────────────
        self._register(
            TruePhenotypeDifferentialMapping(
                phenotype_key="PEDS_FEVER",
                primary_condition="Pediatric Febrile Illness",
                source_department="PEDS",
                differentials=[
                    DifferentialDiagnosis(
                        diff_id="DIFF_SIMPLE_SYNCOPE",
                        name="Pediatric Vasovagal Syncope",
                        target_department="PEDS",
                        target_triage_level="S3",
                        shared_symptoms=["transient collapse", "pallor"],
                        distinguishing_symptoms=["afebrile", "rapid recovery", "postural trigger"],
                        red_flags=["normal body temperature", "no post-ictal state"],
                        patient_wording=[
                            "Fainted briefly while standing in school assembly, completely normal upon lying down.",
                            "Sudden pale skin and brief fainting spell after blood draw, afebrile.",
                        ],
                        clinical_wording=[
                            "Vasovagal syncopal episode in pediatric patient triggered by prolonged standing; afebrile.",
                        ],
                    ),
                ],
            )
        )

    def _register(self, mapping: TruePhenotypeDifferentialMapping) -> None:
        self._mappings[mapping.phenotype_key] = mapping

    def get_differentials_for_text(self, text: str, department: str | None = None) -> list[DifferentialDiagnosis]:
        """Find matching differential diagnosis candidates for clinical text."""
        text_lower = text.lower()
        for mapping in self._mappings.values():
            if department and mapping.source_department.lower() == department.lower():
                return mapping.differentials
            if mapping.primary_condition.lower() in text_lower or any(s in text_lower for s in ["chest pain", "stroke", "fever", "focal weakness", "shortness of breath"]):
                return mapping.differentials
        # Fallback to cardiology differentials if no exact match
        return self._mappings["ACS_AMI"].differentials
