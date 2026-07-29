import pandas as pd
import re

DEPARTMENTS = {
    "Cardiovascular / Pulmonary": "CARDIO",
    "Neurology": "NEURO",
    "Orthopedic": "ORTHO",
    "Gastroenterology": "GI",
    "Urology": "URO",
    "Obstetrics / Gynecology": "OBGYN",
    "ENT - Otolaryngology": "ENT",
    "Ophthalmology": "OPHTH",
    "Emergency Room Reports": "ED",
    "Pediatrics - Neonatal": "PEDS",
    "Psychiatry / Psychology": "PSYCH",
    "Surgery": "SURG",
}

def map_specialty(raw_spec: str) -> tuple[str, str]:
    if not raw_spec:
        return "UNKNOWN", "low"
    code = DEPARTMENTS.get(raw_spec, "GEN_MED")
    conf = "high" if code != "GEN_MED" else "low"
    return code, conf

def score_severity(text: str) -> tuple[str, str]:
    text = str(text).lower()
    if re.search(r"cardiac arrest|unresponsive|no pulse", text):
        return "S1", "regex_heuristic"
    if re.search(r"chest pain|shortness of breath|severe bleeding", text):
        return "S2", "regex_heuristic"
    if re.search(r"fracture|laceration|fever", text):
        return "S3", "regex_heuristic"
    return "S4", "regex_heuristic"

def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) == 0:
        return df
        
    for idx, row in df.iterrows():
        # Map specialty
        if row['department_code'] == 'UNKNOWN' or not row['department_code']:
            code, conf = map_specialty(row['raw_medical_specialty'])
            df.at[idx, 'department_code'] = code
            df.at[idx, 'routing_confidence'] = conf
            
        # Score severity
        if row['severity_label'] == 'UNKNOWN' or not row['severity_label']:
            sev, src = score_severity(row['raw_text'])
            df.at[idx, 'severity_label'] = sev
            df.at[idx, 'severity_label_source'] = src
            
    return df
