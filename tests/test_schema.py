import pandas as pd
import pytest

from src.schema import validate_and_translate_schema

def test_specialist_only_rows():
    df = pd.DataFrame([
        {"raw_text": "Chest pain", "department": "ED"},
        {"raw_text": "Eye pain", "department": "ENT_OPHTHALMO"},
    ])
    result = validate_and_translate_schema(df)
    assert len(result) == 2
    assert result["triage_level"].isna().all()
    assert list(result["department"]) == ["ED", "ENT_OPHTHALMO"]

def test_severity_only_rows():
    df = pd.DataFrame([
        {"raw_text": "Chest pain", "triage_level": "S1"},
        {"raw_text": "Mild headache", "triage_level": "S5"},
    ])
    result = validate_and_translate_schema(df)
    assert len(result) == 2
    assert result["department"].isna().all()
    assert list(result["triage_level"]) == ["S1", "S5"]

def test_dual_labelled_rows():
    df = pd.DataFrame([
        {"raw_text": "Chest pain", "department": "ED", "triage_level": "S1"},
    ])
    result = validate_and_translate_schema(df)
    assert len(result) == 1
    assert result.iloc[0]["department"] == "ED"
    assert result.iloc[0]["triage_level"] == "S1"

def test_unlabeled_rows():
    df = pd.DataFrame([
        {"raw_text": "No labels at all"},
        {"raw_text": "Invalid labels", "department": "FAKE", "triage_level": "S9"},
    ])
    result = validate_and_translate_schema(df)
    assert len(result) == 0

def test_numeric_severity():
    df = pd.DataFrame([
        {"raw_text": "Chest pain", "triage_level": 1},
        {"raw_text": "Minor cut", "triage_level": 5.0},
        {"raw_text": "String numeric", "triage_level": "3"},
    ])
    result = validate_and_translate_schema(df)
    assert len(result) == 3
    assert list(result["triage_level"]) == ["S1", "S5", "S3"]

def test_s_prefixed_severity():
    df = pd.DataFrame([
        {"raw_text": "A", "triage_level": "S1"},
        {"raw_text": "B", "triage_level": "s2"},
        {"raw_text": "C", "triage_level": " S3 "},
    ])
    result = validate_and_translate_schema(df)
    assert len(result) == 3
    assert list(result["triage_level"]) == ["S1", "S2", "S3"]

def test_legacy_column_names():
    df1 = pd.DataFrame([{"text": "A", "department_code": "Emergency", "severity_heuristic": "1"}])
    res1 = validate_and_translate_schema(df1)
    assert len(res1) == 1
    assert list(res1["department"]) == ["ED"]
    assert list(res1["triage_level"]) == ["S1"]
    
    df2 = pd.DataFrame([{"text": "B", "specialist_label": "PEDS", "severity_label": "S2"}])
    res2 = validate_and_translate_schema(df2)
    assert len(res2) == 1
    assert list(res2["department"]) == ["PEDS"]
    assert list(res2["triage_level"]) == ["S2"]

def test_production_dataset_schema():
    df = pd.DataFrame([
        {"raw_text": "A", "department": "ED", "triage_level": 1},
        {"raw_text": "B", "department": "ORTHO", "triage_level": None},
        {"raw_text": "C", "department": None, "triage_level": 2},
        {"raw_text": "D", "department": None, "triage_level": None},
    ])
    result = validate_and_translate_schema(df)
    assert len(result) == 3
    assert list(result["raw_text"]) == ["A", "B", "C"]
    assert list(result["department"].replace({float("nan"): None})) == ["ED", "ORTHO", None]
    assert list(result["triage_level"].replace({float("nan"): None})) == ["S1", None, "S2"]
