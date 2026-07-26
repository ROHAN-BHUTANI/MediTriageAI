# scripts/dataset_enrichment_engine.py
"""Dataset Enrichment Engine

This script orchestrates deterministic synthetic data generation using the
configured transformation plugins, applies diversity scoring thresholds, and
runs quality‑gate validators.

Outputs:
- data/processed/enriched/dataset_enriched.csv (original + synthetic)
- data/processed/enriched/synthetic_samples.csv (synthetic only)
- data/processed/enriched/synthetic_diversity_report.csv
- data/processed/enriched/clinical_validation_report.csv
- data/processed/enriched/duplicate_validation_report.csv
- data/processed/enriched/generation_statistics.json
- data/processed/enriched/enrichment_manifest.json
- data/processed/enriched/dataset_enrichment_report.md
"""
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import random
import pandas as pd

# Project imports
from src.registry import load_plugins, CONFIG_PATH
from src.diversity_scorer import score_sample
from src.clinical_safety_validator import ClinicalSafetyValidator
from src.duplicate_validator import DuplicateValidator
import yaml

# Load configuration
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
SEED = cfg.get("seed", 42)
THRESHOLDS = cfg.get("diversity_thresholds", {
    "lexical_diversity": 0.30,
    "edit_distance_ratio": 0.10,
    "token_overlap": 0.90,
    "novelty_score": 0.20,
})

# Paths
ORIG_PATH = Path("data/processed/improved/dataset_improved.csv")
ENRICHED_DIR = Path("data/processed/enriched")
SYNTHETIC_PATH = ENRICHED_DIR / "synthetic_samples.csv"
ENRICHED_PATH = ENRICHED_DIR / "dataset_enriched.csv"
DIVERSITY_REPORT = ENRICHED_DIR / "synthetic_diversity_report.csv"
CLINICAL_REPORT = ENRICHED_DIR / "clinical_validation_report.csv"
DUPLICATE_REPORT = ENRICHED_DIR / "duplicate_validation_report.csv"
GEN_STATS = ENRICHED_DIR / "generation_statistics.json"
MANIFEST = ENRICHED_DIR / "enrichment_manifest.json"
REPORT_MD = ENRICHED_DIR / "dataset_enrichment_report.md"

def deterministic_id(parent_id: str, specialty: str, seq: int) -> str:
    """Generate synthetic ID of form SYN_<SPECIALTY>_<PARENTID>_<SEQ>.
    parent_id may contain non‑numeric characters – we keep it as‑is.
    """
    return f"SYN_{specialty.upper()}_{parent_id}_{seq:04d}"

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

def main(dry_run: bool = False):
    # Ensure output directory exists
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    # Load original dataset
    orig_df = pd.read_csv(ORIG_PATH)
    required_cols = {"id", "text", "department_code"}
    if not required_cols.issubset(set(orig_df.columns)):
        missing = required_cols - set(orig_df.columns)
        raise ValueError(f"Original dataset missing columns: {missing}")

    plugins = load_plugins()
    rng = random.Random(SEED)

    synthetic_records = []
    diversity_rows = []
    generation_counter: Dict[str, int] = {}
    corpus_texts = orig_df["text"].astype(str).tolist()

    for _, row in orig_df.iterrows():
        parent_id = str(row["id"]).strip()
        specialty = str(row.get("department_code", "UNKNOWN"))
        base_text = str(row["text"]).strip()
        parent_hash = hash_text(base_text)
        seq = generation_counter.get(parent_id, 0) + 1
        generation_counter[parent_id] = seq
        synth_id = deterministic_id(parent_id, specialty, seq)

        transformed_text = base_text
        provenance = {
            "synthetic_id": synth_id,
            "parent_id": parent_id,
            "parent_hash": parent_hash,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "seed": SEED,
            "plugin_chain": [],
            "reversible": [],
        }
        for plugin in plugins:
            transformed_text, meta = plugin.apply(transformed_text, rng)
            provenance["plugin_chain"].append(plugin.name)
            reversible_flag = getattr(plugin, "reversible", False)
            provenance["reversible"].append(reversible_flag)
        scores = score_sample(transformed_text, base_text, corpus_texts)
        edit_ratio = scores["edit_distance"] / max(len(base_text), len(transformed_text), 1)
        scores["edit_distance_ratio"] = edit_ratio
        passed = (
            scores["lexical_diversity"] >= THRESHOLDS.get("lexical_diversity", 0.0)
            and edit_ratio >= THRESHOLDS.get("edit_distance_ratio", 0.0)
            and scores["token_overlap"] <= THRESHOLDS.get("token_overlap", 1.0)
            and scores["novelty_score"] >= THRESHOLDS.get("novelty_score", 0.0)
        )
        synth_row = {
            "id": synth_id,
            "text": transformed_text,
            "department_code": specialty,
            "provenance": json.dumps(provenance),
            "passed_diversity": passed,
        }
        if not dry_run:
            synthetic_records.append(synth_row)
        div_row = {
            "synthetic_id": synth_id,
            "parent_id": parent_id,
            "lexical_diversity": scores["lexical_diversity"],
            "edit_distance": scores["edit_distance"],
            "edit_distance_ratio": edit_ratio,
            "token_overlap": scores["token_overlap"],
            "novelty_score": scores["novelty_score"],
            "passed": passed,
        }
        diversity_rows.append(div_row)

    if not dry_run:
        pd.DataFrame(synthetic_records).to_csv(SYNTHETIC_PATH, index=False)
        enriched_df = pd.concat([orig_df, pd.DataFrame(synthetic_records)], ignore_index=True)
        enriched_df.to_csv(ENRICHED_PATH, index=False)
        pd.DataFrame(diversity_rows).to_csv(DIVERSITY_REPORT, index=False)

        # Validators
        ClinicalSafetyValidator(orig_df.to_dict(orient="records"), synthetic_records).validate()
        DuplicateValidator(ORIG_PATH, SYNTHETIC_PATH).validate()

        stats = {
            "seed": SEED,
            "total_original": len(orig_df),
            "total_synthetic": len(synthetic_records),
            "passed_diversity": sum(r["passed"] for r in diversity_rows),
            "failed_diversity": sum(not r["passed"] for r in diversity_rows),
        }
        with open(GEN_STATS, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        manifest = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config_path": str(CONFIG_PATH),
            "plugins": [p.name for p in plugins],
            "seed": SEED,
            "files": {
                "synthetic_samples": str(SYNTHETIC_PATH),
                "enriched_dataset": str(ENRICHED_PATH),
                "diversity_report": str(DIVERSITY_REPORT),
                "clinical_report": str(CLINICAL_REPORT),
                "duplicate_report": str(DUPLICATE_REPORT),
                "generation_stats": str(GEN_STATS),
            },
        }
        with open(MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        md = (
            "# Dataset Enrichment Report\n\n"
            f"Generated **{stats['total_synthetic']}** synthetic records from **{stats['total_original']}** originals.\n\n"
            f"- Passed diversity: {stats['passed_diversity']}\n"
            f"- Failed diversity: {stats['failed_diversity']}\n\n"
            "Artifacts are stored in `data/processed/enriched/`.\n"
        )
        with open(REPORT_MD, "w", encoding="utf-8") as f:
            f.write(md)
        print("Enrichment completed. See reports in data/processed/enriched/")
    else:
        print(f"Dry run: evaluated {len(diversity_rows)} synthetic candidates (no files written).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dataset Enrichment Engine")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
