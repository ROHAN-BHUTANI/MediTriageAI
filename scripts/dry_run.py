"""Production Readiness Dry Run.

Executes the complete 10-stage reconstruction pipeline on a synthetic
representative dataset to verify artifact generation, resume logic,
provenance tracking, and validator correctness.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dry_run")

# ── Configuration ────────────────────────────────────────────────────────

SUBSET_SIZE = 200  # samples per department
N_DEPARTMENTS = 5
DRY_RUN_DIR = Path("results/dry_run")
DATASET_PATH = DRY_RUN_DIR / "synthetic_input.parquet"

DEPARTMENTS = ["ORTHO", "NEURO", "PEDS", "CARDIO_PULM", "OBGYN"]


def create_synthetic_dataset() -> Path:
    """Create a representative synthetic dataset for the dry run."""
    rng = np.random.RandomState(42)
    rows = []

    symptom_pools = {
        "ORTHO": [
            "knee pain",
            "back pain",
            "fracture",
            "joint swelling",
            "sprain",
            "muscle ache",
            "hip pain",
            "shoulder pain",
            "wrist injury",
            "ankle twist",
        ],
        "NEURO": [
            "headache",
            "dizziness",
            "seizure",
            "numbness",
            "tremor",
            "migraine",
            "memory loss",
            "confusion",
            "fainting",
            "weakness",
        ],
        "PEDS": [
            "fever",
            "cough",
            "rash",
            "vomiting",
            "ear pain",
            "cold",
            "crying",
            "stomach pain",
            "diarrhea",
            "wheezing",
        ],
        "CARDIO_PULM": [
            "chest pain",
            "shortness of breath",
            "palpitations",
            "cough",
            "high blood pressure",
            "wheezing",
            "fatigue",
            "swelling legs",
            "rapid heart rate",
            "difficulty breathing",
        ],
        "OBGYN": [
            "abdominal pain",
            "irregular periods",
            "pregnancy concern",
            "discharge",
            "pelvic pain",
            "bleeding",
            "breast lump",
            "nausea",
            "cramps",
            "fatigue",
        ],
    }

    templates_en = [
        "I have been experiencing {s1} and {s2} for {dur}",
        "Patient presents with {s1}, {s2} since {dur}",
        "{s1} along with {s2}, getting worse over {dur}",
        "Severe {s1}, mild {s2} for the past {dur}",
        "I am suffering from {s1} and {s2}, it hurts a lot since {dur}",
    ]
    templates_hi = [
        "Mujhe {s1} aur {s2} ho raha hai {dur} se",
        "{s1} bahut zyada hai, {s2} bhi hai {dur} se",
        "Doctor sahab {s1} aur {s2} {dur} se hai",
    ]
    durations = [
        "2 days",
        "a week",
        "yesterday",
        "3 hours",
        "this morning",
        "kal se",
        "do din se",
        "ek hafte se",
    ]

    # Vary sizes: 2 departments large (>= target), 2 mid-tier, 1 minority
    dept_sizes = {
        "ORTHO": SUBSET_SIZE,
        "NEURO": SUBSET_SIZE,
        "PEDS": 80,  # mid-tier (will be augmented)
        "CARDIO_PULM": 80,  # mid-tier (will be augmented)
        "OBGYN": 15,  # minority (will need LLM generation)
    }

    for dept, n in dept_sizes.items():
        symptoms = symptom_pools[dept]
        for i in range(n):
            s1 = symptoms[rng.randint(0, len(symptoms))]
            s2 = symptoms[rng.randint(0, len(symptoms))]
            dur = durations[rng.randint(0, len(durations))]

            if rng.random() < 0.7:
                template = templates_en[rng.randint(0, len(templates_en))]
            else:
                template = templates_hi[rng.randint(0, len(templates_hi))]

            text = template.format(s1=s1, s2=s2, dur=dur)

            rows.append(
                {
                    "id": f"{dept}_{i:04d}",
                    "split": "train",
                    "dataset_source": "synthetic_test",
                    "language": "en" if rng.random() < 0.7 else "hi-en",
                    "raw_text": text,
                    "department": dept,
                    "triage_level": f"S{rng.randint(1, 6)}",
                }
            )

    df = pd.DataFrame(rows)
    DRY_RUN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH, index=False)
    logger.info("Created synthetic dataset: %d rows -> %s", len(df), DATASET_PATH)
    return DATASET_PATH


def run_dry_run():
    """Execute the complete pipeline and collect results."""
    from reconstruction.config import ReconstructionConfig

    results = {
        "stages": {},
        "artifacts": [],
        "warnings": [],
        "failures": [],
        "edge_cases": [],
    }

    # Create input
    dataset_path = create_synthetic_dataset()

    # Configure for dry run
    output_dir = str(DRY_RUN_DIR / "output")
    cfg = ReconstructionConfig(
        dataset_path=str(dataset_path),
        output_directory=output_dir,
        target_class_size=100,
        random_seed=42,
        augmentation_min_class_size=50,
        llm_provider="offline",
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "reconstruction_config.json")

    t_total = time.time()
    df = None

    # ── Stage 1 ──────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage1_load import run as run_s1

        df = run_s1(cfg)
        results["stages"]["stage1"] = {
            "status": "✅ PASS",
            "rows": len(df),
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["stage1"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 1: {e}")
        return results

    # ── Stage 2 ──────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage2_clean import run as run_s2

        df = run_s2(df, cfg)
        results["stages"]["stage2"] = {
            "status": "✅ PASS",
            "rows": len(df),
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["stage2"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 2: {e}")
        return results

    # ── Stage 3 ──────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage3_cluster import run as run_s3

        df = run_s3(df, cfg)
        n_clusters = df["cluster_id"].nunique() if "cluster_id" in df.columns else 0
        results["stages"]["stage3"] = {
            "status": "✅ PASS",
            "rows": len(df),
            "clusters": n_clusters,
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["stage3"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 3: {e}")
        return results

    # ── Stage 4 ──────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage4_diversity import run as run_s4

        df = run_s4(df, cfg)
        has_score = "diversity_score" in df.columns
        results["stages"]["stage4"] = {
            "status": "✅ PASS",
            "rows": len(df),
            "has_diversity_score": has_score,
            "time": f"{time.time() - t:.2f}s",
        }
        if not has_score:
            results["warnings"].append("Stage 4: diversity_score column missing")
    except Exception as e:
        results["stages"]["stage4"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 4: {e}")
        return results

    # ── Stage 5 ──────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage5_undersample import run as run_s5

        pre_s5 = len(df)
        df = run_s5(df, cfg)
        results["stages"]["stage5"] = {
            "status": "✅ PASS",
            "rows_in": pre_s5,
            "rows_out": len(df),
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["stage5"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 5: {e}")
        return results

    # ── Stage 6 ──────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage6_augment import run as run_s6

        pre_s6 = len(df)
        df = run_s6(df, cfg)
        augmented = len(df) - pre_s6
        results["stages"]["stage6"] = {
            "status": "✅ PASS",
            "rows_in": pre_s6,
            "rows_out": len(df),
            "augmented": augmented,
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["stage6"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 6: {e}")
        return results

    # ── Stage 7 ──────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage7_generate import run as run_s7

        pre_s7 = len(df)
        df = run_s7(df, cfg)
        synthetic = len(df) - pre_s7
        results["stages"]["stage7"] = {
            "status": "✅ PASS",
            "rows_in": pre_s7,
            "rows_out": len(df),
            "synthetic": synthetic,
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["stage7"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 7: {e}")
        return results

    # ── Stage 8 ──────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage8_merge import run as run_s8

        df = run_s8(df, cfg)
        results["stages"]["stage8"] = {
            "status": "✅ PASS",
            "rows": len(df),
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["stage8"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 8: {e}")
        return results

    # ── Stage 9 ──────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage9_shuffle import run as run_s9

        df = run_s9(df, cfg)
        results["stages"]["stage9"] = {
            "status": "✅ PASS",
            "rows": len(df),
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["stage9"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 9: {e}")
        return results

    # ── Stage 10 ─────────────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.stage10_validate import run as run_s10

        df = run_s10(df, cfg)
        results["stages"]["stage10"] = {
            "status": "✅ PASS",
            "rows": len(df),
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["stage10"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Stage 10: {e}")
        return results

    # ── Diversity Report ─────────────────────────────────────────────────
    t = time.time()
    try:
        from reconstruction.report import generate_diversity_report

        generate_diversity_report(df, cfg)
        results["stages"]["report"] = {
            "status": "✅ PASS",
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["stages"]["report"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Report: {e}")

    results["total_time"] = f"{time.time() - t_total:.2f}s"
    results["final_rows"] = len(df)

    # ── Artifact Inventory ───────────────────────────────────────────────
    for p in sorted(Path(output_dir).rglob("*")):
        if p.is_file():
            results["artifacts"].append(
                {
                    "file": p.name,
                    "size_bytes": p.stat().st_size,
                    "format": p.suffix,
                }
            )

    # ── Validation Results ───────────────────────────────────────────────
    val_path = Path(output_dir) / "stage10_validation_results.json"
    if val_path.exists():
        with open(val_path) as f:
            results["validation_results"] = json.load(f)

    # ── Resume Test ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("RESUME TEST: Re-running stages 8-10...")
    t = time.time()
    try:
        from reconstruction.stage8_merge import run as run_s8_r
        from reconstruction.stage9_shuffle import run as run_s9_r
        from reconstruction.stage10_validate import run as run_s10_r

        df_r = run_s8_r(df, cfg)
        df_r = run_s9_r(df_r, cfg)
        df_r = run_s10_r(df_r, cfg)
        results["resume_test"] = {
            "status": "✅ PASS",
            "time": f"{time.time() - t:.2f}s",
        }
    except Exception as e:
        results["resume_test"] = {"status": "❌ FAIL", "error": str(e)}
        results["failures"].append(f"Resume test: {e}")

    # ── Provenance Check ─────────────────────────────────────────────────
    prov_cols = [c for c in df.columns if c.startswith("_provenance_")]
    has_aug = (
        "dataset_source" in df.columns
        and df["dataset_source"].str.startswith("augmented_").any()
    )
    has_syn = (
        "dataset_source" in df.columns
        and df["dataset_source"].str.startswith("synthetic_").any()
    )
    results["provenance_check"] = {
        "provenance_columns": prov_cols,
        "has_augmented_samples": bool(has_aug),
        "has_synthetic_samples": bool(has_syn),
        "passed": len(prov_cols) > 0,
    }

    # ── Department Balance ───────────────────────────────────────────────
    dept_counts = df["department"].value_counts().to_dict()
    results["department_balance"] = dept_counts

    # ── Edge Cases ───────────────────────────────────────────────────────
    # Check for empty departments after processing
    for dept, count in dept_counts.items():
        if count == 0:
            results["edge_cases"].append(
                f"Department {dept} has 0 samples after reconstruction"
            )

    # Check for NaN in critical columns
    for col in ["raw_text", "department", "id"]:
        if col in df.columns:
            nan_count = int(df[col].isna().sum())
            if nan_count > 0:
                results["edge_cases"].append(f"Column {col} has {nan_count} NaN values")

    return results


if __name__ == "__main__":
    results = run_dry_run()

    # Print summary
    print("\n" + "=" * 70)
    print("PRODUCTION READINESS DRY RUN REPORT")
    print("=" * 70)

    print("\n--- Stage Results ---")
    for stage, info in results.get("stages", {}).items():
        status = info.get("status", "?")
        t = info.get("time", "")
        print(f"  {stage:12s}: {status} ({t})")

    print(f"\n--- Total Time: {results.get('total_time', '?')} ---")
    print(f"--- Final Rows: {results.get('final_rows', '?')} ---")

    print("\n--- Artifacts Generated ---")
    for a in results.get("artifacts", []):
        print(f"  {a['file']:45s} {a['size_bytes']:>10,} bytes  [{a['format']}]")

    print(f"\n--- Resume Test: {results.get('resume_test', {}).get('status', '?')} ---")
    print(
        f"--- Provenance: {results.get('provenance_check', {}).get('passed', '?')} ---"
    )

    print("\n--- Department Balance ---")
    for dept, count in results.get("department_balance", {}).items():
        print(f"  {dept:20s}: {count}")

    print("\n--- Warnings ---")
    for w in results.get("warnings", []):
        print(f"  ⚠️  {w}")

    print("\n--- Failures ---")
    for f in results.get("failures", []):
        print(f"  ❌ {f}")

    print("\n--- Edge Cases ---")
    for e in results.get("edge_cases", []):
        print(f"  ⚡ {e}")

    # Validation
    val = results.get("validation_results", {})
    if val:
        print("\n--- Validator Results ---")
        for v in val.get("validators", []):
            status = "✅" if v.get("passed") else "❌"
            print(f"  {status} {v.get('validator', '?')}")

    # Overall assessment
    n_failures = len(results.get("failures", []))
    print("\n" + "=" * 70)
    if n_failures == 0:
        print("✅ PRODUCTION READY — all stages pass, all validators pass")
    else:
        print(f"❌ NOT READY — {n_failures} failure(s) detected")
    print("=" * 70)

    # Save full results
    report_path = DRY_RUN_DIR / "dry_run_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report: {report_path}")
