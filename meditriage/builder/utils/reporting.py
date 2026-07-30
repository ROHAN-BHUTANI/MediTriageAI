import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def generate_checksum(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_duplicate_report(path: str, dropped_seeds: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Deduplication Report\n")
        f.write("====================\n")
        f.write(f"Total rows dropped: {len(dropped_seeds)}\n\n")
        f.writelines(f"{s}\n" for s in dropped_seeds)


def write_statistics(df: pd.DataFrame, path: str) -> None:
    if len(df) == 0:
        return
    stats = {
        "total_rows": len(df),
        "by_source": df["dataset_source"].value_counts().to_dict(),
        "by_split": df["split"].value_counts().to_dict(),
        "by_severity": df["severity_label"].value_counts().to_dict(),
        "by_department": df["department_code"].value_counts().to_dict(),
        "is_perturbed": df["is_perturbed"].value_counts().to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def write_manifest(
    config, df: pd.DataFrame, adapters: dict, start_time: float, output_dir: Path
) -> None:
    end_time = time.time()

    # Calculate checksums of raw
    raw_checksums = {}
    raw_dir = output_dir.parent / "raw"
    if raw_dir.exists():
        for d in config.active_datasets:
            ds_dir = raw_dir / d
            if ds_dir.exists():
                for f in ds_dir.glob("*"):
                    if f.is_file():
                        raw_checksums[f.name] = generate_checksum(str(f))

    manifest = {
        "builder_version": "2.0.0",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": "3.10+",
        "random_seed": config.random_seed,
        "config_hash": config.get_hash(),
        "build_duration_seconds": end_time - start_time,
        "adapter_versions": {k: v.version for k, v in adapters.items()},
        "raw_checksums": raw_checksums,
        "total_rows_produced": len(df),
    }

    with open(
        output_dir / "processed" / "build_manifest.json", "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)
