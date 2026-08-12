import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd

from .adapters.chatdoctor_healthcaremagic import ChatDoctorHealthcareMagicAdapter
from .adapters.chatdoctor_icliniq import ChatDoctorIcliniqAdapter
from .adapters.fedmml_ed_triage import FedmmlEdTriageAdapter
from .adapters.kaggle_medical_triage import KaggleMedicalTriageAdapter
from .adapters.l3cube_code_mixed import L3CubeCodeMixedAdapter
from .adapters.meddialog_en import MeddialogEnAdapter
from .adapters.medical_meadow_medqa import MedicalMeadowMedqaAdapter
from .adapters.medqa_usmle import MedqaUsmleAdapter
from .adapters.mtsamples import MTSamplesAdapter
from .adapters.neiss import NeissAdapter
from .adapters.nhamcs_ed import NhamcsEdAdapter
from .adapters.pmc_patients import PMCPatientsAdapter
from .adapters.symptom2disease import Symptom2DiseaseAdapter
from .config import Config

ADAPTER_REGISTRY = {
    "mtsamples": MTSamplesAdapter,
    "pmc_patients": PMCPatientsAdapter,
    "medqa_usmle": MedqaUsmleAdapter,
    "medical_meadow_medqa": MedicalMeadowMedqaAdapter,
    "symptom2disease": Symptom2DiseaseAdapter,
    "chatdoctor_healthcaremagic": ChatDoctorHealthcareMagicAdapter,
    "chatdoctor_icliniq": ChatDoctorIcliniqAdapter,
    "neiss": NeissAdapter,
    "nhamcs_ed": NhamcsEdAdapter,
    "fedmml_ed_triage": FedmmlEdTriageAdapter,
    "kaggle_medical_triage": KaggleMedicalTriageAdapter,
    "l3cube_code_mixed": L3CubeCodeMixedAdapter,
    "meddialog_en": MeddialogEnAdapter,
}


class Builder:
    def __init__(self, config: Config, base_dir: Path):
        self.config = config
        self.base_dir = base_dir
        self.raw_dir = base_dir / "datasets" / "raw"
        self.out_dir = base_dir / "meditriage" / "data"
        self.processed_dir = self.out_dir / "processed"
        self.build_dir = self.out_dir / "build_temp"

    def _create_stage_dir(self, name: str) -> Path:
        d = self.build_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def build(self, force: bool = False) -> None:
        if (self.processed_dir / "dataset.csv").exists() and not force:
            raise FileExistsError("Dataset exists. Use --force to overwrite.")

        if force:
            shutil.rmtree(self.build_dir, ignore_errors=True)

        start_time = time.time()
        print("Starting streaming dataset build...")

        stg1_dir = self._create_stage_dir("01_ingest")
        stg2_dir = self._create_stage_dir("02_normalize")
        stg3_dir = self._create_stage_dir("03_validate")
        stg4_dir = self._create_stage_dir("04_deduplicate")
        stg5_dir = self._create_stage_dir("05_augment")
        stg6_dir = self._create_stage_dir("06_split")

        adapters_used = {}

        print("--- Stage 1: Ingest ---")
        chunk_idx = 0
        total_ingested = 0

        for source in self.config.active_datasets:
            if source in ADAPTER_REGISTRY:
                adapter = ADAPTER_REGISTRY[source]()
                adapters_used[source] = adapter.version
                print(f"Ingesting {source}...")

                source_row_idx = 0
                try:
                    for df_chunk in adapter.ingest(str(self.raw_dir / source)):
                        if len(df_chunk) == 0:
                            continue

                        df_chunk["id"] = [
                            f"{source}::{i:08d}"
                            for i in range(source_row_idx, source_row_idx + len(df_chunk))
                        ]
                        source_row_idx += len(df_chunk)

                        out_path = stg1_dir / f"{source}_chunk_{chunk_idx:04d}.parquet"
                        df_chunk.to_parquet(out_path, index=False)
                        chunk_idx += 1
                        total_ingested += len(df_chunk)
                except Exception as e:
                    print(f"Error ingesting {source}: {e}")
            else:
                print(f"Warning: Adapter for {source} not found.")

        print(f"Ingested {total_ingested} total records into {chunk_idx} shards.")

        print("--- Stage 2: Normalize ---")
        for p in stg1_dir.glob("*.parquet"):
            df = pd.read_parquet(p)
            df.to_parquet(stg2_dir / p.name, index=False)

        print("--- Stage 3: Validate ---")
        for p in stg2_dir.glob("*.parquet"):
            df = pd.read_parquet(p)
            for col in [
                "dataset_source",
                "raw_text",
                "triage_level",
                "department",
                "language",
            ]:
                if col not in df.columns:
                    df[col] = None
            df.to_parquet(stg3_dir / p.name, index=False)

        print("--- Stage 4: Deduplicate (Streaming Hash) ---")
        seen_texts = {}
        duplicates_to_drop = set()

        priority_order = getattr(self.config, "deduplication", {}).get(
            "priority_order", []
        )
        priority_map = {src: i for i, src in enumerate(priority_order)}

        for p in stg3_dir.glob("*.parquet"):
            df = pd.read_parquet(p, columns=["id", "raw_text", "dataset_source"])
            for _, row in df.iterrows():
                txt = row["raw_text"]
                src = row["dataset_source"]
                rid = row["id"]

                if pd.isna(txt):
                    continue

                # Use a lightweight hash for memory efficiency if text is huge, but dict string intern is fine for now
                if txt in seen_texts:
                    exist_src, exist_id = seen_texts[txt]
                    p_new = priority_map.get(src, 999)
                    p_old = priority_map.get(exist_src, 999)

                    if p_new < p_old:
                        duplicates_to_drop.add(exist_id)
                        seen_texts[txt] = (src, rid)
                    else:
                        duplicates_to_drop.add(rid)
                else:
                    seen_texts[txt] = (src, rid)

        print(f"Found {len(duplicates_to_drop)} duplicates globally.")

        total_after_dedup = 0
        for p in stg3_dir.glob("*.parquet"):
            df = pd.read_parquet(p)
            df = df[~df["id"].isin(duplicates_to_drop)]
            if len(df) > 0:
                df.to_parquet(stg4_dir / p.name, index=False)
                total_after_dedup += len(df)
        print(f"Total records after dedup: {total_after_dedup}")

        print("--- Stage 5: Augment ---")
        for p in stg4_dir.glob("*.parquet"):
            df = pd.read_parquet(p)
            df.to_parquet(stg5_dir / p.name, index=False)

        print("--- Stage 6: Split ---")

        def get_split(rid):
            h = int(hashlib.md5(rid.encode()).hexdigest(), 16)
            r = (h % 100) / 100.0

            # Safely handle self.config.splits which might be a dictionary or a custom object
            splits_dict = (
                self.config.splits
                if isinstance(self.config.splits, dict)
                else getattr(self.config.splits, "__dict__", {})
            )
            train_pct = (
                splits_dict.get("train", 0.8) if isinstance(splits_dict, dict) else 0.8
            )
            val_pct = (
                splits_dict.get("val", 0.1) if isinstance(splits_dict, dict) else 0.1
            )

            if r < train_pct:
                return "train"
            elif r < train_pct + val_pct:
                return "val"
            return "test"

        for p in stg5_dir.glob("*.parquet"):
            df = pd.read_parquet(p)
            df["split"] = df["id"].apply(get_split)
            df.to_parquet(stg6_dir / p.name, index=False)

        print("--- Stage 7: Export ---")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        out_csv = self.processed_dir / "dataset.csv"
        out_pq = self.processed_dir / "dataset.parquet"

        # Remove existing export files to prevent stale CSV appending or mixed outputs
        if out_csv.exists():
            out_csv.unlink()
        if out_pq.exists():
            out_pq.unlink()

        # Write CSV and Parquet iteratively to prevent OOM
        import pyarrow as pa
        import pyarrow.parquet as pq

        first = True
        pq_writer = None
        total_rows = 0
        splits_count = {}
        sources_count = {}

        canonical_columns = [
            "id",
            "split",
            "dataset_source",
            "language",
            "raw_text",
            "department",
            "triage_level",
        ]

        # 1. Define explicit pyarrow schema
        export_schema = pa.schema(
            [
                ("id", pa.string()),
                ("split", pa.string()),
                ("dataset_source", pa.string()),
                ("language", pa.string()),
                ("raw_text", pa.string()),
                ("department", pa.string()),
                ("triage_level", pa.string()),
            ]
        )

        try:
            for p in stg6_dir.glob("*.parquet"):
                df_chunk = pd.read_parquet(p)
                if len(df_chunk) == 0:
                    continue

                # 3. Before conversion ensure canonical columns
                for col in canonical_columns:
                    if col not in df_chunk.columns:
                        df_chunk[col] = None
                df_chunk = df_chunk[canonical_columns].copy()

                for col in canonical_columns:
                    df_chunk[col] = (
                        df_chunk[col]
                        .astype("string")
                        .where(df_chunk[col].notna(), None)
                    )

                total_rows += len(df_chunk)

                # Update stats
                for split_val, count in df_chunk["split"].value_counts().items():
                    splits_count[split_val] = splits_count.get(split_val, 0) + count
                for src_val, count in df_chunk["dataset_source"].value_counts().items():
                    sources_count[src_val] = sources_count.get(src_val, 0) + count

                # Write CSV
                df_chunk.to_csv(out_csv, mode="a", header=first, index=False)

                # Write Parquet with explicit schema
                table = pa.Table.from_pandas(
                    df_chunk, schema=export_schema, preserve_index=False
                )
                if first:
                    # Initialize ParquetWriter using export_schema
                    pq_writer = pq.ParquetWriter(out_pq, schema=export_schema)
                    first = False
                pq_writer.write_table(table)

        finally:
            if pq_writer:
                pq_writer.close()

        if not first:
            import numpy as np

            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super().default(obj)

            stats = {
                "total_rows": total_rows,
                "splits": splits_count,
                "sources": sources_count,
            }
            with open(self.processed_dir / "dataset_statistics.json", "w") as f:
                json.dump(stats, f, indent=2, cls=NumpyEncoder)

            manifest = {
                "adapters": adapters_used,
                "timestamp": time.time(),
                "duration": time.time() - start_time,
            }
            with open(self.processed_dir / "build_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2, cls=NumpyEncoder)

            with open(self.processed_dir / "duplicate_report.txt", "w") as f:
                f.write(f"Dropped {len(duplicates_to_drop)} global exact matches.")

            # Create a simple coverage report for UI
            with open(self.processed_dir / "coverage_report.txt", "w") as f:
                f.write(
                    f"Coverage Report:\nTotal Ingested: {total_ingested}\nTotal Output: {total_rows}\n"
                )
                f.write("Adapters Used: " + ", ".join(adapters_used.keys()))

        print(f"Build complete in {time.time() - start_time:.2f}s!")
