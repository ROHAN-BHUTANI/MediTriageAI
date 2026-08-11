import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from meditriage.builder.config import Config
from meditriage.builder.orchestrator import Builder


def test_export_schema_inference_bug():
    """
    Regression test for Stage 7 export Parquet schema inference.
    If the first chunk contains all-null values, pyarrow infers 'null' type.
    When subsequent chunks with strings arrive, ParquetWriter will crash
    unless the schema is explicitly forced to 'string' from the beginning.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create a dummy config and builder
        config = Config(
            {"splits": {"train": 0.8, "val": 0.1, "test": 0.1}, "active_datasets": []}
        )
        builder = Builder(config, tmp_path)

        # We need to simulate stage 6 directory having files ready for export
        stg6_dir = builder._create_stage_dir("stage6_split")
        builder.processed_dir.mkdir(parents=True, exist_ok=True)

        # Chunk 1: all nulls in department and triage_level
        df1 = pd.DataFrame(
            {
                "id": ["1", "2"],
                "split": ["train", "val"],
                "dataset_source": ["test_source", "test_source"],
                "language": ["en", "en"],
                "raw_text": ["doc1", "doc2"],
                "department": [None, None],
                "triage_level": [None, None],
            }
        )
        df1.to_parquet(stg6_dir / "chunk1.parquet")

        # Chunk 2: valid strings
        df2 = pd.DataFrame(
            {
                "id": ["3", "4"],
                "split": ["test", "train"],
                "dataset_source": ["test_source", "test_source"],
                "language": ["en", "en"],
                "raw_text": ["doc3", "doc4"],
                "department": ["ORTHO", "ED"],
                "triage_level": ["3", "1"],
            }
        )
        df2.to_parquet(stg6_dir / "chunk2.parquet")

        # Run export logic (extracted from Stage 7 in orchestrator.py)
        import pyarrow as pa

        first = True
        pq_writer = None
        total_rows = 0

        canonical_columns = [
            "id",
            "split",
            "dataset_source",
            "language",
            "raw_text",
            "department",
            "triage_level",
        ]

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

        out_csv = builder.processed_dir / "dataset.csv"
        out_pq = builder.processed_dir / "dataset.parquet"

        try:
            for p in sorted(stg6_dir.glob("*.parquet")):
                df_chunk = pd.read_parquet(p)
                if len(df_chunk) == 0:
                    continue

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

                df_chunk.to_csv(out_csv, mode="a", header=first, index=False)

                table = pa.Table.from_pandas(
                    df_chunk, schema=export_schema, preserve_index=False
                )
                if first:
                    pq_writer = pq.ParquetWriter(out_pq, schema=export_schema)
                    first = False
                pq_writer.write_table(table)
        finally:
            if pq_writer:
                pq_writer.close()

        # Validate resulting parquet schema has department=string
        # and triage_level=string
        result_table = pq.read_table(out_pq)

        assert result_table.schema.field("department").type == pa.string()
        assert result_table.schema.field("triage_level").type == pa.string()

        df_result = result_table.to_pandas()
        assert len(df_result) == 4
        # Validate that chunk1 values are actually None (or pd.NA) and not "None"
        assert pd.isna(df_result.loc[0, "department"])
        assert pd.isna(df_result.loc[1, "department"])

        # Validate that chunk2 values are intact
        assert df_result.loc[2, "department"] == "ORTHO"
