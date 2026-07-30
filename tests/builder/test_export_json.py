import json
import tempfile
from pathlib import Path

import numpy as np

from meditriage.builder.config import Config
from meditriage.builder.orchestrator import Builder


def test_export_json_int64_serialization(monkeypatch):
    """
    Regression test for numpy.int64 serialization failure in Stage 7 export.
    We mock value_counts to return a dict with np.int64 values to ensure
    NumpyEncoder is used during the final json.dump.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        config = Config({"splits": {"train": 0.8}, "active_datasets": []})
        builder = Builder(config, tmp_path)

        stg6_dir = builder._create_stage_dir("06_split")
        builder.processed_dir.mkdir(parents=True, exist_ok=True)

        # We need to simulate the execution of Stage 7 where the issue occurs
        import pandas as pd

        df = pd.DataFrame(
            {
                "id": ["1", "2"],
                "split": ["train", "train"],
                "dataset_source": ["src1", "src2"],
                "language": ["en", "en"],
                "raw_text": ["a", "b"],
                "department": ["ED", "ORTHO"],
                "triage_level": ["1", "2"],
            }
        )
        df.to_parquet(stg6_dir / "chunk1.parquet")

        # Monkeypatch pd.Series.value_counts to force returning np.int64
        original_value_counts = pd.Series.value_counts

        def mock_value_counts(self, *args, **kwargs):
            res = original_value_counts(self, *args, **kwargs)
            # Force conversion to a Series with dtype int64 (numpy)
            return pd.Series([np.int64(v) for v in res], index=res.index)

        monkeypatch.setattr(pd.Series, "value_counts", mock_value_counts)

        # We need a way to run ONLY stage 7, or run_pipeline if it was modular.
        # Since it's all in build(), we'll run build() but we'll mock stages 1-6.
        # Actually, if active_datasets is empty, stages 1-5 will be basically no-ops
        # except stage 4 deduplication might look at stage 3.
        # Let's just create the necessary dummy files to trick the builder into running fast.

        # Create empty stage3 files so stage 4 doesn't crash on missing columns
        stg3_dir = builder._create_stage_dir("03_validate")
        df.to_parquet(stg3_dir / "chunk1.parquet")

        # Run builder
        # It will run stage 4, 5, 6, 7.
        # We have to provide the correct columns in stage 3 for stage 4.
        builder.build()

        # If the bug is not fixed, builder.build() will raise TypeError during json.dump
        assert (builder.processed_dir / "dataset_statistics.json").exists()

        with open(builder.processed_dir / "dataset_statistics.json", "r") as f:
            stats = json.load(f)

        assert stats["total_rows"] == 2
        assert stats["splits"]["train"] == 2
