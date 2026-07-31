"""CLI Runner for Multilingual Dataset Expansion Engine.

Usage:
    python -m meditriage.multilingual.cli --input meditriage/data/processed/dataset.parquet --output results/multilingual/dataset_multilingual.parquet
    python -m meditriage.multilingual.cli --provider gemini --model-name gemini-2.0-flash
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from meditriage.multilingual.config import MultilingualConfig
from meditriage.multilingual.report import generate_multilingual_reports
from meditriage.multilingual.translator import MultilingualTranslator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("multilingual_cli")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual Dataset Expansion Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input parquet dataset path (default: meditriage/data/processed/dataset.parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet dataset path (default: results/multilingual/dataset_multilingual.parquet)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="offline",
        choices=["offline", "gemini", "openai"],
        help="Multilingual provider (default: offline)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en,hi,hi-Latn,hi-en,en-hi",
        help="Comma-separated list of target languages",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom JSON config file",
    )

    args = parser.parse_args(argv)

    if args.config and Path(args.config).exists():
        cfg = MultilingualConfig.load(args.config)
    else:
        cfg = MultilingualConfig()

    if args.input:
        cfg.dataset_path = args.input
    if args.provider:
        cfg.provider = args.provider
    if args.languages:
        cfg.target_languages = [l.strip() for l in args.languages.split(",")]
    if args.num_workers:
        cfg.num_workers = args.num_workers

    in_path = Path(cfg.dataset_path)
    if not in_path.exists():
        logger.error("Input dataset not found: %s", in_path)
        sys.exit(1)

    logger.info("Loading dataset from %s", in_path)
    df = pd.read_parquet(in_path)

    translator = MultilingualTranslator(cfg)
    expanded_df = translator.expand_dataframe(df)

    out_path = Path(
        args.output or (Path(cfg.output_dir) / "dataset_multilingual.parquet")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expanded_df.to_parquet(out_path, index=False)
    logger.info("Saved expanded dataset (%d rows) to %s", len(expanded_df), out_path)

    generate_multilingual_reports(expanded_df, translator.stats, cfg)
    logger.info("Multilingual Expansion Pipeline completed successfully!")


if __name__ == "__main__":
    main()
