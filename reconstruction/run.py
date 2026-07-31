"""Dataset Reconstruction Engine – CLI Runner.

Usage:
    python -m reconstruction.run --stages 1-5
    python -m reconstruction.run --stages 1,2,3
    python -m reconstruction.run --stages 3 --target-class-size 10000
    python -m reconstruction.run --config reconstruction_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from reconstruction.config import ReconstructionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reconstruction")


def parse_stages(stage_spec: str) -> list[int]:
    """Parse stage specification like '1-5' or '1,3,5' into a list of ints."""
    stages = []
    for part in stage_spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            stages.extend(range(int(lo), int(hi) + 1))
        else:
            stages.append(int(part))
    return sorted(set(stages))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dataset Reconstruction Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stages", type=str, default="1-5",
        help="Stages to run, e.g. '1-5' or '1,3,5' (default: 1-5)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file (overrides defaults)",
    )
    parser.add_argument(
        "--target-class-size", type=int, default=None,
        help="Override target class size",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Override dataset path",
    )
    parser.add_argument(
        "--output-directory", type=str, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--llm-provider", type=str, default=None,
        choices=["gemini", "openai", "offline"],
        help="Override LLM provider (gemini | openai | offline)",
    )
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="Override LLM model name",
    )

    args = parser.parse_args(argv)

    # Load config
    if args.config and Path(args.config).exists():
        cfg = ReconstructionConfig.load(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        cfg = ReconstructionConfig()

    # Apply CLI overrides
    if args.target_class_size is not None:
        cfg.target_class_size = args.target_class_size
    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path
    if args.output_directory is not None:
        cfg.output_directory = args.output_directory
    if args.seed is not None:
        cfg.random_seed = args.seed
    if args.llm_provider is not None:
        cfg.llm_provider = args.llm_provider
    if args.llm_model is not None:
        cfg.llm_model = args.llm_model

    stages = parse_stages(args.stages)
    logger.info("Running stages: %s", stages)
    logger.info("Target class size: %d", cfg.target_class_size)
    logger.info("Output directory: %s", cfg.output_directory)

    # Save effective config
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "reconstruction_config.json")

    df = None
    t0 = time.time()

    # Stage 1
    if 1 in stages:
        from reconstruction.stage1_load import run as run_s1

        logger.info("=" * 60)
        logger.info("STAGE 1: Load Dataset")
        logger.info("=" * 60)
        df = run_s1(cfg)

    # Stage 2
    if 2 in stages:
        from reconstruction.stage2_clean import run as run_s2

        if df is None:
            from reconstruction.stage1_load import run as run_s1
            df = run_s1(cfg)

        logger.info("=" * 60)
        logger.info("STAGE 2: Clean Dataset")
        logger.info("=" * 60)
        df = run_s2(df, cfg)

    # Stage 3
    if 3 in stages:
        from reconstruction.stage3_cluster import run as run_s3

        if df is None:
            # Try loading from stage2 output or run stages 1-2
            s2_path = out_dir / "stage3_clusters.parquet"
            if s2_path.exists():
                import pandas as pd
                df = pd.read_parquet(s2_path)
            else:
                from reconstruction.stage1_load import run as run_s1
                from reconstruction.stage2_clean import run as run_s2
                df = run_s1(cfg)
                df = run_s2(df, cfg)

        logger.info("=" * 60)
        logger.info("STAGE 3: Clinical Phenotype Clustering")
        logger.info("=" * 60)
        df = run_s3(df, cfg)

    # Stage 4
    if 4 in stages:
        from reconstruction.stage4_diversity import run as run_s4

        if df is None:
            s3_path = out_dir / "stage3_clusters.parquet"
            if s3_path.exists():
                import pandas as pd
                df = pd.read_parquet(s3_path)
            else:
                raise RuntimeError("Stage 4 requires Stage 3 output. Run stages 1-3 first.")

        logger.info("=" * 60)
        logger.info("STAGE 4: Diversity Scoring")
        logger.info("=" * 60)
        df = run_s4(df, cfg)

    # Stage 5
    if 5 in stages:
        from reconstruction.stage5_undersample import run as run_s5

        if df is None:
            s4_path = out_dir / "stage4_diversity_scores.parquet"
            if s4_path.exists():
                import pandas as pd
                df = pd.read_parquet(s4_path)
            else:
                raise RuntimeError("Stage 5 requires Stage 4 output. Run stages 1-4 first.")

        logger.info("=" * 60)
        logger.info("STAGE 5: Diversity-Maximization Undersampling")
        logger.info("=" * 60)
        df = run_s5(df, cfg)

    # Stage 6
    if 6 in stages:
        from reconstruction.stage6_augment import run as run_s6

        if df is None:
            s5_path = out_dir / "stage5_majority_selected.parquet"
            if s5_path.exists():
                import pandas as pd
                df = pd.read_parquet(s5_path)
            else:
                raise RuntimeError("Stage 6 requires Stage 5 output. Run stages 1-5 first.")

        logger.info("=" * 60)
        logger.info("STAGE 6: Multi-Axis Augmentation")
        logger.info("=" * 60)
        df = run_s6(df, cfg)

    # Stage 7
    if 7 in stages:
        from reconstruction.stage7_generate import run as run_s7

        if df is None:
            s6_path = out_dir / "stage6_augmented.parquet"
            if s6_path.exists():
                import pandas as pd
                df = pd.read_parquet(s6_path)
            else:
                raise RuntimeError("Stage 7 requires Stage 6 output. Run stages 1-6 first.")

        logger.info("=" * 60)
        logger.info("STAGE 7: LLM Synthetic Generation")
        logger.info("=" * 60)
        df = run_s7(df, cfg)

    # Stage 8
    if 8 in stages:
        from reconstruction.stage8_merge import run as run_s8

        if df is None:
            s7_path = out_dir / "stage7_synthetic.parquet"
            if s7_path.exists():
                import pandas as pd
                df = pd.read_parquet(s7_path)
            else:
                raise RuntimeError("Stage 8 requires Stage 7 output. Run stages 1-7 first.")

        logger.info("=" * 60)
        logger.info("STAGE 8: Merge Engine")
        logger.info("=" * 60)
        df = run_s8(df, cfg)

    # Stage 9
    if 9 in stages:
        from reconstruction.stage9_shuffle import run as run_s9

        if df is None:
            s8_path = out_dir / "stage8_merged.parquet"
            if s8_path.exists():
                import pandas as pd
                df = pd.read_parquet(s8_path)
            else:
                raise RuntimeError("Stage 9 requires Stage 8 output. Run stages 1-8 first.")

        logger.info("=" * 60)
        logger.info("STAGE 9: Global Shuffle")
        logger.info("=" * 60)
        df = run_s9(df, cfg)

    # Stage 10
    if 10 in stages:
        from reconstruction.stage10_validate import run as run_s10

        if df is None:
            s9_path = out_dir / "stage9_shuffled.parquet"
            if s9_path.exists():
                import pandas as pd
                df = pd.read_parquet(s9_path)
            else:
                raise RuntimeError("Stage 10 requires Stage 9 output. Run stages 1-9 first.")

        logger.info("=" * 60)
        logger.info("STAGE 10: Validation Engine")
        logger.info("=" * 60)
        df = run_s10(df, cfg)

    # Diversity Report
    if max(stages) >= 10:
        from reconstruction.report import generate_diversity_report

        logger.info("=" * 60)
        logger.info("GENERATING DIVERSITY REPORT")
        logger.info("=" * 60)
        if df is not None:
            generate_diversity_report(df, cfg)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", elapsed)
    if df is not None:
        logger.info("Final dataset: %d rows", len(df))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

