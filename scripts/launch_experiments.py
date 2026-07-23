"""
REF Experiment Execution Launcher.
Provides a thin CLI for orchestrating MediTriageAI evaluation campaigns.
"""

import sys
import argparse
import json
import logging
from pathlib import Path

# Fix path to allow importing from root when run from scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ref.registry import ExperimentRegistry
from ref.metrics.registry import MetricRegistry
from ref.visualization.registry import VisualizationRegistry
from ref.provenance.registry import ProvenanceRegistry
from ref.benchmarks.registry import BenchmarkRegistry
from ref.benchmarks.strategies import BaselineBenchmark, AblationBenchmark

from ref.core.experiment_runner import ExperimentRunner
from ref.core.campaign_runner import CampaignRunner

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="REF Experiment Campaign Runner")
    parser.add_argument("--config", type=str, default="campaign_config.json", help="Path to JSON campaign configuration")
    parser.add_argument("--dry-run", action="store_true", help="Build the execution graph but do not run")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 seed on subset of data for validation")
    parser.add_argument("--resume", action="store_true", help="Resume from campaign_state.json if interrupted")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    # Instantiate Registries
    experiment_registry = ExperimentRegistry()
    metric_registry = MetricRegistry()
    viz_registry = VisualizationRegistry()
    
    prov_config = {"enable_fingerprint_git": True, "enable_fingerprint_environment": True}
    prov_registry = ProvenanceRegistry(prov_config)
    
    bench_config = {"benchmark_engine_enabled": True}
    bench_registry = BenchmarkRegistry(bench_config)
    bench_registry.discover_strategies([BaselineBenchmark, AblationBenchmark])

    # Inject Dependencies
    exp_runner = ExperimentRunner(
        experiment_registry=experiment_registry,
        metric_registry=metric_registry,
        viz_registry=viz_registry,
        prov_registry=prov_registry,
        benchmark_registry=bench_registry
    )

    campaign_runner = CampaignRunner(experiment_runner=exp_runner)

    # Execute Campaign
    try:
        campaign_runner.execute_campaign(
            config=config,
            is_smoke_test=args.smoke_test,
            is_dry_run=args.dry_run,
            resume=args.resume
        )
    except Exception as e:
        logger.error(f"Campaign failed fundamentally: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
