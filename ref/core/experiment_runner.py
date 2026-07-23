"""
Experiment Runner for the Research Experiment Framework (REF).

Responsible for executing a single experimental configuration.
Strictly delegates to the REF subsystems (Core, Metrics, Visualization, Provenance, Benchmark).
"""

import logging
from typing import Any
from pathlib import Path

from ref.registry import ExperimentRegistry
from ref.experiments import AblationExperiment
from ref.metrics.pipeline import MetricPipeline
from ref.metrics.registry import MetricRegistry
from ref.visualization.pipeline import VisualizationPipeline
from ref.visualization.registry import VisualizationRegistry
from ref.provenance.pipeline import ProvenancePipeline
from ref.provenance.registry import ProvenanceRegistry
from ref.benchmarks.pipeline import BenchmarkPipeline
from ref.benchmarks.registry import BenchmarkRegistry

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Executes a single experiment configuration and routes output telemetry 
    through all strict REF pipelines.
    """

    def __init__(
        self,
        experiment_registry: ExperimentRegistry,
        metric_registry: MetricRegistry,
        viz_registry: VisualizationRegistry,
        prov_registry: ProvenanceRegistry,
        benchmark_registry: BenchmarkRegistry
    ):
        self.experiment_registry = experiment_registry
        self.metric_registry = metric_registry
        self.viz_registry = viz_registry
        self.prov_registry = prov_registry
        self.benchmark_registry = benchmark_registry

    def run(self, experiment_id: str, config: dict[str, Any], output_dir: Path, is_smoke_test: bool = False) -> None:
        """
        Executes a single experiment by invoking the 10-stage lifecycle,
        and subsequently delegating to telemetry pipelines.
        """
        logger.info(f"Running Experiment: {experiment_id} in {output_dir}")
        
        # 1. Instantiate the Experiment through REF Core
        # For evaluation & ablation runs mapped to our experiments
        modules_enabled = {
            "ccsm": not config.get("ablate_ccsm", False),
            "aces": not config.get("ablate_aces", False),
            "amco": not config.get("ablate_amco", False),
            "dccf": not config.get("ablate_dccf", False),
        }
        
        # Using AblationExperiment as the base driver for campaign logic
        experiment = AblationExperiment(
            registry=self.experiment_registry,
            name=experiment_id,
            hypothesis=f"Testing configuration: {experiment_id}",
            dataset=config.get("dataset_primary", "default"),
            modules_enabled=modules_enabled,
            config_overrides={"smoke_test": is_smoke_test},
            seed=config.get("seed", 42)
        )
        
        # Override workspace to ensure strict output isolation
        experiment.workspace = output_dir
        
        # Execute REF Core Lifecycle
        report = experiment.execute_lifecycle()
        
        # Build context for telemetry pipelines
        context = {
            "experiment_id": experiment_id,
            "report": report.to_dict() if report else {},
            "output_dir": str(output_dir),
            "seed": config.get("seed", 42)
        }
        
        # 2. Invoke Metrics Engine
        logger.info(f"[{experiment_id}] Invoking Metrics Engine")
        metrics_pipeline = MetricPipeline(self.metric_registry, experiment_id=experiment_id)
        metrics_pipeline.execute(context)
        
        # 3. Invoke Visualization Engine
        from ref.visualization.types import VisualizationRequest
        logger.info(f"[{experiment_id}] Invoking Visualization Engine")
        viz_pipeline = VisualizationPipeline(self.viz_registry)
        viz_request = VisualizationRequest(
            experiment_id=experiment_id,
            output_dir=str(output_dir),
            metric_report_dict=report.metrics.to_dict() if report and report.metrics else {},
            config_overrides=config
        )
        viz_pipeline.execute(viz_request)
        
        # 4. Invoke Provenance Pipeline
        logger.info(f"[{experiment_id}] Invoking Provenance Framework")
        prov_pipeline = ProvenancePipeline(self.prov_registry, experiment_id=experiment_id, output_dir=str(output_dir))
        prov_pipeline.execute(context)
        
        # 5. Invoke Benchmark Engine
        logger.info(f"[{experiment_id}] Invoking Benchmark Engine")
        bench_pipeline = BenchmarkPipeline(self.benchmark_registry, report_id=f"bench_{experiment_id}", output_dir=str(output_dir))
        bench_pipeline.execute(context)
        
        logger.info(f"Experiment {experiment_id} completed. Artifacts registered in {output_dir}")
