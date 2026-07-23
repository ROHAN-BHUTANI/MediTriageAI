"""
Execution Pipeline for the REF Metrics Engine.

Orchestrates the 7-stage global execution across all registered metric providers.
Collection -> Provider Dispatch -> Metric Computation -> Validation -> Aggregation -> Serialization -> Reporting
"""

from typing import Any
import logging
from collections import OrderedDict

from ref.metrics.types import MetricGroup, MetricReport, MetricCollection
from ref.metrics.registry import MetricRegistry

logger = logging.getLogger(__name__)

class MetricPipeline:
    """
    Executes the metric aggregation pipeline strictly mapping raw tensors
    into final validated, serialized MetricReports.
    """

    def __init__(self, registry: MetricRegistry, experiment_id: str):
        self.registry = registry
        self.experiment_id = experiment_id
        
        # State tracking through pipeline
        self.raw_inputs: dict[str, Any] = {}
        self.dispatched_collections: dict[str, MetricCollection] = OrderedDict()
        self.aggregated_groups: dict[str, MetricGroup] = OrderedDict()
        self.final_report: MetricReport | None = None

    def execute(self, raw_inputs: dict[str, Any]) -> MetricReport:
        """Execute the strict 7-stage metrics pipeline."""
        
        # Stage 1: Collection (Store inputs at pipeline level)
        self.collection_stage(raw_inputs)
        
        # Stages 2-6: Provider Dispatch, Compute, Validation, Serialization, Aggregation
        self.dispatch_and_aggregate_stage()
        
        # Stage 7: Reporting
        return self.reporting_stage()

    def collection_stage(self, raw_inputs: dict[str, Any]) -> None:
        """Stage 1: Ingest universal input payload."""
        logger.debug("Metrics Pipeline Stage 1: Collection")
        self.raw_inputs = raw_inputs

    def dispatch_and_aggregate_stage(self) -> None:
        """
        Stages 2-6: Dispatch to providers, execute lifecycle (Compute->Validate->Serialize), 
        and Aggregate into MetricGroups.
        """
        logger.debug("Metrics Pipeline Stages 2-6: Dispatch & Aggregate")
        
        groups = self.registry.get_all_groups()
        for group_name in groups:
            providers = self.registry.get_providers_for_group(group_name)
            
            group_collections = OrderedDict()
            for provider in providers:
                # Dispatch initiates the provider's collect->compute->validate->serialize->report loop
                meta = provider.get_metadata()
                logger.debug(f"Dispatching to provider: {meta.name}")
                
                collection = provider.execute_lifecycle(self.raw_inputs)
                group_collections[meta.name] = collection
                self.dispatched_collections[meta.name] = collection
                
            # Aggregate group
            group = MetricGroup(group_name=group_name, collections=group_collections)
            group.validate()
            self.aggregated_groups[group_name] = group

    def reporting_stage(self) -> MetricReport:
        """Stage 7: Report wrapping all aggregated groups."""
        logger.debug("Metrics Pipeline Stage 7: Reporting")
        self.final_report = MetricReport(
            experiment_id=self.experiment_id,
            groups=self.aggregated_groups
        )
        self.final_report.validate()
        return self.final_report
