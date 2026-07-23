"""
Execution Pipeline for the REF Visualization Engine.

Orchestrates the global execution across all registered visualization providers.
MetricReport -> Dispatch -> Render -> Validate -> Export -> Collect -> Report.
"""

from typing import Any
import logging
from collections import OrderedDict

from ref.visualization.types import (
    VisualizationRequest,
    VisualizationCollection,
    VisualizationReport
)
from ref.visualization.registry import VisualizationRegistry

logger = logging.getLogger(__name__)

class VisualizationPipeline:
    """
    Executes the visualization generation pipeline strictly mapping MetricReports
    into final validated, serialized VisualizationReports.
    """

    def __init__(self, registry: VisualizationRegistry):
        self.registry = registry
        
        # State tracking through pipeline
        self.request: VisualizationRequest | None = None
        self.collections: dict[str, VisualizationCollection] = OrderedDict()
        self.final_report: VisualizationReport | None = None

    def execute(self, request: VisualizationRequest) -> VisualizationReport:
        """Execute the strict visualization pipeline."""
        
        self.request = request
        logger.debug(f"Visualization Pipeline started for EXP: {request.experiment_id}")
        
        # Dispatch to providers to handle (Collect, Prepare, Render, Validate, Export, Report)
        self.dispatch_stage()
        
        # Package everything into the final VisualizationReport
        return self.reporting_stage()

    def dispatch_stage(self) -> None:
        """
        Dispatch the request to all configured visualization providers.
        Each provider strictly follows its immutable BaseVisualizationProvider lifecycle.
        """
        logger.debug("Visualization Pipeline: Dispatch Stage")
        
        groups = self.registry.get_all_groups()
        for group_name in groups:
            providers = self.registry.get_providers_for_group(group_name)
            
            for provider in providers:
                meta = provider.get_metadata()
                logger.debug(f"Dispatching to visualization provider: {meta.name}")
                
                # Providers manage their own lifecycle returning a collection
                collection = provider.execute_lifecycle(self.request)
                
                # Aggregate into pipeline scope
                if meta.name in self.collections:
                    # In a highly modular setup, we might append artifacts instead of overwriting,
                    # but names are unique per provider class.
                    pass
                
                self.collections[meta.name] = collection

    def reporting_stage(self) -> VisualizationReport:
        """Final pipeline stage: Report wrapping all aggregated collections."""
        logger.debug("Visualization Pipeline: Reporting Stage")
        
        if not self.request:
            raise RuntimeError("Cannot generate report without an initial request.")
            
        self.final_report = VisualizationReport(
            experiment_id=self.request.experiment_id,
            collections=self.collections
        )
        self.final_report.validate()
        return self.final_report
