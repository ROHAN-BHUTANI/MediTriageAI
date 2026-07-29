"""
Base provider for the REF Visualization Engine.

Defines the strictly enforced lifecycle for all visualization providers:
collect() -> prepare() -> render() -> validate() -> export() -> report().
"""

from abc import ABC, abstractmethod
from typing import Any
import logging
from pathlib import Path
import hashlib

from ref.visualization.types import VisualizationCollection, VisualizationMetadata, VisualizationRequest, VisualizationArtifact

logger = logging.getLogger(__name__)

class BaseVisualizationProvider(ABC):
    """
    Abstract base class for all visualization engines.
    Forces strict decoupling of metric collection from matplotlib/seaborn rendering.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.raw_data: dict[str, Any] = {}
        self.prepared_data: dict[str, Any] = {}
        self.rendered_objects: dict[str, Any] = {}  # e.g., matplotlib figures
        self.exported_artifacts: dict[str, VisualizationArtifact] = {}
        self.collection: VisualizationCollection | None = None

    @abstractmethod
    def get_metadata(self) -> VisualizationMetadata:
        """Define identity and versioning for this provider."""

    def execute_lifecycle(self, request: VisualizationRequest) -> VisualizationCollection:
        """
        Immutable execution template.
        """
        logger.debug(f"Executing visualization lifecycle: {self.__class__.__name__}")
        
        # Stage 1: Collect
        self.raw_data = self.collect(request.metric_report_dict)
        
        # Stage 2: Prepare
        self.prepared_data = self.prepare(self.raw_data)
        
        # Stage 3: Render
        self.rendered_objects = self.render(self.prepared_data)
        
        # Stage 4: Validate
        self.validate(self.rendered_objects)
        
        # Stage 5: Export
        self.exported_artifacts = self.export(self.rendered_objects, request)
        
        # Stage 6: Report
        self.collection = self.report(self.exported_artifacts)
        
        return self.collection

    @abstractmethod
    def collect(self, metric_report_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Extract required metrics/tensors from the centralized MetricReport dictionary.
        """

    @abstractmethod
    def prepare(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Format metrics specifically for plotting (e.g. binning, smoothing).
        """

    @abstractmethod
    def render(self, prepared_data: dict[str, Any]) -> dict[str, Any]:
        """
        Produce actual plot objects (matplotlib Figure, plotly Figure, etc).
        Returns a dict of name -> figure_object.
        """

    @abstractmethod
    def validate(self, rendered_objects: dict[str, Any]) -> None:
        """
        Assert that plots are populated, axes are labeled, or data wasn't empty.
        Raises VisualizationValidationError if constraints fail.
        """

    def _generate_deterministic_filename(self, request: VisualizationRequest, plot_name: str, format: str) -> str:
        """Helper to create unique hashed filename."""
        raw = f"{request.experiment_id}_{self.get_metadata().name}_{plot_name}"
        uid = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]
        # Clean up the name for files
        clean_name = plot_name.replace(" ", "_").lower()
        return f"{clean_name}_{uid}.{format}"

    def export(self, rendered_objects: dict[str, Any], request: VisualizationRequest) -> dict[str, VisualizationArtifact]:
        """
        Default export implementation saving to disk.
        Subclasses can override if they need complex export (e.g. PDF multi-page).
        By default, simulates/stubs the save to disk.
        """
        artifacts = {}
        formats = self.config.get("export_formats", ["png"])
        
        output_dir = Path(request.output_dir)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        meta = self.get_metadata()
        
        for name, fig in rendered_objects.items():
            for fmt in formats:
                filename = self._generate_deterministic_filename(request, name, fmt)
                file_path = plots_dir / filename
                
                # In a real implementation we would call fig.savefig(file_path, format=fmt)
                # We stub it by writing a dummy file for the artifact pointers
                if not file_path.exists():
                    file_path.write_text(f"Dummy plot content for {name} in format {fmt}")
                
                # Register artifact
                art_key = f"{name}_{fmt}"
                artifacts[art_key] = VisualizationArtifact(
                    name=name,
                    file_path=str(file_path.absolute()),
                    format=fmt,
                    metadata={"provider": meta.name, "version": meta.version}
                )
                
        return artifacts

    def report(self, exported_artifacts: dict[str, VisualizationArtifact]) -> VisualizationCollection:
        """
        Wrap the artifact pointers into a strict VisualizationCollection structure.
        """
        meta = self.get_metadata()
        collection = VisualizationCollection(
            provider_name=meta.name,
            artifacts=exported_artifacts
        )
        collection.validate()
        return collection
