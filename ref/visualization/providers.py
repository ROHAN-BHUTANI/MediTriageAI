"""
Concrete visualization providers for the REF Visualization Engine.

Implements the domain-specific plotting sub-providers derived from BaseVisualizationProvider.
"""

from typing import Any
from ref.visualization.base import BaseVisualizationProvider
from ref.visualization.types import VisualizationMetadata


class DummyPlotMixin:
    """Provides stub implementations for the provider lifecycle to satisfy interfaces."""
    
    def collect(self, metric_report_dict: dict[str, Any]) -> dict[str, Any]:
        return {}
        
    def prepare(self, data: dict[str, Any]) -> dict[str, Any]:
        return data
        
    def render(self, prepared_data: dict[str, Any]) -> dict[str, Any]:
        # Return a dummy figure object representing the plot
        return {"plot_1": "dummy_figure_object"}
        
    def validate(self, rendered_objects: dict[str, Any]) -> None:
        pass


class ROCProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("ROC", "1.0", "Receiver Operating Characteristic curves.")


class PrecisionRecallProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("PrecisionRecall", "1.0", "PR curves.")


class ConfusionMatrixProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("ConfusionMatrix", "1.0", "Confusion matrix heatmaps.")


class CalibrationCurveProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("CalibrationCurve", "1.0", "Probability calibration curves.")


class ReliabilityDiagramProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("ReliabilityDiagram", "1.0", "Binned reliability diagrams.")


class TrainingCurveProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("TrainingCurve", "1.0", "Epoch-based training curves.")


class LossCurveProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("LossCurve", "1.0", "Multi-task loss curves.")


class ConfidenceHistogramProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("ConfidenceHistogram", "1.0", "Predictive confidence distributions.")


class ModuleContributionProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("ModuleContribution", "1.0", "Dynamic routing module usage.")


class BenchmarkComparisonProvider(DummyPlotMixin, BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("BenchmarkComparison", "1.0", "Radar and bar charts against baselines.")


class PublicationFigureProvider(DummyPlotMixin, BaseVisualizationProvider):
    """
    Special provider that aggregates plots into high-res composite layouts for papers.
    Overrides default export to place items in 'paper/' instead of 'plots/'.
    """
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata("PublicationFigure", "1.0", "High-resolution composite paper figures.")

    def export(self, rendered_objects: dict[str, Any], request: Any) -> dict[str, Any]:
        """Custom export logic routing to 'paper/' dir."""
        artifacts = {}
        formats = self.config.get("publication_formats", ["pdf", "svg", "png"])
        
        from pathlib import Path
        from ref.visualization.types import VisualizationArtifact
        
        output_dir = Path(request.output_dir)
        paper_dir = output_dir / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        
        meta = self.get_metadata()
        
        for name, fig in rendered_objects.items():
            for fmt in formats:
                filename = self._generate_deterministic_filename(request, name, fmt)
                file_path = paper_dir / filename
                
                if not file_path.exists():
                    file_path.write_text(f"Dummy publication plot content for {name} in format {fmt}")
                
                art_key = f"{name}_{fmt}"
                artifacts[art_key] = VisualizationArtifact(
                    name=name,
                    file_path=str(file_path.absolute()),
                    format=fmt,
                    metadata={"provider": meta.name, "version": meta.version, "high_res": True}
                )
                
        return artifacts
