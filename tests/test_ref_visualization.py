import shutil

import pytest

from ref.visualization.base import BaseVisualizationProvider
from ref.visualization.pipeline import VisualizationPipeline
from ref.visualization.providers import PublicationFigureProvider
from ref.visualization.registry import VisualizationRegistry
from ref.visualization.types import (
    VisualizationArtifact,
    VisualizationCollection,
    VisualizationMetadata,
    VisualizationReport,
    VisualizationRequest,
    VisualizationValidationError,
)


@pytest.fixture
def temp_output(tmp_path):
    output = tmp_path / "viz_out"
    output.mkdir()
    yield output
    if output.exists():
        shutil.rmtree(output)


class MockPlotProvider(BaseVisualizationProvider):
    def get_metadata(self) -> VisualizationMetadata:
        return VisualizationMetadata(
            name="MockPlot",
            version="1.0",
            description="Mock plot provider",
            dependencies=[],
        )

    def collect(self, metric_report_dict: dict) -> dict:
        return {"data": [1, 2, 3]}

    def prepare(self, data: dict) -> dict:
        return {"prepared": sum(data["data"])}

    def render(self, prepared_data: dict) -> dict:
        return {"my_plot": f"Fig({prepared_data['prepared']})"}

    def validate(self, rendered_objects: dict) -> None:
        if "my_plot" not in rendered_objects:
            raise VisualizationValidationError("Plot missing")


def test_visualization_data_structures():
    req = VisualizationRequest(
        experiment_id="E1", output_dir=".", metric_report_dict={}
    )
    req.validate()

    with pytest.raises(VisualizationValidationError):
        VisualizationRequest(
            experiment_id="", output_dir=".", metric_report_dict={}
        ).validate()

    art = VisualizationArtifact(name="plot", file_path="/fake/path.png", format="png")
    art.validate()

    col = VisualizationCollection(provider_name="Mock", artifacts={"plot": art})
    col.validate()

    rep = VisualizationReport(experiment_id="E1", collections={"Mock": col})
    rep.validate()

    # Test serialization deterministic hash
    assert hasattr(rep, "report_hash")
    d = rep.to_dict()
    assert "report_hash" in d

    rep2 = VisualizationReport.from_dict(d)
    assert rep2.report_hash == rep.report_hash


def test_provider_interface_and_deterministic_exports(temp_output):
    provider = MockPlotProvider({"export_formats": ["png", "svg"]})
    req = VisualizationRequest(
        experiment_id="EXP_TEST", output_dir=str(temp_output), metric_report_dict={}
    )

    collection = provider.execute_lifecycle(req)

    assert isinstance(collection, VisualizationCollection)
    assert len(collection.artifacts) == 2

    # Verify deterministic naming and disk write
    plots_dir = temp_output / "plots"
    assert plots_dir.exists()
    files = list(plots_dir.iterdir())
    assert len(files) == 2

    file_names = [f.name for f in files]
    assert any(f.endswith(".png") for f in file_names)
    assert any(f.endswith(".svg") for f in file_names)
    assert any("my_plot" in f for f in file_names)


def test_publication_figure_isolation(temp_output):
    provider = PublicationFigureProvider({"publication_formats": ["pdf"]})
    req = VisualizationRequest(
        experiment_id="EXP_PUB", output_dir=str(temp_output), metric_report_dict={}
    )

    collection = provider.execute_lifecycle(req)

    paper_dir = temp_output / "paper"
    assert paper_dir.exists()
    files = list(paper_dir.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith(".pdf")

    # Check artifact metadata correctly registered as high_res
    art = list(collection.artifacts.values())[0]
    assert art.metadata.get("high_res") is True


def test_visualization_registry_and_config():
    config = {
        "visualization_engine_enabled": True,
        "enable_plot_roc": True,
        "enable_plot_precisionrecall": False,
        "enable_group_eval": True,
    }

    registry = VisualizationRegistry(config)
    from ref.visualization.providers import PrecisionRecallProvider, ROCProvider

    registry.discover_providers([ROCProvider, PrecisionRecallProvider])

    registry.register("eval", ["ROC", "PrecisionRecall"])

    providers = registry.get_providers_for_group("eval")
    assert len(providers) == 1
    assert isinstance(providers[0], ROCProvider)


def test_visualization_pipeline(temp_output):
    registry = VisualizationRegistry({"export_formats": ["png"]})
    registry.discover_providers([MockPlotProvider])
    registry.register("test_group", ["MockPlot"])

    pipeline = VisualizationPipeline(registry)
    req = VisualizationRequest(
        experiment_id="EXP_PIPE", output_dir=str(temp_output), metric_report_dict={}
    )

    report = pipeline.execute(req)

    assert isinstance(report, VisualizationReport)
    assert "MockPlot" in report.collections
    assert len(report.collections["MockPlot"].artifacts) == 1
