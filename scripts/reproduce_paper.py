import argparse
import logging
from pathlib import Path

from scripts.paper.diagrams import (
    generate_architecture_diagram,
    generate_pipeline_diagram,
)
from scripts.paper.manifest import ManifestManager
from scripts.paper.plots import (
    plot_calibration,
    plot_confusion_matrix,
    plot_grad_cam,
    plot_learning_curves,
    plot_pr_curve,
    plot_roc_curve,
)
from scripts.paper.tables import (
    generate_ablation_table,
    generate_dataset_table,
    generate_model_comparison_table,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the paper generation script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="MediTriageAI Paper Artifact Generator"
    )
    parser.add_argument("--all", action="store_true", help="Generate all artifacts")
    parser.add_argument("--tables", action="store_true", help="Generate tables")
    parser.add_argument(
        "--figures", action="store_true", help="Generate figures (plots)"
    )
    parser.add_argument(
        "--diagrams",
        action="store_true",
        help="Generate architecture and pipeline diagrams",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that generation can proceed without executing",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing paper artifacts before generation",
    )
    parser.add_argument(
        "--manifest-only", action="store_true", help="Only regenerate the manifest file"
    )
    return parser.parse_args()


def main() -> None:
    """Execute the artifact generation logic based on CLI arguments."""
    args = parse_args()

    if not (
        args.all
        or args.tables
        or args.figures
        or args.diagrams
        or args.manifest_only
        or args.verify
        or args.clean
    ):
        logger.info("No target specified. Defaulting to --all.")
        args.all = True

    if args.verify:
        logger.info("Verification mode enabled. Checking directories...")
        for d in [
            "paper_artifacts/diagrams",
            "paper_artifacts/figures",
            "paper_artifacts/tables",
            "paper_artifacts/templates",
            "paper_artifacts/manifests",
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)
        logger.info("Verification complete. Directories are available.")
        return

    logger.info("Starting Reproducible Paper Generation...")
    manifest = ManifestManager()

    if args.clean:
        logger.info(
            "Clean requested. (Not implemented safely yet - skipping dangerous delete)"
        )

    if args.all or args.diagrams:
        logger.info("Generating Diagrams...")
        generate_architecture_diagram(Path("paper_artifacts/diagrams/architecture.png"))
        generate_architecture_diagram(Path("paper_artifacts/diagrams/architecture.pdf"))
        manifest.add_entry(
            "architecture_diagram",
            "E-PATH-CO-REASON Architecture",
            [],
            "None",
            "paper_artifacts/diagrams/architecture.png",
            "scripts/paper/diagrams.py",
        )

        generate_pipeline_diagram(Path("paper_artifacts/diagrams/pipeline.png"))
        generate_pipeline_diagram(Path("paper_artifacts/diagrams/pipeline.pdf"))
        manifest.add_entry(
            "pipeline_diagram",
            "Builder Pipeline Diagram",
            [],
            "None",
            "paper_artifacts/diagrams/pipeline.png",
            "scripts/paper/diagrams.py",
        )

    if args.all or args.tables:
        logger.info("Generating Tables...")
        generate_dataset_table(Path("paper_artifacts/tables/dataset_statistics"))
        manifest.add_entry(
            "dataset_statistics",
            "Dataset size and distributions",
            ["meditriage/data/processed/dataset_statistics.json"],
            "Builder Pipeline",
            "paper_artifacts/tables/dataset_statistics.csv",
            "scripts/paper/tables.py",
        )

        generate_model_comparison_table(Path("paper_artifacts/tables/model_comparison"))
        manifest.add_entry(
            "model_comparison",
            "Model evaluation metrics comparison",
            ["dashboard_web/data/results.json"],
            "Training and Evaluation",
            "paper_artifacts/tables/model_comparison.csv",
            "scripts/paper/tables.py",
        )

        generate_ablation_table(Path("paper_artifacts/templates/ablation_study"))
        manifest.add_entry(
            "ablation_template",
            "Ablation study templates",
            [],
            "Ablation Pipeline",
            "paper_artifacts/templates/ablation_study.csv",
            "scripts/paper/tables.py",
        )

    if args.all or args.figures:
        logger.info("Generating Plot Placeholders...")
        plot_confusion_matrix(
            None, None, [], Path("paper_artifacts/figures/confusion_matrix.png")
        )
        plot_confusion_matrix(
            None, None, [], Path("paper_artifacts/figures/confusion_matrix.pdf")
        )
        manifest.add_entry(
            "confusion_matrix",
            "Confusion Matrix heatmap",
            [],
            "Full Evaluation",
            "paper_artifacts/figures/confusion_matrix.png",
            "scripts/paper/plots.py",
        )

        plot_roc_curve(None, None, Path("paper_artifacts/figures/roc_curve.png"))
        plot_roc_curve(None, None, Path("paper_artifacts/figures/roc_curve.pdf"))
        manifest.add_entry(
            "roc_curve",
            "ROC Curve",
            [],
            "Full Evaluation",
            "paper_artifacts/figures/roc_curve.png",
            "scripts/paper/plots.py",
        )

        plot_pr_curve(None, None, Path("paper_artifacts/figures/pr_curve.png"))
        plot_pr_curve(None, None, Path("paper_artifacts/figures/pr_curve.pdf"))
        manifest.add_entry(
            "pr_curve",
            "PR Curve",
            [],
            "Full Evaluation",
            "paper_artifacts/figures/pr_curve.png",
            "scripts/paper/plots.py",
        )

        plot_calibration(None, None, Path("paper_artifacts/figures/calibration.png"))
        plot_learning_curves(None, Path("paper_artifacts/figures/learning_curves.png"))
        plot_grad_cam(Path("paper_artifacts/figures/grad_cam_attention.png"))

    manifest.save()
    logger.info("Paper artifact generation completed successfully.")


if __name__ == "__main__":
    main()
