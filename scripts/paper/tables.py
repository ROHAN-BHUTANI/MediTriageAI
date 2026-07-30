import json
import logging
from pathlib import Path

import pandas as pd

from .utils import save_table

logger = logging.getLogger(__name__)


def generate_dataset_table(output_base_path: Path) -> None:
    """Generate and save the dataset statistics table.

    Args:
        output_base_path (Path): Base path for the output table.
    """
    stats_path = Path("meditriage/data/processed/dataset_statistics.json")
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            df = pd.DataFrame(
                [
                    {"Metric": "Total Rows", "Value": stats.get("total_rows", "N/A")},
                    {
                        "Metric": "With Department",
                        "Value": stats.get("rows_with_department", "N/A"),
                    },
                    {
                        "Metric": "With Severity",
                        "Value": stats.get("rows_with_severity", "N/A"),
                    },
                    {
                        "Metric": "With Both",
                        "Value": stats.get("rows_with_both", "N/A"),
                    },
                ]
            )
            logger.info("Generated dataset statistics table from real data.")
        except Exception as e:
            logger.error(f"Failed to read {stats_path}: {e}")
            df = pd.DataFrame([{"Metric": "Dataset", "Value": "Error Reading JSON"}])
    else:
        logger.warning(
            f"Missing {stats_path}. Please build the dataset. Generating placeholder table."
        )
        df = pd.DataFrame([{"Metric": "Dataset", "Value": "Pending Generation"}])

    save_table(df, output_base_path)


def generate_model_comparison_table(output_base_path: Path) -> None:
    """Generate and save the model comparison metrics table.

    Args:
        output_base_path (Path): Base path for the output table.
    """
    results_path = Path("dashboard_web/data/results.json")
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            records = []
            for mets in results.get("models", []):
                records.append(
                    {
                        "Model": mets.get("name", "Unknown"),
                        "Spec F1": mets.get("specialist_f1", "Pending"),
                        "Sev F1": mets.get("severity_f1", "Pending"),
                        "Adj Error": mets.get("adjusted_error_rate", "Pending"),
                    }
                )
            df = pd.DataFrame(records)
            logger.info("Generated model comparison table from real metrics.")
        except Exception as e:
            logger.error(f"Failed to read {results_path}: {e}")
            df = pd.DataFrame(
                [
                    {
                        "Model": "Error",
                        "Spec F1": "Error",
                        "Sev F1": "Error",
                        "Adj Error": "Error",
                    }
                ]
            )
    else:
        logger.warning(
            f"Missing {results_path}. Please run evaluations. Generating placeholder table."
        )
        df = pd.DataFrame(
            [
                {
                    "Model": "Baseline",
                    "Spec F1": "Pending",
                    "Sev F1": "Pending",
                    "Adj Error": "Pending",
                },
                {
                    "Model": "E-PATH-CO-REASON",
                    "Spec F1": "Pending",
                    "Sev F1": "Pending",
                    "Adj Error": "Pending",
                },
            ]
        )

    save_table(df, output_base_path)


def generate_ablation_table(output_base_path: Path) -> None:
    """Generate and save the ablation study template table.

    Args:
        output_base_path (Path): Base path for the output table.
    """
    # This is a template
    df = pd.DataFrame(
        [
            {
                "Ablation": "Full E-PATH-CO-REASON",
                "Spec F1": "Pending",
                "Sev F1": "Pending",
            },
            {"Ablation": "w/o Co-reasoning", "Spec F1": "Pending", "Sev F1": "Pending"},
            {
                "Ablation": "w/o Emergent Path",
                "Spec F1": "Pending",
                "Sev F1": "Pending",
            },
        ]
    )
    logger.info("Generated ablation study template table.")
    save_table(df, output_base_path)
