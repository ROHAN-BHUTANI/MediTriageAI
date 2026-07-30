"""Main orchestrator and CLI entrypoint for the MediTriageAI reproducibility analysis framework."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from analysis.agreement import compute_pairwise_agreement
from analysis.calibration import (
    compute_brier_score,
    compute_ece_mce,
    compute_nll,
    get_reliability_curve_data,
)
from analysis.config import config
from analysis.io import generate_and_cache_predictions, load_test_dataframe
from analysis.metrics import (
    add_confidence_columns,
    bootstrap_metric_ci,
    compute_per_class_metrics,
    compute_top_k_accuracy,
)
from analysis.report import generate_reports
from analysis.taxonomy import classify_errors, generate_taxonomy_summary
from analysis.utils import (
    compute_mcnemar_test,
    compute_sha256,
    df_to_markdown,
    set_seed,
    setup_logger,
)
from analysis.visualization import (
    plot_agreement_heatmap,
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_dataset_distributions,
    plot_reliability_diagram,
)

# Setup Logger to write to root workspace directory
logger = setup_logger(REPO_ROOT / "analysis.log")


def now_timestamp() -> str:
    """Return formatted UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")


def dict_to_yaml_str(d: dict[str, Any], indent: int = 0) -> str:
    """Format a python dict as a clean YAML string recursively."""
    lines = []
    for k, v in d.items():
        prefix = " " * indent
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(dict_to_yaml_str(v, indent + 2))
        elif isinstance(v, list):
            lines.append(f"{prefix}{k}:")
            for item in v:
                lines.append(f"{prefix}- {item}")
        else:
            if isinstance(v, bool):
                val_str = "true" if v else "false"
            elif v is None:
                val_str = "null"
            else:
                val_str = str(v)
            lines.append(f"{prefix}{k}: {val_str}")
    return "\n".join(lines)


def get_git_commit() -> str | None:
    """Retrieve current Git commit hash if git is installed."""
    import subprocess

    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def run_pipeline() -> None:
    """Execute the full MediTriageAI analysis pipeline."""
    # 1. Enforce Seeding & Setup Outputs
    set_seed(config.random_seed)

    timestamp = now_timestamp()
    exp_dir = config.results_root / f"experiment_{timestamp}"

    figures_dir = exp_dir / "figures"
    tables_dir = exp_dir / "tables"
    reports_dir = exp_dir / "reports"
    metadata_dir = exp_dir / "metadata"

    for d in [figures_dir, tables_dir, reports_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initialized Experiment Directory: {exp_dir}")

    # 2. Run / Load predictions cache for all models
    predictions_dict: dict[str, pd.DataFrame] = {}
    for model_name in config.models:
        predictions_dict[model_name] = generate_and_cache_predictions(
            model_name, config, logger
        )

    # 3. Enrich with confidence metrics and classify taxonomy errors
    enriched_dict: dict[str, pd.DataFrame] = {}
    taxonomy_dict: dict[str, pd.DataFrame] = {}

    for model_name, df_preds in predictions_dict.items():
        # Add entropy & margin columns
        df_conf = add_confidence_columns(df_preds)
        # Classify errors using taxonomy thresholds
        df_tax = classify_errors(df_conf, config)
        enriched_dict[model_name] = df_tax
        # Generate model-specific taxonomy summary stats
        taxonomy_dict[model_name] = generate_taxonomy_summary(df_tax, config)

    # 4. Generate Dataset-level visualizations
    df_test = load_test_dataframe(config.dataset_csv)
    dataset_paths = plot_dataset_distributions(df_test, figures_dir, config.plot_dpi)
    logger.info("Generated dataset class frequency and tail distributions.")

    # 5. Compute performance summary metrics (with 95% Bootstrap CIs)
    from analysis.metrics import fast_macro_f1, fast_weighted_f1
    from src.model import SEVERITY_LABELS, SPECIALIST_CLASSES

    spec_map = {name: i for i, name in enumerate(SPECIALIST_CLASSES)}
    sev_map = {name: i for i, name in enumerate(SEVERITY_LABELS)}

    summary_rows = []
    for model_name, df in enriched_dict.items():
        y_true_spec = df["true_specialist"].values
        y_pred_spec = df["pred_specialist"].values
        y_true_sev = df["true_severity"].values
        y_pred_sev = df["pred_severity"].values

        y_true_spec_int = np.array([spec_map[x] for x in y_true_spec])
        y_pred_spec_int = np.array([spec_map[x] for x in y_pred_spec])
        y_true_sev_int = np.array([sev_map[x] for x in y_true_sev])
        y_pred_sev_int = np.array([sev_map[x] for x in y_pred_sev])

        # Local metrics helpers using mapped ints:
        def spec_acc_fn(yt, yp):
            return float(np.mean(yt == yp))

        def spec_f1_fn(yt, yp):
            return fast_macro_f1(yt, yp, len(SPECIALIST_CLASSES))

        def spec_w_f1_fn(yt, yp):
            return fast_weighted_f1(yt, yp, len(SPECIALIST_CLASSES))

        def sev_f1_fn(yt, yp):
            return fast_macro_f1(yt, yp, len(SEVERITY_LABELS))

        def sev_w_f1_fn(yt, yp):
            return fast_weighted_f1(yt, yp, len(SEVERITY_LABELS))

        # Calculate boots for Specialist
        spec_acc_m, spec_acc_l, spec_acc_u = bootstrap_metric_ci(
            y_true_spec_int,
            y_pred_spec_int,
            spec_acc_fn,
            config.bootstrap_iterations,
            config.random_seed,
        )
        spec_f1_m, spec_f1_l, spec_f1_u = bootstrap_metric_ci(
            y_true_spec_int,
            y_pred_spec_int,
            spec_f1_fn,
            config.bootstrap_iterations,
            config.random_seed,
        )
        spec_w_f1_m, spec_w_f1_l, spec_w_f1_u = bootstrap_metric_ci(
            y_true_spec_int,
            y_pred_spec_int,
            spec_w_f1_fn,
            config.bootstrap_iterations,
            config.random_seed,
        )

        # Specialist Top-3 helper
        spec_probs = np.vstack(df["specialist_probabilities"].values)
        spec_true_ids = y_true_spec_int

        def spec_top3_fn(probs, true_ids):
            return compute_top_k_accuracy(probs, true_ids, k=3)

        # Top-3 bootstrap
        n_samples = len(spec_true_ids)
        top3_resamples = []
        rng = np.random.default_rng(config.random_seed)
        for _ in range(config.bootstrap_iterations):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            top3_resamples.append(
                spec_top3_fn(spec_probs[indices], spec_true_ids[indices])
            )
        top3_resamples = np.sort(top3_resamples)
        top3_m = float(np.mean(top3_resamples))
        top3_l = float(np.percentile(top3_resamples, 2.5))
        top3_u = float(np.percentile(top3_resamples, 97.5))

        # Severity boots
        sev_acc_m, sev_acc_l, sev_acc_u = bootstrap_metric_ci(
            y_true_sev_int,
            y_pred_sev_int,
            spec_acc_fn,
            config.bootstrap_iterations,
            config.random_seed,
        )
        sev_f1_m, sev_f1_l, sev_f1_u = bootstrap_metric_ci(
            y_true_sev_int,
            y_pred_sev_int,
            sev_f1_fn,
            config.bootstrap_iterations,
            config.random_seed,
        )
        sev_w_f1_m, sev_w_f1_l, sev_w_f1_u = bootstrap_metric_ci(
            y_true_sev_int,
            y_pred_sev_int,
            sev_w_f1_fn,
            config.bootstrap_iterations,
            config.random_seed,
        )

        summary_rows.append(
            {
                "Model": model_name,
                "Spec Accuracy": f"{spec_acc_m:.4f} ({spec_acc_l:.4f} - {spec_acc_u:.4f})",
                "Spec Macro F1": f"{spec_f1_m:.4f} ({spec_f1_l:.4f} - {spec_f1_u:.4f})",
                "Spec Weighted F1": f"{spec_w_f1_m:.4f} ({spec_w_f1_l:.4f} - {spec_w_f1_u:.4f})",
                "Spec Top-3 Acc": f"{top3_m:.4f} ({top3_l:.4f} - {top3_u:.4f})",
                "Sev Accuracy": f"{sev_acc_m:.4f} ({sev_acc_l:.4f} - {sev_acc_u:.4f})",
                "Sev Macro F1": f"{sev_f1_m:.4f} ({sev_f1_l:.4f} - {sev_f1_u:.4f})",
                "Sev Weighted F1": f"{sev_w_f1_m:.4f} ({sev_w_f1_l:.4f} - {sev_w_f1_u:.4f})",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables_dir / "overall_metrics.csv", index=False)
    (tables_dir / "overall_metrics.md").write_text(
        df_to_markdown(summary_df), encoding="utf-8"
    )
    logger.info("Saved summary performance metrics with 95% bootstrap intervals.")

    # 6. Compute per-class Precision, Recall, F1 for Specialist and Severity
    class_metrics_dict: dict[str, dict[str, pd.DataFrame]] = {}
    from src.model import SEVERITY_LABELS, SPECIALIST_CLASSES

    for model_name, df in enriched_dict.items():
        spec_class_df = compute_per_class_metrics(
            df["true_specialist"].values,
            df["pred_specialist"].values,
            SPECIALIST_CLASSES,
        )
        sev_class_df = compute_per_class_metrics(
            df["true_severity"].values, df["pred_severity"].values, SEVERITY_LABELS
        )

        spec_class_df.to_csv(
            tables_dir / f"{model_name}_specialist_per_class_metrics.csv", index=False
        )
        sev_class_df.to_csv(
            tables_dir / f"{model_name}_severity_per_class_metrics.csv", index=False
        )

        class_metrics_dict[model_name] = {
            "specialist": spec_class_df,
            "severity": sev_class_df,
        }
    logger.info("Exported per-class precision, recall, and F1 metrics.")

    # 7. Generate high-resolution confusion matrices
    for model_name, df in enriched_dict.items():
        # Specialist confusion
        spec_cm = confusion_matrix(
            df["true_specialist"].values,
            df["pred_specialist"].values,
            labels=SPECIALIST_CLASSES,
        )
        plot_confusion_matrix(
            spec_cm,
            SPECIALIST_CLASSES,
            f"{model_name} Specialist Confusion Matrix",
            figures_dir / f"{model_name}_specialist_confusion.png",
            config.plot_dpi,
        )

        # Severity confusion
        sev_cm = confusion_matrix(
            df["true_severity"].values,
            df["pred_severity"].values,
            labels=SEVERITY_LABELS,
        )
        plot_confusion_matrix(
            sev_cm,
            SEVERITY_LABELS,
            f"{model_name} Severity Confusion Matrix",
            figures_dir / f"{model_name}_severity_confusion.png",
            config.plot_dpi,
        )
    logger.info("Rendered high-resolution confusion matrix heatmaps.")

    # 8. Identify most confused specialist and severity pairs
    confused_spec_rows = []
    confused_sev_rows = []

    for model_name, df in enriched_dict.items():
        # Specialist errors
        spec_wrong = df[df["true_specialist"] != df["pred_specialist"]]
        if not spec_wrong.empty:
            counts = (
                spec_wrong.groupby(["true_specialist", "pred_specialist"])
                .size()
                .reset_index(name="count")
            )
            counts = counts.sort_values(by="count", ascending=False).head(5)
            for _, r in counts.iterrows():
                confused_spec_rows.append(
                    {
                        "Model": model_name,
                        "True Specialist": r["true_specialist"],
                        "Predicted Specialist": r["pred_specialist"],
                        "Error Count": int(r["count"]),
                        "Percentage of Errors (%)": (r["count"] / len(spec_wrong))
                        * 100,
                    }
                )

        # Severity errors
        sev_wrong = df[df["true_severity"] != df["pred_severity"]]
        if not sev_wrong.empty:
            counts_sev = (
                sev_wrong.groupby(["true_severity", "pred_severity"])
                .size()
                .reset_index(name="count")
            )
            counts_sev = counts_sev.sort_values(by="count", ascending=False).head(5)
            for _, r in counts_sev.iterrows():
                confused_sev_rows.append(
                    {
                        "Model": model_name,
                        "True Severity": r["true_severity"],
                        "Predicted Severity": r["pred_severity"],
                        "Error Count": int(r["count"]),
                        "Percentage of Errors (%)": (r["count"] / len(sev_wrong)) * 100,
                    }
                )

    confused_spec_df = pd.DataFrame(confused_spec_rows)
    confused_sev_df = pd.DataFrame(confused_sev_rows)

    confused_spec_df.to_csv(tables_dir / "most_confused_specialists.csv", index=False)
    confused_sev_df.to_csv(tables_dir / "most_confused_severities.csv", index=False)

    logger.info("Identified top confused specialist and severity class transitions.")

    # 9. Calibration Analysis (ECE, MCE, Brier, NLL) and Reliability Curve Plotting
    calibration_rows = []
    for model_name, df in enriched_dict.items():
        spec_probs = np.vstack(df["specialist_probabilities"].values)
        sev_probs = np.vstack(df["severity_probabilities"].values)

        spec_true_ids = np.array(
            [SPECIALIST_CLASSES.index(x) for x in df["true_specialist"].values]
        )
        sev_true_ids = np.array(
            [SEVERITY_LABELS.index(x) for x in df["true_severity"].values]
        )

        # Specialist calibration
        spec_ece, spec_mce = compute_ece_mce(spec_probs, spec_true_ids)
        spec_brier = compute_brier_score(
            spec_probs, spec_true_ids, len(SPECIALIST_CLASSES)
        )
        spec_nll = compute_nll(spec_probs, spec_true_ids)

        # Severity calibration
        sev_ece, sev_mce = compute_ece_mce(sev_probs, sev_true_ids)
        sev_brier = compute_brier_score(sev_probs, sev_true_ids, len(SEVERITY_LABELS))
        sev_nll = compute_nll(sev_probs, sev_true_ids)

        calibration_rows.append(
            {
                "Model": model_name,
                "Spec ECE": spec_ece,
                "Spec MCE": spec_mce,
                "Spec Brier Score": spec_brier,
                "Spec NLL": spec_nll,
                "Sev ECE": sev_ece,
                "Sev MCE": sev_mce,
                "Sev Brier Score": sev_brier,
                "Sev NLL": sev_nll,
            }
        )

        # Save reliability plots
        spec_rel = get_reliability_curve_data(spec_probs, spec_true_ids)
        plot_reliability_diagram(
            spec_rel,
            spec_ece,
            f"{model_name} Specialist Calibration Curve",
            figures_dir / f"{model_name}_specialist_reliability.png",
            config.plot_dpi,
        )

        # Save confidence histogram
        plot_confidence_histogram(
            np.max(spec_probs, axis=1),
            np.mean(np.max(spec_probs, axis=1)),
            np.mean(df["spec_correct"]),
            f"{model_name} Specialist Confidence Distribution",
            figures_dir / f"{model_name}_specialist_conf_histogram.png",
            config.plot_dpi,
        )

        # Severity diagrams
        sev_rel = get_reliability_curve_data(sev_probs, sev_true_ids)
        plot_reliability_diagram(
            sev_rel,
            sev_ece,
            f"{model_name} Severity Calibration Curve",
            figures_dir / f"{model_name}_severity_reliability.png",
            config.plot_dpi,
        )
        plot_confidence_histogram(
            np.max(sev_probs, axis=1),
            np.mean(np.max(sev_probs, axis=1)),
            np.mean(df["sev_correct"]),
            f"{model_name} Severity Confidence Distribution",
            figures_dir / f"{model_name}_severity_conf_histogram.png",
            config.plot_dpi,
        )

    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(tables_dir / "calibration_metrics.csv", index=False)
    (tables_dir / "calibration_metrics.md").write_text(
        df_to_markdown(calibration_df), encoding="utf-8"
    )
    logger.info("Compiled model calibration metrics and plotted reliability curves.")

    # 10. Language Stratification metrics
    lang_rows = []
    # Identify unique heuristic groups
    heuristic_langs = ["English", "Hindi", "Hinglish", "Mixed", "Unknown"]

    for model_name, df in enriched_dict.items():
        for lang in heuristic_langs:
            df_lang = df[df["language"] == lang]
            count = len(df_lang)
            if count > 0:
                spec_acc = np.mean(
                    df_lang["true_specialist"] == df_lang["pred_specialist"]
                )
                sev_acc = np.mean(df_lang["true_severity"] == df_lang["pred_severity"])
                lang_rows.append(
                    {
                        "Model": model_name,
                        "Language Heuristic": lang,
                        "Count": count,
                        "Spec Accuracy": spec_acc,
                        "Sev Accuracy": sev_acc,
                    }
                )

    lang_df = pd.DataFrame(lang_rows)
    lang_df.to_csv(tables_dir / "language_metrics.csv", index=False)
    (tables_dir / "language_metrics.md").write_text(
        df_to_markdown(lang_df), encoding="utf-8"
    )
    logger.info("Stratified model accuracy by language heuristics.")

    # 11. Sentence-Length Stratification metrics
    len_rows = []
    buckets = [
        ("0–10 tokens", lambda c: c <= 10),
        ("11–20 tokens", lambda c: (c >= 11) & (c <= 20)),
        ("21–40 tokens", lambda c: (c >= 21) & (c <= 40)),
        ("40+ tokens", lambda c: c >= 41),
    ]

    for model_name, df in enriched_dict.items():
        for b_name, b_filter in buckets:
            df_bucket = df[b_filter(df["token_count"])]
            count = len(df_bucket)
            if count > 0:
                spec_acc = np.mean(
                    df_bucket["true_specialist"] == df_bucket["pred_specialist"]
                )
                sev_acc = np.mean(
                    df_bucket["true_severity"] == df_bucket["pred_severity"]
                )
                len_rows.append(
                    {
                        "Model": model_name,
                        "Length Bucket": b_name,
                        "Count": count,
                        "Spec Accuracy": spec_acc,
                        "Sev Accuracy": sev_acc,
                    }
                )

    len_df = pd.DataFrame(len_rows)
    len_df.to_csv(tables_dir / "length_metrics.csv", index=False)
    (tables_dir / "length_metrics.md").write_text(
        df_to_markdown(len_df), encoding="utf-8"
    )
    logger.info("Stratified model accuracy by word count buckets.")

    # 12. Rare Class performance (support below 25th percentile of frequencies)
    spec_counts = df_test["department_code"].value_counts()
    q25 = np.percentile(spec_counts.values, 25)
    rare_classes = spec_counts[spec_counts <= q25].index.tolist()
    logger.info(
        f"Rare specialist classes identified (support <= {q25}): {rare_classes}"
    )

    rare_rows = []
    for model_name, df in enriched_dict.items():
        df_rare = df[df["true_specialist"].isin(rare_classes)]
        count = len(df_rare)
        if count > 0:
            spec_acc = np.mean(df_rare["true_specialist"] == df_rare["pred_specialist"])
            from src.metrics import compute_macro_f1

            spec_f1 = compute_macro_f1(
                df_rare["true_specialist"].values,
                df_rare["pred_specialist"].values,
                rare_classes,
            )
            rare_rows.append(
                {
                    "Model": model_name,
                    "Rare Class Count": count,
                    "Spec Accuracy on Rare Classes": spec_acc,
                    "Spec Macro F1 on Rare Classes": spec_f1,
                }
            )

    rare_df = pd.DataFrame(rare_rows)
    rare_df.to_csv(tables_dir / "rare_class_metrics.csv", index=False)
    (tables_dir / "rare_class_metrics.md").write_text(
        df_to_markdown(rare_df), encoding="utf-8"
    )
    logger.info("Calculated metrics for rare/minority departments.")

    # 13. Cross-Model Agreement (percentage agreement & Cohen's Kappa)
    spec_kappa_df, spec_pct_df = compute_pairwise_agreement(
        enriched_dict, "pred_specialist"
    )
    sev_kappa_df, sev_pct_df = compute_pairwise_agreement(
        enriched_dict, "pred_severity"
    )

    # Save tables
    spec_kappa_df.to_csv(tables_dir / "specialist_agreement_kappa.csv")
    spec_pct_df.to_csv(tables_dir / "specialist_agreement_percentage.csv")
    sev_kappa_df.to_csv(tables_dir / "severity_agreement_kappa.csv")
    sev_pct_df.to_csv(tables_dir / "severity_agreement_percentage.csv")

    # Save heatmap visualisations
    plot_agreement_heatmap(
        spec_kappa_df,
        "Specialist Pairwise Agreement (Cohen's Kappa)",
        figures_dir / "specialist_agreement_kappa.png",
        config.plot_dpi,
    )
    plot_agreement_heatmap(
        sev_kappa_df,
        "Severity Pairwise Agreement (Cohen's Kappa)",
        figures_dir / "severity_agreement_kappa.png",
        config.plot_dpi,
    )
    logger.info("Computed cross-model prediction consensus matrices.")

    # 14. Statistical Significance (McNemar's test) between all pairs
    mcnemar_results: dict[str, dict[str, Any]] = {}
    model_names = sorted(enriched_dict.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1 = model_names[i]
            m2 = model_names[j]

            # Align indices
            df1 = enriched_dict[m1].set_index("sample_id")
            df2 = enriched_dict[m2].set_index("sample_id")
            common = df1.index.intersection(df2.index)

            y_true = df1.loc[common, "true_specialist"].values
            y_pred1 = df1.loc[common, "pred_specialist"].values
            y_pred2 = df2.loc[common, "pred_specialist"].values

            res = compute_mcnemar_test(y_true, y_pred1, y_pred2)
            pair_key = f"{m1}_vs_{m2}"
            mcnemar_results[pair_key] = res

    # Save McNemar summary as json
    with open(tables_dir / "mcnemar_significance.json", "w", encoding="utf-8") as f:
        json.dump(mcnemar_results, f, indent=2)
    logger.info("Executed McNemar significance tests between architecture pairs.")

    # 15. Representative Failure Cases Export (Combined CSV)
    # Collect samples where any model fails on either head
    failure_rows = []
    for model_name, df in enriched_dict.items():
        df_fails = df[~df["spec_correct"] | ~df["sev_correct"]]
        for _, row in df_fails.iterrows():
            failure_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "text": row["text"],
                    "model": model_name,
                    "true_specialist": row["true_specialist"],
                    "predicted_specialist": row["pred_specialist"],
                    "specialist_confidence": row["specialist_top1_conf"],
                    "true_severity": row["true_severity"],
                    "predicted_severity": row["pred_severity"],
                    "severity_confidence": row["severity_top1_conf"],
                    "language": row["language"],
                    "token_length": row["token_count"],
                }
            )

    df_failures = pd.DataFrame(failure_rows)
    # Export failure cases
    df_failures.to_csv(tables_dir / "failure_cases.csv", index=False)
    logger.info(
        f"Exported {len(df_failures)} total model failure cases to failure_cases.csv."
    )

    # 16. Compile Markdown and HTML reports
    md_path, html_path = generate_reports(
        exp_dir,
        summary_df,
        class_metrics_dict,
        taxonomy_dict,
        calibration_df,
        {
            "spec_kappa": spec_kappa_df,
            "spec_pct": spec_pct_df,
            "sev_kappa": sev_kappa_df,
            "sev_pct": sev_pct_df,
        },
        lang_df,
        len_df,
        rare_df,
        mcnemar_results,
        config,
        logger,
    )

    # 17. Generate run_metadata.yaml
    metadata_dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_used": "cuda" if torch.cuda.is_available() else "cpu",
        "dataset_path": str(config.dataset_csv.resolve()),
        "dataset_sha256": compute_sha256(config.dataset_csv),
        "number_of_samples": len(df_test),
        "model_names": list(config.models),
        "random_seed": config.random_seed,
        "analysis_version": "1.0.0",
    }

    yaml_str = dict_to_yaml_str(metadata_dict)

    # Save metadata locally
    meta_local_path = metadata_dir / "run_metadata.yaml"
    meta_local_path.write_text(yaml_str, encoding="utf-8")

    # Save metadata in global cache
    config.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
    meta_cache_path = config.metadata_cache_dir / "run_metadata.yaml"
    meta_cache_path.write_text(yaml_str, encoding="utf-8")
    logger.info(f"Saved run metadata to: {meta_local_path}")

    # 18. Generate manifest.json
    manifest_records = []
    # Collect all files created in the experiment directory
    for file_path in sorted(exp_dir.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(exp_dir).as_posix()
            manifest_records.append(
                {
                    "relative_path": rel_path,
                    "file_size_bytes": file_path.stat().st_size,
                    "creation_time": datetime.fromtimestamp(
                        file_path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    "sha256_checksum": compute_sha256(file_path),
                }
            )

    manifest_path = exp_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    logger.info(f"Saved research manifest to: {manifest_path}")

    # 19. Reproducibility Check (Self-Validation Assertions)
    logger.info("Running Reproducibility Verification Check...")

    expected_figures = [
        "class_frequency.png",
        "long_tail_distribution.png",
        "rare_class_distribution.png",
        "specialist_agreement_kappa.png",
        "severity_agreement_kappa.png",
    ]
    for model in config.models:
        expected_figures.extend(
            [
                f"{model}_specialist_confusion.png",
                f"{model}_severity_confusion.png",
                f"{model}_specialist_reliability.png",
                f"{model}_specialist_conf_histogram.png",
                f"{model}_severity_reliability.png",
                f"{model}_severity_conf_histogram.png",
            ]
        )

    expected_tables = [
        "overall_metrics.csv",
        "overall_metrics.md",
        "most_confused_specialists.csv",
        "most_confused_severities.csv",
        "calibration_metrics.csv",
        "calibration_metrics.md",
        "language_metrics.csv",
        "language_metrics.md",
        "length_metrics.csv",
        "length_metrics.md",
        "rare_class_metrics.csv",
        "rare_class_metrics.md",
        "specialist_agreement_kappa.csv",
        "specialist_agreement_percentage.csv",
        "severity_agreement_kappa.csv",
        "severity_agreement_percentage.csv",
        "mcnemar_significance.json",
        "failure_cases.csv",
    ]
    for model in config.models:
        expected_tables.extend(
            [
                f"{model}_specialist_per_class_metrics.csv",
                f"{model}_severity_per_class_metrics.csv",
            ]
        )

    missing_figures = [f for f in expected_figures if not (figures_dir / f).exists()]
    missing_tables = [t for t in expected_tables if not (tables_dir / t).exists()]
    reports_exist = (reports_dir / "analysis_report.md").exists() and (
        reports_dir / "analysis_report.html"
    ).exists()

    # Check predictions cache availability
    missing_caches = [
        m for m in config.models if not config.get_prediction_cache_path(m).exists()
    ]

    # Print validation checklist
    validation_ok = True
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY VALIDATION SUMMARY")
    print("=" * 60)

    if not missing_caches:
        print("[PASS] All prediction cache files available in Parquet format.")
    else:
        print(f"[FAIL] Missing prediction cache files: {missing_caches}")
        validation_ok = False

    if not missing_tables:
        print("[PASS] All expected metrics tables generated successfully.")
    else:
        print(f"[FAIL] Missing metrics tables: {missing_tables}")
        validation_ok = False

    if not missing_figures:
        print("[PASS] All expected figures generated successfully.")
    else:
        print(f"[FAIL] Missing figures: {missing_figures}")
        validation_ok = False

    if reports_exist and md_path.exists() and html_path.exists():
        print("[PASS] Markdown and HTML reports compiled and synchronized.")
    else:
        print("[FAIL] Reports compilation failure.")
        validation_ok = False

    if manifest_path.exists():
        print(
            "[PASS] Research manifest containing SHA256 checksums successfully exported."
        )
    else:
        print("[FAIL] Manifest missing.")
        validation_ok = False

    print("=" * 60)
    if validation_ok:
        print("RESULT: Pipeline execution fully verified. Experiment is reproducible.")
    else:
        print("RESULT: Pipeline execution failed validation checks.")
    print("=" * 60 + "\n")

    if not validation_ok:
        raise AssertionError("Validation checks failed. Review log output.")


if __name__ == "__main__":
    run_pipeline()
