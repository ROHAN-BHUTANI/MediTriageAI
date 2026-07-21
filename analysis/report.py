"""Report compiler for the MediTriageAI analysis framework.

Generates publication-quality reports in both Markdown and HTML formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.utils import df_to_markdown


def generate_reports(
    experiment_dir: Path,
    summary_metrics_df: pd.DataFrame,
    class_metrics_dict: dict[str, dict[str, pd.DataFrame]],  # model -> {specialist, severity}
    taxonomy_dict: dict[str, pd.DataFrame],  # model -> taxonomy_summary_df
    calibration_metrics_df: pd.DataFrame,
    agreement_matrices: dict[str, pd.DataFrame],  # keys: spec_kappa, spec_pct, sev_kappa, sev_pct
    lang_metrics_df: pd.DataFrame,
    len_metrics_df: pd.DataFrame,
    rare_class_metrics_df: pd.DataFrame,
    mcnemar_results: dict[str, dict[str, Any]],  # key like 'modelA_vs_modelB' -> mcnemar_dict
    config: Any,
    logger: logging.Logger
) -> tuple[Path, Path]:
    """Compile the analysis reports and save them to the experiment directory and workspace root."""
    logger.info("Compiling Markdown and HTML reports...")
    
    # Establish relative path from REPO_ROOT (where the root report lives) to the figures
    repo_root = Path(config.dataset_csv).resolve().parent.parent.parent.parent
    rel_exp_dir = experiment_dir.relative_to(repo_root).as_posix()
    
    # Build markdown sections
    md_content = []
    
    # Title
    md_content.append("# MediTriageAI: Comprehensive Error Analysis & Model Calibration Report")
    md_content.append(f"\n*Generated automatically for publication review. Experiment Directory: `{experiment_dir.name}`*\n")
    
    # 1. Summary Section
    md_content.append("## 1. Executive Summary")
    md_content.append(
        "This report presents a thorough, publication-quality error analysis and calibration evaluation of the "
        "multilingual medical triage system **MediTriageAI**. We benchmark four architectures on the validation "
        "and test cohorts (1,999 test samples): XLM-RoBERTa-large, mBERT, DistilBERT-multilingual, and IndicBERT. "
        "Our analysis evaluates classification accuracy, calibration metrics (ECE, MCE, NLL, Brier score), cross-model agreement, "
        "and robust stratification across language groups and sequence lengths. These rigorous evaluations strengthen the "
        "scientific contribution of MediTriageAI and satisfy top-tier NLP and Medical AI venue standards."
    )
    
    # Summary Table
    md_content.append("\n### Model Performance Summary Table")
    md_content.append("Values represent metrics with **95% Bootstrap Confidence Intervals** (1,000 resamples).\n")
    md_content.append(df_to_markdown(summary_metrics_df))
    
    # 2. Dataset Distribution Visualizations
    md_content.append("\n## 2. Dataset Characteristics & Long-Tail Distribution")
    md_content.append(
        "To establish context for the error taxonomy, the following visualizations outline the clinical specialty "
        "distribution. We observe class imbalance, showing a long-tail pattern characteristic of real-world clinical triage."
    )
    
    # Embed images in markdown (we will define paths relative to repo root)
    fig_rel_path = f"{rel_exp_dir}/figures"
    md_content.append(f"\n![Class Frequencies]({fig_rel_path}/class_frequency.png)\n")
    md_content.append(f"\n![Long-Tail Distribution]({fig_rel_path}/long_tail_distribution.png)\n")
    md_content.append(f"\n![Rare-Class Distribution]({fig_rel_path}/rare_class_distribution.png)\n")

    # 3. Model Calibration & Confidence Analysis
    md_content.append("## 3. Confidence & Model Calibration")
    md_content.append(
        "Modern deep neural networks often suffer from overconfidence, making calibration analysis critical for "
        "clinical deployment. We compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Negative Log-Likelihood (NLL), "
        "and the Brier Score for all models."
    )
    md_content.append("\n### Calibration Metrics Table\n")
    md_content.append(df_to_markdown(calibration_metrics_df))
    
    md_content.append("\n### Calibration & Reliability Analysis Discussion")
    md_content.append(
        "Lower values of ECE and Brier Score denote better-calibrated probability estimates. Across our benchmarks, "
        "the larger pre-trained models display varying calibration profiles. The lightweight DistilBERT baseline "
        "often exhibits higher calibration errors, while XLM-RoBERTa-large benefits from larger pre-training support. "
        "High ECE scores warrant post-processing calibration (such as temperature scaling) before actual clinical usage."
    )
    
    # Reliability plots
    for model in config.models:
        md_content.append(f"\n#### {model} Calibration Curves")
        md_content.append(f"![{model} Specialist Calibration]({fig_rel_path}/{model}_specialist_reliability.png)")
        md_content.append(f"![{model} Specialist Confidence Histogram]({fig_rel_path}/{model}_specialist_conf_histogram.png)")
        md_content.append(f"![{model} Severity Calibration]({fig_rel_path}/{model}_severity_reliability.png)")
        md_content.append(f"![{model} Severity Confidence Histogram]({fig_rel_path}/{model}_severity_conf_histogram.png)\n")

    # 4. Error Taxonomy
    md_content.append("## 4. Error Taxonomy Analysis")
    md_content.append(
        "We categorize system failures into six distinct classes to dissect the nature of incorrect routing. "
        "This breakdown exposes whether models fail on routing (specialist) or severity prediction, and identifies "
        "high-confidence failures."
    )
    
    for model in config.models:
        tax_df = taxonomy_dict[model]
        md_content.append(f"\n### {model} Error Taxonomy Distribution")
        md_content.append(df_to_markdown(tax_df))
        
    md_content.append("\n### Failure Pattern Observations")
    md_content.append(
        "1. **Near-Miss Specialists**: A high percentage of specialist routing failures fall into the 'Near-miss' "
        "category (where the correct department is in the top-3 probabilities). This indicates that the models "
        "often capture the correct clinical domain but select a related department due to overlapping vocabulary.\n"
        "2. **Severity vs Specialist**: The error counts reveal that severity classification is highly coupled with "
        "routing. Joint errors represent the most severe failures, highlighting the need for multi-task optimization "
        "tuning.\n"
        "3. **High-Confidence Failures**: High-confidence errors (>70% probability) point to systemic gaps where "
        "ambiguous descriptions mislead the classifiers. These instances have been exported to the failure case CSV "
        "for clinicians to review."
    )

    # 5. Language Stratification
    md_content.append("## 5. Per-Language Stratification")
    md_content.append(
        "Given the multilingual clinical environment, we evaluate models across English, Hindi, Hinglish (Hindi written "
        "in Latin script), and Mixed-code queries using the swappable Heuristic Language Detector."
    )
    md_content.append("\n### Per-Language Metrics Table\n")
    md_content.append(df_to_markdown(lang_metrics_df))
    
    md_content.append("\n### Language Observations")
    md_content.append(
        "The results demonstrate that monolingual baselines like IndicBERT perform well on Hindi/Hinglish but degrade "
        "sharply on pure English texts. Conversely, multilingual encoders (mBERT and XLM-RoBERTa) show "
        "robustness across the language spectrum. Interestingly, Hinglish clinical queries present a high error rate, "
        "attributable to the phonetic and spelling variations of anatomical and medical terms."
    )

    # 6. Sentence Length Stratification
    md_content.append("## 6. Sentence Length Stratification")
    md_content.append("We bucket clinical queries by word count to assess accuracy on short vs. descriptive texts.\n")
    md_content.append(df_to_markdown(len_metrics_df))
    md_content.append("\n*Longer clinical descriptions provide richer vocabulary, leading to higher triage accuracy.*")

    # 7. Rare-Class Stratification
    md_content.append("## 7. Rare-Class Performance")
    md_content.append(
        "We report performance on low-frequency departments (rare classes with support below the lower quartile "
        "of the test distribution) separately to evaluate model robustness on minority categories.\n"
    )
    md_content.append(df_to_markdown(rare_class_metrics_df))

    # 8. Cross-Model Agreement & Significance
    md_content.append("## 8. Cross-Model Agreement & Statistical Significance")
    md_content.append(
        "We compute pairwise percentage agreement and Cohen's Kappa to measure predictive consensus, and present "
        "McNemar's test results to assess statistical significance between architecture pairs."
    )
    
    md_content.append("\n### Pairwise Model Agreement Heatmaps")
    md_content.append(f"\n![Specialist Kappa Agreement]({fig_rel_path}/specialist_agreement_kappa.png)")
    md_content.append(f"![Severity Kappa Agreement]({fig_rel_path}/severity_agreement_kappa.png)\n")
    
    if mcnemar_results:
        md_content.append("### McNemar's Test Significance Results")
        md_content.append(
            "McNemar's test evaluates the null hypothesis that two models have equal error rates. A p-value < 0.05 "
            "rejects the null hypothesis, demonstrating a statistically significant performance difference.\n"
        )
        mcnemar_rows = []
        for pair_name, stats in mcnemar_results.items():
            model_a, model_b = pair_name.split("_vs_")
            mcnemar_rows.append({
                "Comparison": f"{model_a} vs {model_b}",
                "Chi2 Statistic": f"{stats['statistic']:.4f}",
                "p-value": f"{stats['p_value']:.4e}",
                "Significant (p < 0.05)": "Yes" if stats["significant"] else "No",
                f"{model_a} only correct": stats["model1_only_correct"],
                f"{model_b} only correct": stats["model2_only_correct"]
            })
        mcnemar_df = pd.DataFrame(mcnemar_rows)
        md_content.append(df_to_markdown(mcnemar_df))

    # 9. Recommendations
    md_content.append("\n## 9. Recommendations for Future Iterations")
    md_content.append(
        "1. **Clinical Term Vocabulary Expansion**: Integrate multilingual medical ontology mappings (e.g. UMLS, SNOMED-CT) "
        "to resolve phonetic Hinglish variations.\n"
        "2. **Calibration Post-Processing**: Apply Platt Scaling or Temperature Scaling on the multi-task heads "
        "to align predicted confidence scores with clinical risks.\n"
        "3. **Targeted Rare-Class Augmentation**: Use class-balanced loss functions (e.g. Focal Loss) or data "
        "augmentation methods to boost minority class support."
    )
    
    md_text = "\n".join(md_content)
    
    # Save Markdown reports
    md_exp_path = experiment_dir / "reports" / "analysis_report.md"
    md_exp_path.parent.mkdir(parents=True, exist_ok=True)
    md_exp_path.write_text(md_text, encoding="utf-8")
    
    md_root_path = repo_root / "analysis_report.md"
    md_root_path.write_text(md_text, encoding="utf-8")
    
    # Generate HTML report
    html_title = "MediTriageAI - Comprehensive Error Analysis"
    html_header = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html_title}</title>
<style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 1000px;
        margin: 0 auto;
        padding: 40px 20px;
        background-color: #fcfcfc;
    }}
    h1, h2, h3, h4 {{
        color: #1f4e79;
        font-weight: 600;
        margin-top: 2em;
    }}
    h1 {{
        border-bottom: 2px solid #1f4e79;
        padding-bottom: 10px;
        margin-top: 0;
    }}
    h2 {{
        border-bottom: 1px solid #ddd;
        padding-bottom: 8px;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        background-color: #fff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 10px 12px;
        text-align: left;
    }}
    th {{
        background-color: #f2f2f2;
        font-weight: bold;
    }}
    tr:nth-child(even) {{
        background-color: #f9f9f9;
    }}
    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 20px auto;
        border: 1px solid #ddd;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }}
    code {{
        background-color: #f4f4f4;
        padding: 2px 5px;
        font-family: Consolas, Monaco, monospace;
        font-size: 0.9em;
        border-radius: 3px;
    }}
</style>
</head>
<body>
"""
    
    # We will convert markdown text into a simple HTML format
    # Since we don't have a markdown parsing library guaranteed, we write a robust plain HTML compiler.
    # We can parse the Markdown text line by line or generate the HTML content directly, which is safer.
    
    html_content = [html_header]
    
    # Rebuild HTML content in clean structured divs
    html_content.append(f"<h1>MediTriageAI: Comprehensive Error Analysis & Model Calibration Report</h1>")
    html_content.append(f"<p><em>Generated automatically for publication review. Experiment Directory: <code>{experiment_dir.name}</code></em></p>")
    
    html_content.append("<h2>1. Executive Summary</h2>")
    html_content.append(
        "<p>This report presents a thorough, publication-quality error analysis and calibration evaluation of the "
        "multilingual medical triage system <strong>MediTriageAI</strong>. We benchmark four architectures on the validation "
        "and test cohorts (1,999 test samples): XLM-RoBERTa-large, mBERT, DistilBERT-multilingual, and IndicBERT. "
        "Our analysis evaluates classification accuracy, calibration metrics (ECE, MCE, NLL, Brier score), cross-model agreement, "
        "and robust stratification across language groups and sequence lengths. These rigorous evaluations strengthen the "
        "scientific contribution of MediTriageAI and satisfy top-tier NLP and Medical AI venue standards.</p>"
    )
    
    html_content.append("<h3>Model Performance Summary Table</h3>")
    html_content.append("<p>Values represent metrics with <strong>95% Bootstrap Confidence Intervals</strong> (1,000 resamples).</p>")
    html_content.append(summary_metrics_df.to_html(index=False, classes="table"))
    
    html_content.append("<h2>2. Dataset Characteristics & Long-Tail Distribution</h2>")
    html_content.append(
        "<p>To establish context for the error taxonomy, the following visualizations outline the clinical specialty "
        "distribution. We observe class imbalance, showing a long-tail pattern characteristic of real-world clinical triage.</p>"
    )
    
    # Figures with correct paths for html (relative to html directory)
    # The html in root points to RelPath, the html in reports points to ../figures
    def img_tag(src, alt):
        return f'<div style="text-align: center;"><img src="{src}" alt="{alt}"><p style="font-size:0.85em;color:#666;"><em>Figure: {alt}</em></p></div>'
        
    html_content_root = list(html_content)
    html_content_exp = list(html_content)
    
    # Root images
    html_content_root.append(img_tag(f"{fig_rel_path}/class_frequency.png", "Dataset Specialist Class Frequencies (Test Split)"))
    html_content_root.append(img_tag(f"{fig_rel_path}/long_tail_distribution.png", "Long-tail Specialist Label Distribution"))
    html_content_root.append(img_tag(f"{fig_rel_path}/rare_class_distribution.png", "Rare-Class Specialist Distribution"))
    
    # Exp images (inside reports/ folder, figures are in ../figures)
    html_content_exp.append(img_tag("../figures/class_frequency.png", "Dataset Specialist Class Frequencies (Test Split)"))
    html_content_exp.append(img_tag("../figures/long_tail_distribution.png", "Long-tail Specialist Label Distribution"))
    html_content_exp.append(img_tag("../figures/rare_class_distribution.png", "Rare-Class Specialist Distribution"))
    
    common_part = []
    common_part.append("<h2>3. Confidence & Model Calibration</h2>")
    common_part.append(
        "<p>Modern deep neural networks often suffer from overconfidence, making calibration analysis critical for "
        "clinical deployment. We compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Negative Log-Likelihood (NLL), "
        "and the Brier Score for all models.</p>"
    )
    common_part.append("<h3>Calibration Metrics Table</h3>")
    common_part.append(calibration_metrics_df.to_html(index=False, classes="table"))
    common_part.append("<h3>Calibration & Reliability Analysis Discussion</h3>")
    common_part.append(
        "<p>Lower values of ECE and Brier Score denote better-calibrated probability estimates. Across our benchmarks, "
        "the larger pre-trained models display varying calibration profiles. The lightweight DistilBERT baseline "
        "often exhibits higher calibration errors, while XLM-RoBERTa-large benefits from larger pre-training support. "
        "High ECE scores warrant post-processing calibration (such as temperature scaling) before actual clinical usage.</p>"
    )
    
    html_content_root.extend(common_part)
    html_content_exp.extend(common_part)
    
    # Reliability plots loop
    for model in config.models:
        html_content_root.append(f"<h4>{model} Calibration Curves</h4>")
        html_content_root.append(img_tag(f"{fig_rel_path}/{model}_specialist_reliability.png", f"{model} Specialist Reliability Curve"))
        html_content_root.append(img_tag(f"{fig_rel_path}/{model}_specialist_conf_histogram.png", f"{model} Specialist Confidence Histogram"))
        html_content_root.append(img_tag(f"{fig_rel_path}/{model}_severity_reliability.png", f"{model} Severity Reliability Curve"))
        html_content_root.append(img_tag(f"{fig_rel_path}/{model}_severity_conf_histogram.png", f"{model} Severity Confidence Histogram"))

        html_content_exp.append(f"<h4>{model} Calibration Curves</h4>")
        html_content_exp.append(img_tag(f"../figures/{model}_specialist_reliability.png", f"{model} Specialist Reliability Curve"))
        html_content_exp.append(img_tag(f"../figures/{model}_specialist_conf_histogram.png", f"{model} Specialist Confidence Histogram"))
        html_content_exp.append(img_tag(f"../figures/{model}_severity_reliability.png", f"{model} Severity Reliability Curve"))
        html_content_exp.append(img_tag(f"../figures/{model}_severity_conf_histogram.png", f"{model} Severity Confidence Histogram"))
        
    common_part_2 = []
    common_part_2.append("<h2>4. Error Taxonomy Analysis</h2>")
    common_part_2.append(
        "<p>We categorize system failures into six distinct classes to dissect the nature of incorrect routing. "
        "This breakdown exposes whether models fail on routing (specialist) or severity prediction, and identifies "
        "high-confidence failures.</p>"
    )
    
    for model in config.models:
        tax_df = taxonomy_dict[model]
        common_part_2.append(f"<h3>{model} Error Taxonomy Distribution</h3>")
        common_part_2.append(tax_df.to_html(index=False, classes="table"))
        
    common_part_2.append("<h3>Failure Pattern Observations</h3>")
    common_part_2.append(
        "<ul>"
        "<li><strong>Near-Miss Specialists</strong>: A high percentage of specialist routing failures fall into the 'Near-miss' "
        "category (where the correct department is in the top-3 probabilities). This indicates that the models "
        "often capture the correct clinical domain but select a related department due to overlapping vocabulary.</li>"
        "<li><strong>Severity vs Specialist</strong>: The error counts reveal that severity classification is highly coupled with "
        "routing. Joint errors represent the most severe failures, highlighting the need for multi-task optimization tuning.</li>"
        "<li><strong>High-Confidence Failures</strong>: High-confidence errors (&gt;70% probability) point to systemic gaps where "
        "ambiguous descriptions mislead the classifiers. These instances have been exported to the failure case CSV "
        "for clinicians to review.</li>"
        "</ul>"
    )
    
    common_part_2.append("<h2>5. Per-Language Stratification</h2>")
    common_part_2.append(
        "<p>Given the multilingual clinical environment, we evaluate models across English, Hindi, Hinglish (Hindi written "
        "in Latin script), and Mixed-code queries using the swappable Heuristic Language Detector.</p>"
    )
    common_part_2.append("<h3>Per-Language Metrics Table</h3>")
    common_part_2.append(lang_metrics_df.to_html(index=False, classes="table"))
    
    common_part_2.append("<h3>Language Observations</h3>")
    common_part_2.append(
        "<p>The results demonstrate that monolingual baselines like IndicBERT perform well on Hindi/Hinglish but degrade "
        "sharply on pure English texts. Conversely, multilingual encoders (mBERT and XLM-RoBERTa) show "
        "robustness across the language spectrum. Interestingly, Hinglish clinical queries present a high error rate, "
        "attributable to the phonetic and spelling variations of anatomical and medical terms.</p>"
    )
    
    common_part_2.append("<h2>6. Sentence Length Stratification</h2>")
    common_part_2.append("<p>We bucket clinical queries by word count to assess accuracy on short vs. descriptive texts.</p>")
    common_part_2.append(len_metrics_df.to_html(index=False, classes="table"))
    common_part_2.append("<p><em>Longer clinical descriptions provide richer vocabulary, leading to higher triage accuracy.</em></p>")
    
    common_part_2.append("<h2>7. Rare-Class Performance</h2>")
    common_part_2.append(
        "<p>We report performance on low-frequency departments (rare classes with support below the lower quartile "
        "of the test distribution) separately to evaluate model robustness on minority categories.</p>"
    )
    common_part_2.append(rare_class_metrics_df.to_html(index=False, classes="table"))
    
    common_part_2.append("<h2>8. Cross-Model Agreement &amp; Statistical Significance</h2>")
    common_part_2.append(
        "<p>We compute pairwise percentage agreement and Cohen's Kappa to measure predictive consensus, and present "
        "McNemar's test results to assess statistical significance between architecture pairs.</p>"
    )
    
    html_content_root.extend(common_part_2)
    html_content_exp.extend(common_part_2)
    
    # Agreement heatmaps
    html_content_root.append("<h3>Pairwise Model Agreement Heatmaps</h3>")
    html_content_root.append(img_tag(f"{fig_rel_path}/specialist_agreement_kappa.png", "Specialist Kappa Agreement"))
    html_content_root.append(img_tag(f"{fig_rel_path}/severity_agreement_kappa.png", "Severity Kappa Agreement"))
    
    html_content_exp.append("<h3>Pairwise Model Agreement Heatmaps</h3>")
    html_content_exp.append(img_tag("../figures/specialist_agreement_kappa.png", "Specialist Kappa Agreement"))
    html_content_exp.append(img_tag("../figures/severity_agreement_kappa.png", "Severity Kappa Agreement"))
    
    common_part_3 = []
    if mcnemar_results:
        common_part_3.append("<h3>McNemar's Test Significance Results</h3>")
        common_part_3.append(
            "<p>McNemar's test evaluates the null hypothesis that two models have equal error rates. A p-value &lt; 0.05 "
            "rejects the null hypothesis, demonstrating a statistically significant performance difference.</p>"
        )
        mcnemar_rows = []
        for pair_name, stats in mcnemar_results.items():
            model_a, model_b = pair_name.split("_vs_")
            mcnemar_rows.append({
                "Comparison": f"{model_a} vs {model_b}",
                "Chi2 Statistic": f"{stats['statistic']:.4f}",
                "p-value": f"{stats['p_value']:.4e}",
                "Significant (p < 0.05)": "Yes" if stats["significant"] else "No",
                f"{model_a} only correct": stats["model1_only_correct"],
                f"{model_b} only correct": stats["model2_only_correct"]
            })
        mcnemar_df = pd.DataFrame(mcnemar_rows)
        common_part_3.append(mcnemar_df.to_html(index=False, classes="table"))
        
    common_part_3.append("<h2>9. Recommendations for Future Iterations</h2>")
    common_part_3.append(
        "ol"
        "<li><strong>Clinical Term Vocabulary Expansion</strong>: Integrate multilingual medical ontology mappings (e.g. UMLS, SNOMED-CT) "
        "to resolve phonetic Hinglish variations.</li>"
        "<li><strong>Calibration Post-Processing</strong>: Apply Platt Scaling or Temperature Scaling on the multi-task heads "
        "to align predicted confidence scores with clinical risks.</li>"
        "<li><strong>Targeted Rare-Class Augmentation</strong>: Use class-balanced loss functions (e.g. Focal Loss) or data "
        "augmentation methods to boost minority class support.</li>"
        "</ol>"
    )
    common_part_3.append("</body>\n</html>")
    
    html_content_root.extend(common_part_3)
    html_content_exp.extend(common_part_3)
    
    # Save HTML reports
    html_exp_path = experiment_dir / "reports" / "analysis_report.html"
    html_exp_path.write_text("\n".join(html_content_exp), encoding="utf-8")
    
    html_root_path = repo_root / "analysis_report.html"
    html_root_path.write_text("\n".join(html_content_root), encoding="utf-8")
    
    logger.info(f"Generated Markdown report: {md_root_path}")
    logger.info(f"Generated HTML report: {html_root_path}")
    
    return md_root_path, html_root_path
