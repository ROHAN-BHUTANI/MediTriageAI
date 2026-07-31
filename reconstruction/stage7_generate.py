"""Stage 7 – LLM Synthetic Generation.

For extreme minority classes (< augmentation_min_class_size), uses LLM
providers to generate clinically diverse synthetic samples.

Writes:
  stage7_synthetic.parquet           – dataset with synthetic samples
  stage7_generation_report.json      – generation statistics
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reconstruction.config import ReconstructionConfig
from reconstruction.llm import (
    GeneratedSample,
    get_provider,
    hash_prompt,
    list_providers,
)
# Ensure all providers are registered
import reconstruction.llm.offline_provider  # noqa: F401

try:
    import reconstruction.llm.gemini_provider  # noqa: F401
except ImportError:
    pass
try:
    import reconstruction.llm.openai_provider  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger(__name__)

STAGE_NAME = "stage7_generate"

_PROMPT_TEMPLATE = """You are generating realistic patient complaints for the {department} department.

Clinical context: {department} handles conditions such as: {examples}

Generate a single realistic patient complaint. Requirements:
- Preserve the department: {department}
- Preserve clinical intent and diagnosis consistency
- Maximize linguistic diversity (English, Hindi, Hinglish, broken English)
- Maximize symptom expression diversity
- Maximize demographic variation (age, gender, background)
- Do NOT merely paraphrase the examples
- Sound like a real patient, not a medical textbook

Symptoms: {symptoms}

Patient complaint:"""


def _extract_symptoms(texts: list[str]) -> str:
    """Extract representative symptoms from a list of sample texts."""
    # Take first few unique words that might be symptoms
    all_words = set()
    for t in texts[:10]:
        all_words.update(t.lower().split()[:10])
    return ", ".join(list(all_words)[:15])


def generate_for_class(
    class_df: pd.DataFrame,
    target_size: int,
    cfg: ReconstructionConfig,
) -> tuple[pd.DataFrame, dict]:
    """Generate synthetic samples for an extreme minority class.

    Args:
        class_df: DataFrame for one department.
        target_size: Desired count.
        cfg: Reconstruction config.

    Returns:
        Tuple of (expanded DataFrame, report dict).
    """
    dept = class_df["department"].iloc[0]
    original_size = len(class_df)
    deficit = target_size - original_size

    if deficit <= 0:
        return class_df, {"department": dept, "action": "passthrough", "generated": 0}

    # Resolve provider
    provider_name = cfg.llm_provider
    if provider_name not in list_providers():
        provider_name = "offline"
        logger.warning("Provider '%s' not registered, falling back to offline", cfg.llm_provider)

    provider = get_provider(provider_name)
    logger.info("Using LLM provider: %s", provider.name)

    source_texts = class_df["raw_text"].tolist()
    symptoms = _extract_symptoms(source_texts)
    examples = "; ".join(source_texts[:5])

    prompt = _PROMPT_TEMPLATE.format(
        department=dept,
        examples=examples,
        symptoms=symptoms,
    )
    prompt_hash = hash_prompt(prompt)

    generated_rows = []
    batch_size = min(10, deficit)
    generated_count = 0
    rejected_count = 0

    while generated_count < deficit:
        n_needed = min(batch_size, deficit - generated_count)
        try:
            texts = provider.generate(prompt, n=n_needed, seed=cfg.random_seed + generated_count)
        except Exception as e:
            logger.warning("Generation error: %s", e)
            break

        for text in texts:
            if not provider.validate(text, dept):
                rejected_count += 1
                continue

            gen_id = str(uuid.uuid4())[:12]
            source_row = class_df.iloc[generated_count % original_size].to_dict()
            row = {
                "id": f"syn_{dept}_{gen_id}",
                "split": source_row.get("split", "train"),
                "dataset_source": f"synthetic_{provider.name}",
                "language": "en",  # Provider may vary
                "raw_text": text,
                "department": dept,
                "triage_level": source_row.get("triage_level"),
                "_provenance_source_id": source_row.get("id", ""),
                "_provenance_prompt_hash": prompt_hash,
                "_provenance_provider": provider.name,
                "_provenance_generation_id": gen_id,
                "_provenance_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            generated_rows.append(row)
            generated_count += 1

            if generated_count >= deficit:
                break

    syn_df = pd.DataFrame(generated_rows) if generated_rows else pd.DataFrame()
    result = pd.concat([class_df, syn_df], ignore_index=True) if not syn_df.empty else class_df

    if len(result) > target_size:
        result = result.head(target_size)

    report = {
        "department": dept,
        "action": "synthetic_generation",
        "original_size": original_size,
        "generated": generated_count,
        "rejected": rejected_count,
        "final_size": len(result),
        "provider": provider.name,
        "prompt_hash": prompt_hash,
    }

    return result, report


def run(df: pd.DataFrame, cfg: ReconstructionConfig) -> pd.DataFrame:
    """Execute Stage 7: generate synthetic samples for extreme minority classes."""
    out_dir = Path(cfg.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / "stage7_synthetic.parquet"
    report_path = out_dir / "stage7_generation_report.json"

    if output_path.exists():
        logger.info("Stage 7 artifacts found, resuming from %s", output_path)
        return pd.read_parquet(output_path)

    target = cfg.target_class_size
    all_parts = []
    all_reports = []

    for dept in sorted(df["department"].unique()):
        dept_df = df[df["department"] == dept].copy()
        size = len(dept_df)

        if size >= target or size >= cfg.augmentation_min_class_size:
            all_parts.append(dept_df)
            all_reports.append({"department": dept, "action": "skip", "size": size})
        else:
            generated, report = generate_for_class(dept_df, target, cfg)
            all_parts.append(generated)
            all_reports.append(report)
            logger.info("  %s: %d -> %d (synthetic)", dept, size, len(generated))

    result = pd.concat(all_parts, ignore_index=True)
    result.to_parquet(output_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"departments": all_reports}, f, indent=2)

    logger.info("Stage 7 complete. %d total samples.", len(result))
    return result
