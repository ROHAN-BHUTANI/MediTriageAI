"""Single-text inference CLI for the MediTriageAI demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.distilbert_multi import DistilBertMultilingualModel
from models.indic_bert import IndicBertModel
from models.mbert import MBertModel
from models.xlm_roberta import XLMRobertaLargeModel

MODEL_MAP = {
    "xlm_roberta": XLMRobertaLargeModel,
    "mbert": MBertModel,
    "distilbert": DistilBertMultilingualModel,
    "indicbert": IndicBertModel,
}

SPECIALIST_LABELS = [
    "CARDIO_PULM",
    "ED",
    "ENT_OPHTHALMO",
    "GEN_MED",
    "GI",
    "NEURO",
    "OBGYN",
    "ONCOLOGY_HEME",
    "ORTHO",
    "PEDS",
    "PSYCH",
    "RENAL_URO",
    "SURGERY",
]

SEVERITY_LABELS = ["S1 URGENT", "S2 EMERGENT", "S3 URGENT", "S4", "S5"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-text MediTriageAI inference.")
    parser.add_argument("--model", required=True, choices=MODEL_MAP.keys())
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format: text (default) or JSON",
    )
    return parser


def format_topk(scores: torch.Tensor, labels: list[str], top_k: int = 3) -> list[tuple[str, float]]:
    values, indices = torch.topk(scores, k=min(top_k, scores.numel()))
    return [(labels[idx], float(value)) for value, idx in zip(values.tolist(), indices.tolist())]


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path | None) -> None:
    if checkpoint is None or not checkpoint.exists():
        return
    candidate_files = [checkpoint]
    if checkpoint.is_dir():
        candidate_files = [
            checkpoint / "pytorch_model.bin",
            checkpoint / "model.pt",
            checkpoint / "model.bin",
        ]
    for candidate in candidate_files:
        if candidate.exists() and candidate.suffix in {".bin", ".pt"}:
            state_dict = torch.load(candidate, map_location="cpu")
            if isinstance(state_dict, dict):
                model.load_state_dict(state_dict, strict=False)
            return


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    console = Console()
    model = MODEL_MAP[args.model]()
    tokenizer = model.get_tokenizer()
    built_model = model.build(None)
    _load_checkpoint(built_model, args.checkpoint)

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        specialist_logits, severity_logits = built_model(inputs["input_ids"], inputs["attention_mask"])
    specialist_probs = torch.softmax(specialist_logits[0], dim=-1)
    severity_probs = torch.softmax(severity_logits[0], dim=-1)
    specialist_top3 = format_topk(specialist_probs, SPECIALIST_LABELS)
    severity_top2 = format_topk(severity_probs, SEVERITY_LABELS, top_k=2)
    novel = " * Novel" if getattr(model, "is_novel_contribution", False) else ""
    text_preview = args.text[:40] + ("..." if len(args.text) > 40 else "")

    if args.output_format == "json":
        result = {
            "input": args.text,
            "model": model.display_name,
            "is_novel_contribution": getattr(model, "is_novel_contribution", False),
            "specialist_routing": [
                {"label": label, "confidence": score} for label, score in specialist_top3
            ],
            "severity_triage": [
                {"label": label, "confidence": score} for label, score in severity_top2
            ],
        }
        print(json.dumps(result, indent=2))
        return

    # Text output: show the panel
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "MediTriageAI - Clinical Triage Inference",
                    f"Input:  {text_preview}",
                    f"Model:  {model.display_name}{novel}",
                    "",
                    "SPECIALIST ROUTING",
                    *[f"  -> {label:<15} (confidence: {score:.3f})" for label, score in specialist_top3],
                    "",
                    "SEVERITY TRIAGE",
                    *[f"  -> {label:<15} (confidence: {score:.3f})" for label, score in severity_top2],
                    "",
                    "⚠  RESEARCH PROTOTYPE — NOT clinically validated.",
                    "     Do NOT use for real triage decisions.",
                    "     Labels are regex-heuristic derived (low confidence).",
                ]
            ),
            border_style="blue",
        )
    )


if __name__ == "__main__":
    main()