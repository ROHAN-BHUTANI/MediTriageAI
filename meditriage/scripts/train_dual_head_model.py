"""
train_dual_head_model.py
---------------------------
MediTriageTransformer: dual-head XLM-RoBERTa-large sequence classifier for
joint specialist-department routing and ESI-style severity triage, with a
live Rich terminal dashboard for training visibility.

LABEL SPACES (matches the locked dataset, NOT a placeholder count):
  - Specialist routing head: 13 classes (src/specialty_mapping.py DEPARTMENTS,
    validated against all 40 raw MTSamples categories with zero unmapped
    values -- see docs/01_clinical_taxonomy.md Section 3). NOT 10 classes;
    if a 10-category schema is genuinely wanted, specialty_mapping.py must be
    redesigned and re-validated against real data first, not just resized
    here, or this head would silently train against a label space that does
    not match its own taxonomy doc.
  - Severity triage head: 5 classes (S1-S5, ESI-style; src/severity_heuristic.py).

VOCABULARY INJECTION: the injected tokens are NOT an arbitrary placeholder
list -- they are programmatically derived from the actual canonical-word ->
phonetic-variant table in src/hinglish_perturbation.py (the same table that
generated every Hinglish row in data/processed/dataset.csv). This keeps the
injected vocabulary auditable back to the one source of truth for what noisy
spellings our data actually contains, rather than guessing at markers like
"drd"/"bkar"/"shans" that don't correspond to anything the pipeline produces.

ENVIRONMENT CAVEAT (read before assuming this was run for real):
This script was written and unit-tested in a CPU-only, GPU-less container
with the huggingface.co domain blocked by network policy -- meaning the real
xlm-roberta-large pretrained weights and SentencePiece tokenizer files
CANNOT be downloaded in that environment. Every component below (model
construction, vocabulary injection, canonical-embedding copying, the joint
loss, the optimizer/scheduler, and the Rich dashboard) has been verified for
correctness using `transformers.XLMRobertaConfig` to build a small randomly-
initialized stand-in (same architecture class, tiny dimensions) instead of
downloading the real 560M-parameter checkpoint -- see
`tests/test_train_dual_head_model.py`. When run in an environment with
internet access and a GPU, set `--model-name xlm-roberta-large` (the
default) and it will download and fine-tune the real checkpoint; no code
changes are needed to go from the tiny test config to the real one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    XLMRobertaConfig,
    XLMRobertaModel,
    get_linear_schedule_with_warmup,
)

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from specialty_mapping import DEPARTMENTS  # noqa: E402

DEPARTMENT_CODES: list[str] = sorted(DEPARTMENTS.keys())
SEVERITY_LABELS: list[str] = ["S1", "S2", "S3", "S4", "S5"]

NUM_DEPARTMENT_CLASSES = len(DEPARTMENT_CODES)  # 13, not a hardcoded guess
NUM_SEVERITY_CLASSES = len(SEVERITY_LABELS)  # 5

DEPARTMENT_TO_IDX = {code: i for i, code in enumerate(DEPARTMENT_CODES)}
SEVERITY_TO_IDX = {label: i for i, label in enumerate(SEVERITY_LABELS)}


# ===========================================================================
# 1. Vocabulary injection: derived from the REAL perturbation variant table
# ===========================================================================


def load_canonical_to_variants_map() -> dict[str, list[str]]:
    """
    Programmatically derive the canonical-anchor -> noisy-variant token
    mapping from src/hinglish_perturbation.py's actual variant table, instead
    of hardcoding a guessed token list. This is the same table that produced
    every Hinglish row in data/processed/dataset.csv, so the injected
    vocabulary is guaranteed to match tokens the model will actually see in
    training data.
    """
    from hinglish_perturbation import (  # local import: only needed here
        _FINAL_H_DROP_REPLACEMENTS,
        _FINAL_H_DROP_WORDS,
        _VARIANT_TABLE,
    )

    canonical_to_variants: dict[str, list[str]] = {}
    word_boundary_pattern = re.compile(r"^\\b(.+)\\b$")

    for variant in _VARIANT_TABLE:
        match = word_boundary_pattern.match(variant.pattern.pattern)
        canonical = match.group(1) if match else variant.pattern.pattern
        canonical_to_variants.setdefault(canonical, [])
        for alt in variant.alternatives:
            if alt != canonical and alt not in canonical_to_variants[canonical]:
                canonical_to_variants[canonical].append(alt)

    for word in _FINAL_H_DROP_WORDS:
        replacement = _FINAL_H_DROP_REPLACEMENTS.get(word, word[:-1])
        canonical_to_variants.setdefault(word, [])
        if replacement != word and replacement not in canonical_to_variants[word]:
            canonical_to_variants[word].append(replacement)

    return canonical_to_variants


@dataclass
class VocabInjectionPlan:
    canonical_to_variants: dict[str, list[str]]
    new_tokens: list[str]  # flattened, de-duplicated variant tokens to add

    @property
    def n_new_tokens(self) -> int:
        return len(self.new_tokens)


def build_vocab_injection_plan() -> VocabInjectionPlan:
    canonical_to_variants = load_canonical_to_variants_map()
    seen: set[str] = set()
    new_tokens: list[str] = []
    for variants in canonical_to_variants.values():
        for token in variants:
            if token not in seen:
                seen.add(token)
                new_tokens.append(token)
    return VocabInjectionPlan(canonical_to_variants=canonical_to_variants, new_tokens=new_tokens)


def inject_vocabulary_and_init_embeddings(
    model: XLMRobertaModel,
    tokenizer: PreTrainedTokenizerBase,
    plan: VocabInjectionPlan,
    console: Console | None = None,
) -> int:
    """
    Add `plan.new_tokens` to the tokenizer, resize the model's token
    embeddings to match, and explicitly copy each new token's embedding
    weight from its canonical anchor word's embedding:

        W[t_new] = W[t_anchor]

    This shields early-epoch gradients from the destabilization that comes
    from a brand-new randomly-initialized embedding row producing
    arbitrary-scale activations alongside a pretrained backbone -- the new
    token starts out numerically and semantically equivalent to a word the
    model already understands, then specializes from there as training
    proceeds.

    IMPORTANT: we pass mean_resizing=False to resize_token_embeddings.
    transformers' default mean-based resizing would otherwise overwrite our
    explicit canonical-copy initialization immediately after we set it (or,
    depending on call order, our copy would be the one that wins -- either
    way, relying on the library default here would make this code's actual
    initialization behavior implicit and version-dependent rather than
    explicit, which is the opposite of what canonical-anchor init is for).

    Returns the number of new tokens actually added (tokens already present
    in the vocabulary are skipped, not double-added).

    IMPLEMENTATION NOTE on a real discrepancy found during testing:
    tokenizer.add_tokens() return value cannot be trusted as the true count
    of vocabulary growth. It counts how many of the requested strings were
    not already registered in the tokenizer's *added-tokens* table, but it
    does NOT check whether a string already exists as an ordinary piece in
    the *base* SentencePiece vocabulary -- if it does, add_tokens "adds" it
    anyway (incrementing its own counter) without actually growing
    len(tokenizer), because the string already maps to an existing token ID.
    Concretely: with our tiny test vocabulary, "he"/"hy"/"ap" were already
    ordinary subword pieces, so add_tokens reported n_added=53 while
    len(tokenizer) only grew by 50. We therefore compute the returned count
    from the actual length delta, not the library's return value, so callers
    (e.g. total_steps/warmup calculations, or anything sizing a new
    embedding table) get a number that matches reality.
    """
    vocab_size_before = len(tokenizer)

    # CRITICAL ORDERING REQUIREMENT, found via testing: anchor vectors MUST
    # be computed BEFORE any new tokens are added to the tokenizer. Adding a
    # variant token can itself change how its own canonical word re-
    # tokenizes -- e.g. "aapka" tokenizes as ['_aap','ka'] before injection,
    # but once "apka" is added as its own vocabulary entry, greedy BPE may
    # re-segment "aapka" differently (picking up the new "apka" piece),
    # silently changing what "the anchor vector" even refers to mid-loop.
    # Computing every anchor's embedding snapshot up front, against the
    # ORIGINAL pre-injection vocabulary, makes "W[t_new] = W[t_anchor]"
    # well-defined and independent of dict/insertion order.
    embedding_layer = model.get_input_embeddings()
    anchor_vectors: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for canonical_word in plan.canonical_to_variants:
            anchor_ids = tokenizer.encode(canonical_word, add_special_tokens=False)
            if len(anchor_ids) != 1:
                # Canonical anchor isn't a single subword token in this
                # tokenizer's vocabulary -- fall back to averaging its
                # subword embeddings as the anchor vector, rather than
                # silently skipping the copy or crashing.
                anchor_vectors[canonical_word] = (
                    embedding_layer.weight[anchor_ids].mean(dim=0).clone()
                )
            else:
                anchor_vectors[canonical_word] = embedding_layer.weight[anchor_ids[0]].clone()

    tokenizer.add_tokens(plan.new_tokens)
    n_added = len(tokenizer) - vocab_size_before

    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    embedding_layer = model.get_input_embeddings()  # re-fetch: resize may reallocate
    already_initialized: set[int] = set()  # token ids claimed by an earlier anchor this call
    with torch.no_grad():
        for canonical_word, variants in plan.canonical_to_variants.items():
            anchor_vector = anchor_vectors[canonical_word]

            for variant_token in variants:
                variant_id = tokenizer.convert_tokens_to_ids(variant_token)
                if variant_id is None or variant_id == tokenizer.unk_token_id:
                    continue  # token wasn't actually added (e.g. duplicate); skip
                if variant_id < vocab_size_before:
                    # IMPORTANT: this variant_token string already existed as
                    # an ordinary token in the BASE vocabulary before this
                    # injection call (see the n_added discrepancy noted
                    # above -- e.g. "ap" is both a real pretrained subword
                    # piece AND happens to be listed as a phonetic variant of
                    # "aap"). Canonical-anchor initialization is only meant
                    # to shield genuinely NEW, randomly-initialized embedding
                    # rows from gradient destabilization -- overwriting a
                    # pre-existing token's already-meaningful (in the real
                    # XLM-R checkpoint, pretrained) embedding with an
                    # unrelated anchor's vector would destroy real signal,
                    # not protect anything. Skip it.
                    continue
                if variant_id in already_initialized:
                    # MANY-TO-ONE COLLISION, found via testing: a few variant
                    # tokens are listed under more than one canonical anchor
                    # in hinglish_perturbation.py's table -- e.g. "nai" is a
                    # documented variant of BOTH "nahi" and "nahin" (which
                    # are themselves mutual variants of each other). Without
                    # this guard, whichever anchor happens to be processed
                    # last in dict-iteration order would silently overwrite
                    # the first anchor's initialization, making the result
                    # depend on dict insertion order rather than being a
                    # well-defined function of the variant table. We instead
                    # make "first anchor encountered wins" an explicit,
                    # deterministic rule (dict iteration order is stable and
                    # insertion-ordered in Python, so this is reproducible).
                    continue
                embedding_layer.weight[variant_id] = anchor_vector.clone()
                already_initialized.add(variant_id)

    if console is not None:
        console.print(
            f"[green]Vocabulary injection complete:[/green] "
            f"{n_added} new tokens added, embeddings initialized from "
            f"{len(plan.canonical_to_variants)} canonical anchors "
            f"(source: src/hinglish_perturbation.py variant table)."
        )
    return n_added


# ===========================================================================
# 2. Dual-head model architecture
# ===========================================================================


class MediTriageTransformer(nn.Module):
    """
    Dual-head sequence classifier on top of an XLM-RoBERTa encoder.

    Two independent linear heads branch off the pooled [CLS] (first-token)
    representation of the final hidden state:
      - specialist_head: projects to NUM_DEPARTMENT_CLASSES (13)
      - severity_head:   projects to NUM_SEVERITY_CLASSES (5)

    The heads are intentionally simple single linear layers (with dropout)
    rather than deeper MLPs -- this matches the "decoupled architecture"
    design referenced in the project's prior work, where task-specific
    capacity is kept minimal and the shared encoder does the representation
    learning (cf. MT-Clinical BERT's per-task linear heads on a shared
    encoder, PMC8449623, which we cited when designing the joint loss).
    """

    def __init__(self, encoder: XLMRobertaModel, hidden_dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.specialist_head = nn.Linear(hidden_size, NUM_DEPARTMENT_CLASSES)
        self.severity_head = nn.Linear(hidden_size, NUM_SEVERITY_CLASSES)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS-token pooled representation: first token of the last hidden state.
        cls_representation = encoder_output.last_hidden_state[:, 0, :]
        cls_representation = self.dropout(cls_representation)

        specialist_logits = self.specialist_head(cls_representation)
        severity_logits = self.severity_head(cls_representation)
        return specialist_logits, severity_logits


# ===========================================================================
# 3. Weighted joint loss
# ===========================================================================


@dataclass
class JointLossWeights:
    alpha_specialist: float = 1.0
    beta_severity: float = 1.2  # prioritizes severity-triage stability


class JointLoss(nn.Module):
    """
    L_joint = alpha * L_specialist + beta * L_severity

    Both task losses are standard CrossEntropyLoss over their respective
    logits/label pairs. Returns the combined scalar loss plus the two
    component losses (detached, for logging) so the dashboard can show
    per-task loss curves without re-computing anything.
    """

    def __init__(self, weights: JointLossWeights) -> None:
        super().__init__()
        self.weights = weights
        self.specialist_loss_fn = nn.CrossEntropyLoss()
        self.severity_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        specialist_logits: torch.Tensor,
        specialist_labels: torch.Tensor,
        severity_logits: torch.Tensor,
        severity_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        specialist_loss = self.specialist_loss_fn(specialist_logits, specialist_labels)
        severity_loss = self.severity_loss_fn(severity_logits, severity_labels)
        joint_loss = (
            self.weights.alpha_specialist * specialist_loss
            + self.weights.beta_severity * severity_loss
        )
        return {
            "joint_loss": joint_loss,
            "specialist_loss": specialist_loss.detach(),
            "severity_loss": severity_loss.detach(),
        }


# ===========================================================================
# 4. Dataset
# ===========================================================================


class MediTriageDataset(Dataset):
    """
    Wraps data/processed/dataset.csv rows for a given split. Tokenizes lazily
    (per __getitem__) rather than pre-tokenizing the whole split into memory,
    since the full dataset has ~16-20k rows per split and lazy tokenization
    keeps memory bounded regardless of corpus size.
    """

    def __init__(
        self,
        rows: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "specialist_label": torch.tensor(
                DEPARTMENT_TO_IDX[row["department_code"]], dtype=torch.long
            ),
            "severity_label": torch.tensor(
                SEVERITY_TO_IDX[row["severity_heuristic"]], dtype=torch.long
            ),
        }


def load_split_rows(dataset_csv: Path, split: str) -> list[dict]:
    """
    Load rows for a given split from the processed dataset CSV.

    Defensively coerces the `text` column to string and DROPS rows where the
    underlying text is empty/NaN, logging how many were dropped. This is a
    real, non-hypothetical edge case: 33 MTSamples source rows have a
    genuinely missing `transcription` field (see docs/01_clinical_taxonomy.md
    and the original data-layer validation), and a handful of those land in
    each split. A NaN/empty text field cannot be tokenized or trained on
    meaningfully, so we exclude those specific rows here rather than crash
    mid-epoch or silently feed the tokenizer a string "nan".
    """
    import pandas as pd

    df = pd.read_csv(dataset_csv)
    if split not in set(df["split"].unique()):
        raise ValueError(
            f"Split {split!r} not found in {dataset_csv}. "
            f"Available splits: {sorted(df['split'].unique())}"
        )
    split_df = df[df["split"] == split].copy()

    is_valid_text = split_df["text"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)
    n_dropped = int((~is_valid_text).sum())
    if n_dropped > 0:
        print(
            f"[load_split_rows] Dropping {n_dropped} row(s) from split={split!r} "
            f"with missing/empty text (traces back to MTSamples rows with no "
            f"transcription -- see docs/01_clinical_taxonomy.md). "
            f"Remaining: {int(is_valid_text.sum())} rows.",
            file=sys.stderr,
        )
    split_df = split_df[is_valid_text]

    return split_df.to_dict(orient="records")


# ===========================================================================
# 5. Rich live dashboard
# ===========================================================================


@dataclass
class RunningMetrics:
    """Exponentially-weighted running metrics for the live dashboard, reset
    at the start of each epoch."""

    specialist_loss_sum: float = 0.0
    severity_loss_sum: float = 0.0
    specialist_correct: int = 0
    severity_correct: int = 0
    n_examples: int = 0
    n_batches: int = 0

    def update(
        self,
        specialist_loss: float,
        severity_loss: float,
        specialist_logits: torch.Tensor,
        severity_logits: torch.Tensor,
        specialist_labels: torch.Tensor,
        severity_labels: torch.Tensor,
    ) -> None:
        batch_size = specialist_labels.size(0)
        self.specialist_loss_sum += specialist_loss * batch_size
        self.severity_loss_sum += severity_loss * batch_size
        self.specialist_correct += (
            (specialist_logits.argmax(dim=-1) == specialist_labels).sum().item()
        )
        self.severity_correct += (
            (severity_logits.argmax(dim=-1) == severity_labels).sum().item()
        )
        self.n_examples += batch_size
        self.n_batches += 1

    @property
    def specialist_avg_loss(self) -> float:
        return self.specialist_loss_sum / self.n_examples if self.n_examples else 0.0

    @property
    def severity_avg_loss(self) -> float:
        return self.severity_loss_sum / self.n_examples if self.n_examples else 0.0

    @property
    def specialist_acc(self) -> float:
        return self.specialist_correct / self.n_examples if self.n_examples else 0.0

    @property
    def severity_acc(self) -> float:
        return self.severity_correct / self.n_examples if self.n_examples else 0.0


def build_metrics_table(metrics: RunningMetrics, epoch: int, lr: float) -> Table:
    table = Table(title=f"Epoch {epoch} -- Live Training Metrics", expand=True)
    table.add_column("Task", style="bold cyan")
    table.add_column("Loss", justify="right")
    table.add_column("Train Acc", justify="right")

    table.add_row(
        "Specialist Routing",
        f"{metrics.specialist_avg_loss:.4f}",
        f"{metrics.specialist_acc:.2%}",
    )
    table.add_row(
        "Severity Triage",
        f"{metrics.severity_avg_loss:.4f}",
        f"{metrics.severity_acc:.2%}",
    )
    table.caption = f"Learning rate: {lr:.2e}  |  Batches seen this epoch: {metrics.n_batches}"
    return table


def build_validation_summary_table(
    epoch: int,
    val_metrics: RunningMetrics,
    elapsed_seconds: float,
) -> Table:
    table = Table(title=f"Epoch {epoch} -- Validation Summary", expand=True)
    table.add_column("Metric", style="bold magenta")
    table.add_column("Specialist Routing", justify="right")
    table.add_column("Severity Triage", justify="right")

    table.add_row(
        "Validation Loss",
        f"{val_metrics.specialist_avg_loss:.4f}",
        f"{val_metrics.severity_avg_loss:.4f}",
    )
    table.add_row(
        "Validation Accuracy",
        f"{val_metrics.specialist_acc:.2%}",
        f"{val_metrics.severity_acc:.2%}",
    )
    table.caption = f"Epoch wall-clock time: {elapsed_seconds:.1f}s"
    return table


# ===========================================================================
# 6. Training orchestration
# ===========================================================================


@dataclass
class TrainingConfig:
    model_name: str = "xlm-roberta-large"
    dataset_csv: Path = REPO_ROOT / "data" / "processed" / "dataset.csv"
    output_dir: Path = REPO_ROOT / "models" / "meditriage_dual_head"
    max_length: int = 256
    batch_size: int = 16
    n_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.10
    loss_weights: JointLossWeights = field(default_factory=JointLossWeights)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_train_batches_per_epoch: int | None = None  # for smoke-testing only


def build_model_and_tokenizer(
    config: TrainingConfig, console: Console
) -> tuple[MediTriageTransformer, PreTrainedTokenizerBase]:
    console.print(f"[bold]Loading base encoder:[/bold] {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    encoder = XLMRobertaModel.from_pretrained(config.model_name)

    plan = build_vocab_injection_plan()
    console.print(
        f"[bold]Vocabulary injection plan:[/bold] {plan.n_new_tokens} variant "
        f"tokens derived from {len(plan.canonical_to_variants)} canonical "
        f"anchors in src/hinglish_perturbation.py"
    )
    inject_vocabulary_and_init_embeddings(encoder, tokenizer, plan, console=console)

    model = MediTriageTransformer(encoder)
    model.to(config.device)
    return model, tokenizer


def run_training(config: TrainingConfig) -> None:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]MediTriageTransformer[/bold]\n"
            f"Dual-head XLM-R training -- {NUM_DEPARTMENT_CLASSES} specialist "
            f"classes x {NUM_SEVERITY_CLASSES} severity classes",
            border_style="blue",
        )
    )

    model, tokenizer = build_model_and_tokenizer(config, console)

    train_rows = load_split_rows(config.dataset_csv, "train")
    val_rows = load_split_rows(config.dataset_csv, "val")
    console.print(f"Loaded {len(train_rows)} train rows, {len(val_rows)} val rows.")

    train_dataset = MediTriageDataset(train_rows, tokenizer, config.max_length)
    val_dataset = MediTriageDataset(val_rows, tokenizer, config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    n_train_batches = (
        min(len(train_loader), config.max_train_batches_per_epoch)
        if config.max_train_batches_per_epoch
        else len(train_loader)
    )
    total_steps = n_train_batches * config.n_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    joint_loss_fn = JointLoss(config.loss_weights)

    progress_columns = [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]

    for epoch in range(1, config.n_epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        running = RunningMetrics()

        with Progress(*progress_columns, console=console) as progress:
            epoch_task = progress.add_task(
                f"Epoch {epoch}/{config.n_epochs} [train]", total=n_train_batches
            )

            with Live(
                build_metrics_table(running, epoch, optimizer.param_groups[0]["lr"]),
                console=console,
                refresh_per_second=4,
                transient=False,
            ) as live:
                for batch_idx, batch in enumerate(train_loader):
                    if (
                        config.max_train_batches_per_epoch
                        and batch_idx >= config.max_train_batches_per_epoch
                    ):
                        break

                    input_ids = batch["input_ids"].to(config.device)
                    attention_mask = batch["attention_mask"].to(config.device)
                    specialist_labels = batch["specialist_label"].to(config.device)
                    severity_labels = batch["severity_label"].to(config.device)

                    specialist_logits, severity_logits = model(input_ids, attention_mask)
                    loss_dict = joint_loss_fn(
                        specialist_logits, specialist_labels, severity_logits, severity_labels
                    )

                    optimizer.zero_grad()
                    loss_dict["joint_loss"].backward()
                    optimizer.step()
                    scheduler.step()

                    running.update(
                        specialist_loss=loss_dict["specialist_loss"].item(),
                        severity_loss=loss_dict["severity_loss"].item(),
                        specialist_logits=specialist_logits.detach(),
                        severity_logits=severity_logits.detach(),
                        specialist_labels=specialist_labels,
                        severity_labels=severity_labels,
                    )

                    progress.update(epoch_task, advance=1)
                    live.update(
                        build_metrics_table(running, epoch, optimizer.param_groups[0]["lr"])
                    )

        val_metrics = run_validation(model, val_loader, joint_loss_fn, config, console)
        epoch_elapsed = time.monotonic() - epoch_start
        console.print(build_validation_summary_table(epoch, val_metrics, epoch_elapsed))

    save_checkpoint(model, tokenizer, config, console)


@torch.no_grad()
def run_validation(
    model: MediTriageTransformer,
    val_loader: DataLoader,
    joint_loss_fn: JointLoss,
    config: TrainingConfig,
    console: Console,
) -> RunningMetrics:
    model.eval()
    running = RunningMetrics()
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Validation"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        val_task = progress.add_task("Validation", total=len(val_loader))
        for batch in val_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            specialist_labels = batch["specialist_label"].to(config.device)
            severity_labels = batch["severity_label"].to(config.device)

            specialist_logits, severity_logits = model(input_ids, attention_mask)
            loss_dict = joint_loss_fn(
                specialist_logits, specialist_labels, severity_logits, severity_labels
            )
            running.update(
                specialist_loss=loss_dict["specialist_loss"].item(),
                severity_loss=loss_dict["severity_loss"].item(),
                specialist_logits=specialist_logits,
                severity_logits=severity_logits,
                specialist_labels=specialist_labels,
                severity_labels=severity_labels,
            )
            progress.update(val_task, advance=1)
    return running


def save_checkpoint(
    model: MediTriageTransformer,
    tokenizer: PreTrainedTokenizerBase,
    config: TrainingConfig,
    console: Console,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.output_dir / "model_state_dict.pt")
    tokenizer.save_pretrained(config.output_dir)
    with (config.output_dir / "label_spaces.json").open("w") as f:
        json.dump(
            {
                "department_codes": DEPARTMENT_CODES,
                "severity_labels": SEVERITY_LABELS,
                "loss_weights": {
                    "alpha_specialist": config.loss_weights.alpha_specialist,
                    "beta_severity": config.loss_weights.beta_severity,
                },
            },
            f,
            indent=2,
        )
    console.print(f"[bold green]Checkpoint saved to {config.output_dir}[/bold green]")


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train MediTriageTransformer dual-head model.")
    parser.add_argument("--model-name", default="xlm-roberta-large")
    parser.add_argument("--dataset-csv", type=Path, default=TrainingConfig.dataset_csv)
    parser.add_argument("--output-dir", type=Path, default=TrainingConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--n-epochs", type=int, default=TrainingConfig.n_epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--alpha-specialist", type=float, default=1.0)
    parser.add_argument("--beta-severity", type=float, default=1.2)
    parser.add_argument(
        "--max-train-batches-per-epoch",
        type=int,
        default=None,
        help="Cap batches/epoch for smoke testing; omit for full training.",
    )
    args = parser.parse_args()
    return TrainingConfig(
        model_name=args.model_name,
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        loss_weights=JointLossWeights(
            alpha_specialist=args.alpha_specialist, beta_severity=args.beta_severity
        ),
        max_train_batches_per_epoch=args.max_train_batches_per_epoch,
    )


if __name__ == "__main__":
    run_training(parse_args())
