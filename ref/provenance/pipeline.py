"""
Execution Pipeline for the REF Provenance Framework.

Orchestrates the global execution across all registered provenance fingerprint providers.
Collection -> Provider Dispatch -> Fingerprint Generation -> Validation -> Aggregation -> Serialization -> Manifest Generation.
"""

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from ref.provenance.registry import ProvenanceRegistry
from ref.provenance.types import ExperimentFingerprint, ProvenanceManifest

logger = logging.getLogger(__name__)


class ProvenancePipeline:
    """
    Executes the provenance generation pipeline strictly mapping raw state context
    into final validated, serialized ProvenanceManifests.
    """

    def __init__(
        self, registry: ProvenanceRegistry, experiment_id: str, output_dir: str
    ):
        self.registry = registry
        self.experiment_id = experiment_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State tracking through pipeline
        self.context: dict[str, Any] = {}
        self.fingerprints: dict[str, ExperimentFingerprint] = OrderedDict()
        self.final_manifest: ProvenanceManifest | None = None

    def execute(self, context: dict[str, Any]) -> ProvenanceManifest:
        """Execute the strict provenance pipeline."""

        logger.debug(f"Provenance Pipeline started for EXP: {self.experiment_id}")

        # Stage 1: Collection
        self.collection_stage(context)

        # Stages 2-6: Provider Dispatch, Fingerprint, Validate, Serialize, Aggregate
        self.dispatch_and_aggregate_stage()

        # Stage 7: Manifest Generation (and JSON export)
        return self.manifest_generation_stage()

    def collection_stage(self, context: dict[str, Any]) -> None:
        """Stage 1: Ingest overarching environment context payload."""
        logger.debug("Provenance Pipeline: Collection Stage")
        self.context = context

    def dispatch_and_aggregate_stage(self) -> None:
        """
        Stages 2-6: Dispatch to providers, execute lifecycle
        (Fingerprint->Validate->Serialize), and Aggregate into ExperimentFingerprints.
        """
        logger.debug("Provenance Pipeline: Dispatch & Aggregate Stage")

        providers = self.registry.get_all_providers()

        for provider in providers:
            name = provider.get_provider_name()
            logger.debug(f"Dispatching to provenance provider: {name}")

            fingerprint = provider.execute_lifecycle(self.context)
            self.fingerprints[name] = fingerprint

    def manifest_generation_stage(self) -> ProvenanceManifest:
        """Stage 7: Manifest wrapping all fingerprints and dumping provenance.json."""
        logger.debug("Provenance Pipeline: Manifest Generation Stage")

        self.final_manifest = ProvenanceManifest(
            experiment_id=self.experiment_id, fingerprints=self.fingerprints
        )
        self.final_manifest.validate()

        # Dump provenance.json automatically (Stage 6 requirement)
        manifest_path = self.output_dir / "provenance.json"

        with open(manifest_path, "w", encoding="utf-8") as f:
            # We dump the deterministic representation
            json.dump(self.final_manifest.to_dict(), f, indent=4, sort_keys=True)

        logger.info(f"Provenance manifest written to: {manifest_path}")
        return self.final_manifest
