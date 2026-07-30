import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
MANIFEST_PATH = Path("paper_artifacts/manifests/figure_manifest.json")


class ManifestManager:
    """Manages the figure and table generation manifest for reproducibility."""

    def __init__(self) -> None:
        """Initializes the ManifestManager by loading any existing manifest."""
        self.entries: list[dict[str, Any]] = []
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing manifest: {e}")

    def add_entry(
        self,
        name: str,
        description: str,
        input_files: list[str],
        required_experiment: str,
        output_path: str,
        generator_script: str,
    ) -> None:
        """Add or update a generation entry in the manifest.

        Args:
            name (str): Identifier for the artifact.
            description (str): Human-readable description.
            input_files (List[str]): Files required to generate this artifact.
            required_experiment (str): The experiment process needed.
            output_path (str): The destination path of the artifact.
            generator_script (str): The script that generated the artifact.
        """
        entry = {
            "name": name,
            "description": description,
            "input_files": input_files,
            "required_experiment": required_experiment,
            "output_path": str(output_path),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "generator_script": generator_script,
        }
        self.entries = [e for e in self.entries if e["name"] != name]
        self.entries.append(entry)

    def save(self) -> None:
        """Saves the current entries to the JSON manifest file."""
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=4)
        logger.info(f"Manifest saved to {MANIFEST_PATH}")
