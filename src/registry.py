# src/registry.py
"""Utility to dynamically import transformation plugins based on configuration.

The registry reads ``config/enrichment_config.yaml`` which contains an ``enabled_plugins``
list.  For each plugin name it attempts to import the class from
``src.transforms.<snake_case_name>`` and instantiate it.
"""

import importlib
import yaml
from pathlib import Path
from typing import List

from .transformation_base import TransformationPlugin

CONFIG_PATH = Path(__file__).parents[1] / "config" / "enrichment_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _plugin_module_name(plugin_name: str) -> str:
    # Convert CamelCase to snake_case file name.
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r"\1_\2", plugin_name)
    snake = re.sub('([a-z0-9])([A-Z])', r"\1_\2", s1).lower()
    return f"src.transforms.{snake}"


def load_plugins() -> List[TransformationPlugin]:
    cfg = load_config()
    plugins: List[TransformationPlugin] = []
    for name in cfg.get("enabled_plugins", []):
        module_name = _plugin_module_name(name)
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, name)
            if not issubclass(cls, TransformationPlugin):
                raise TypeError(f"{name} does not inherit from TransformationPlugin")
            plugins.append(cls())
        except Exception as e:
            raise ImportError(f"Failed to load plugin '{name}' from '{module_name}': {e}")
    return plugins
