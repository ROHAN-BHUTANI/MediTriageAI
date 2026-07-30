from pathlib import Path

import yaml

from src.dataset_adapters import (
    ChatDoctorAdapter,
    DatasetAdapter,
    L3CubeAdapter,
    NHAMCSAdapter,
    PMCPatientsAdapter,
)

CONFIG_PATH = Path(__file__).parents[1] / "config" / "dataset_config.yaml"

DATASET_ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "nhamcs_ed": NHAMCSAdapter,
    "chatdoctor_healthcaremagic": ChatDoctorAdapter,
    "chatdoctor_icliniq": ChatDoctorAdapter,
    "l3cube_code_mixed": L3CubeAdapter,
    "pmc_patients": PMCPatientsAdapter,
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_enabled_adapters() -> list[DatasetAdapter]:
    """Instantiate and return all enabled dataset adapters according to the config."""
    config = load_config()
    adapters = []

    for name, adapter_cls in DATASET_ADAPTER_REGISTRY.items():
        ds_config = config.get(name, {})
        if ds_config.get("enabled", False):
            path = ds_config.get("path", f"datasets/raw/{name}")

            # For adapters that take a dataset_name argument (e.g., ChatDoctor/L3Cube)
            if adapter_cls in [ChatDoctorAdapter, L3CubeAdapter]:
                adapters.append(adapter_cls(path, dataset_name=name))
            else:
                adapters.append(adapter_cls(path))

    return adapters
