import pytest
import os
import shutil
import json
from pathlib import Path

from ref.provenance.types import (
    ExecutionEnvironment,
    DatasetFingerprint,
    ConfigurationFingerprint,
    CheckpointFingerprint,
    HardwareProfile,
    SoftwareProfile,
    ExperimentFingerprint,
    ProvenanceManifest,
    ProvenanceValidationError
)
from ref.provenance.base import BaseFingerprintProvider
from ref.provenance.registry import ProvenanceRegistry
from ref.provenance.pipeline import ProvenancePipeline
from ref.provenance.providers import DatasetFingerprintProvider

@pytest.fixture
def temp_output(tmp_path):
    output = tmp_path / "prov_out"
    output.mkdir()
    yield output
    if output.exists():
        shutil.rmtree(output)

class MockFingerprintProvider(BaseFingerprintProvider):
    def get_provider_name(self) -> str:
        return "MockFingerprint"
        
    def collect(self, context: dict) -> dict:
        return {"seed": context.get("seed", 42)}
        
    def fingerprint(self, data: dict) -> dict:
        return {"hashed_seed": str(data["seed"]) + "_hash"}
        
    def validate(self, fingerprinted_data: dict) -> None:
        if not fingerprinted_data["hashed_seed"]:
            raise ProvenanceValidationError("Missing hash")
            
    def serialize(self, fingerprinted_data: dict) -> dict:
        return fingerprinted_data

def test_provenance_data_structures():
    env = ExecutionEnvironment(timestamp="2026", seed=42, ref_version="1.0", repository_version="2.0")
    env.validate()
    
    with pytest.raises(ProvenanceValidationError):
        ExecutionEnvironment(timestamp="", seed=42, ref_version="1.0", repository_version="2.0").validate()

    ds = DatasetFingerprint(dataset_name="D", checksum="C", version="1")
    ds.validate()
    
    fp = ExperimentFingerprint(provider_name="Test", payload={"a": 1})
    fp.validate()
    
    man = ProvenanceManifest(experiment_id="E1", fingerprints={"Test": fp})
    man.validate()
    
    # Deterministic Check
    assert hasattr(man, "manifest_hash")
    d = man.to_dict()
    assert "manifest_hash" in d
    
    man2 = ProvenanceManifest.from_dict(d)
    assert man2.manifest_hash == man.manifest_hash

def test_provider_interface_compliance():
    provider = MockFingerprintProvider()
    fp = provider.execute_lifecycle({"seed": 99})
    
    assert isinstance(fp, ExperimentFingerprint)
    assert fp.provider_name == "MockFingerprint"
    assert fp.payload["hashed_seed"] == "99_hash"

def test_registry_integrity_and_config():
    config = {
        "provenance_engine_enabled": True,
        "enable_fingerprint_datasetfingerprint": False,
        "enable_fingerprint_mockfingerprint": True
    }
    
    registry = ProvenanceRegistry(config)
    registry.discover_providers([DatasetFingerprintProvider, MockFingerprintProvider])
    
    registry.register(["DatasetFingerprint", "MockFingerprint"])
    
    providers = registry.get_all_providers()
    assert len(providers) == 1
    assert isinstance(providers[0], MockFingerprintProvider)

def test_provenance_pipeline_and_manifest(temp_output):
    registry = ProvenanceRegistry({})
    registry.discover_providers([MockFingerprintProvider])
    registry.register(["MockFingerprint"])
    
    pipeline = ProvenancePipeline(registry, experiment_id="EXP_PROV", output_dir=str(temp_output))
    manifest = pipeline.execute({"seed": 777})
    
    assert isinstance(manifest, ProvenanceManifest)
    assert "MockFingerprint" in manifest.fingerprints
    
    # Verify provenance.json generation
    prov_file = temp_output / "provenance.json"
    assert prov_file.exists()
    
    with open(prov_file, "r") as f:
        data = json.load(f)
        assert data["experiment_id"] == "EXP_PROV"
        assert "manifest_hash" in data
        assert data["fingerprints"]["MockFingerprint"]["payload"]["hashed_seed"] == "777_hash"
