import pytest
import os
import shutil
import json
from pathlib import Path

from ref.core.campaign_runner import CampaignRunner
from ref.core.experiment_runner import ExperimentRunner

class MockExperimentRunner:
    def __init__(self):
        self.calls = []
        self.should_fail = False

    def run(self, experiment_id, config, output_dir, is_smoke_test=False):
        if self.should_fail and experiment_id.startswith("fail_exp"):
            raise RuntimeError("Simulated failure")
        self.calls.append({
            "id": experiment_id,
            "config": config,
            "output_dir": output_dir,
            "is_smoke_test": is_smoke_test
        })

@pytest.fixture
def mock_campaign_config():
    return {
        "campaign_name": "test_campaign",
        "output_dir": "test_outputs",
        "seeds": [1, 2],
        "experiments": [
            {"id": "exp1"},
            {"id": "fail_exp"},
            {"id": "exp3"}
        ]
    }

@pytest.fixture
def temp_output(tmp_path):
    out = tmp_path / "test_outputs"
    yield out
    if out.exists():
        shutil.rmtree(out)

def test_campaign_scheduling_and_directories(mock_campaign_config, temp_output):
    mock_campaign_config["output_dir"] = str(temp_output)
    
    runner = MockExperimentRunner()
    campaign = CampaignRunner(experiment_runner=runner)
    campaign.execute_campaign(mock_campaign_config)
    
    assert len(runner.calls) == 6 # 3 exps * 2 seeds
    assert runner.calls[0]["id"] == "exp1_seed_1"
    assert runner.calls[0]["config"]["seed"] == 1
    assert "exp1" in str(runner.calls[0]["output_dir"]).replace("\\", "/")
    assert "seed_1" in str(runner.calls[0]["output_dir"]).replace("\\", "/")
    
    # Check that it saves the state
    state_file = temp_output / "campaign_state.json"
    assert state_file.exists()
    
    with open(state_file, "r") as f:
        state = json.load(f)
        assert len(state["completed_runs"]) == 6

def test_resume_logic(mock_campaign_config, temp_output):
    mock_campaign_config["output_dir"] = str(temp_output)
    temp_output.mkdir(parents=True)
    
    # Pre-populate state
    state = {"completed_runs": ["exp1/seed_1", "exp1/seed_2"]}
    with open(temp_output / "campaign_state.json", "w") as f:
        json.dump(state, f)
        
    runner = MockExperimentRunner()
    campaign = CampaignRunner(experiment_runner=runner)
    
    campaign.execute_campaign(mock_campaign_config, resume=True)
    
    assert len(runner.calls) == 4 # Should skip first 2
    assert runner.calls[0]["id"] == "fail_exp_seed_1"

def test_failure_recovery(mock_campaign_config, temp_output):
    mock_campaign_config["output_dir"] = str(temp_output)
    
    runner = MockExperimentRunner()
    runner.should_fail = True
    campaign = CampaignRunner(experiment_runner=runner)
    
    # Should not crash, should catch error and continue
    campaign.execute_campaign(mock_campaign_config)
    
    assert len(runner.calls) == 4 # exp1 (x2) and exp3 (x2) succeed
    
    state_file = temp_output / "campaign_state.json"
    with open(state_file, "r") as f:
        state = json.load(f)
        assert len(state["completed_runs"]) == 4
        assert "fail_exp/seed_1" not in state["completed_runs"]

def test_dry_run_mode(mock_campaign_config, temp_output):
    mock_campaign_config["output_dir"] = str(temp_output)
    
    runner = MockExperimentRunner()
    campaign = CampaignRunner(experiment_runner=runner)
    
    campaign.execute_campaign(mock_campaign_config, is_dry_run=True)
    
    assert len(runner.calls) == 0 # No actual runs
    assert not (temp_output / "campaign_state.json").exists()

def test_smoke_test_mode(mock_campaign_config, temp_output):
    mock_campaign_config["output_dir"] = str(temp_output)
    
    runner = MockExperimentRunner()
    campaign = CampaignRunner(experiment_runner=runner)
    
    campaign.execute_campaign(mock_campaign_config, is_smoke_test=True)
    
    assert len(runner.calls) == 1 # 1 exp, 1 seed
    assert runner.calls[0]["is_smoke_test"] is True
