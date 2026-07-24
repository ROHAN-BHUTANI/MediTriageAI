"""
Campaign Runner for the Research Experiment Framework (REF).

Responsible for scheduling, configuration dispatch, seed iteration, 
state tracking, resume, and failure recovery.
"""

import json
import logging
import traceback
from pathlib import Path
from typing import Any

from ref.core.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)

class CampaignRunner:
    """
    Orchestrates an entire experimental campaign across multiple seeds and configs.
    Maintains state to support pausing, resuming, and failure recovery.
    """

    def __init__(self, experiment_runner: ExperimentRunner):
        self.experiment_runner = experiment_runner
        self._state: dict[str, Any] = {"completed_runs": []}
        self.state_file: Path | None = None

    def execute_campaign(
        self, 
        config: dict[str, Any], 
        is_smoke_test: bool = False, 
        is_dry_run: bool = False,
        resume: bool = False
    ) -> None:
        """Executes the campaign schedule."""
        campaign_name = config.get("campaign_name", "default_campaign")
        output_root = Path(config.get("output_dir", "outputs/"))
        output_root.mkdir(parents=True, exist_ok=True)
        
        self.state_file = output_root / "campaign_state.json"
        
        if resume and self.state_file.exists():
            with open(self.state_file, "r") as f:
                self._state = json.load(f)
            logger.info(f"Resuming campaign from {self.state_file}. {len(self._state['completed_runs'])} completed runs found.")
        else:
            self._state = {"completed_runs": []}
            if not is_dry_run:
                self._save_state()

        experiments = config.get("experiments", [])
        seeds = config.get("seeds", [42])
        
        if is_smoke_test:
            logger.info("SMOKE TEST MODE: Limiting to 1 experiment and 1 seed.")
            experiments = experiments[:1]
            seeds = seeds[:1]
            
        total_runs = len(experiments) * len(seeds)
        logger.info(f"Campaign '{campaign_name}' scheduled for {total_runs} total runs.")
        
        if is_dry_run:
            logger.info("DRY RUN MODE: Displaying execution plan only.")
            for exp in experiments:
                for seed in seeds:
                    run_key = f"{exp['id']}_seed{seed}"
                    target_dir = output_root / run_key
                    logger.info(f" [PLAN] Would run: {run_key} -> {target_dir}")
            return

        run_idx = 0
        for exp in experiments:
            for seed in seeds:
                run_idx += 1
                run_key = f"{exp['id']}/seed_{seed}"
                
                if run_key in self._state["completed_runs"]:
                    logger.info(f"[{run_idx}/{total_runs}] Skipping {run_key} (already completed).")
                    continue
                    
                target_dir = output_root / exp['id'] / f"seed_{seed}"
                target_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"[{run_idx}/{total_runs}] Starting {run_key}...")
                
                # Merge seed into exp config
                run_config = exp.copy()
                run_config["seed"] = seed
                
                try:
                    # Pass the unique run_key so the Registry generates a unique experiment ID
                    self.experiment_runner.run(
                        experiment_id=run_key.replace("/", "_"),
                        config=run_config,
                        output_dir=target_dir,
                        is_smoke_test=is_smoke_test,
                        resume=resume
                    )
                    
                    self._state["completed_runs"].append(run_key)
                    self._save_state()
                    logger.info(f"[{run_idx}/{total_runs}] Successfully completed {run_key}.")
                    
                except Exception as e:
                    logger.error(f"[{run_idx}/{total_runs}] FAILURE in {run_key}: {e}")
                    logger.error(traceback.format_exc())
                    logger.warning("Continuing to next scheduled run to prevent campaign collapse.")

    def _save_state(self) -> None:
        """Safely flushes internal state to disk."""
        if self.state_file:
            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=4)
