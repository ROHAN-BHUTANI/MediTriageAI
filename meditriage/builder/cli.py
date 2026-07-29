import argparse
import sys
from pathlib import Path

from .config import Config
from .orchestrator import Builder

def main():
    parser = argparse.ArgumentParser(description="MediTriageAI Dataset Builder")
    parser.add_argument("command", choices=["build", "validate", "stats", "manifest", "clean"])
    parser.add_argument("--config", type=str, default="config/dataset_config.yaml")
    parser.add_argument("--force", action="store_true")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    if args.command == "clean":
        processed_dir = base_dir / "meditriage" / "data" / "processed"
        for f in processed_dir.glob("*"):
            if f.is_file():
                f.unlink()
        print("Cleaned processed directory.")
        return
        
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
        
    if not config_path.exists():
        # Make dummy config for tests if it doesn't exist
        print(f"Config {config_path} not found. Creating default.")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            f.write('''
random_seed: 42
splits:
  train: 0.8
  val: 0.1
  test: 0.1
active_datasets:
  - mtsamples
  - pmc_patients
augmentation:
  hinglish:
    enabled_for: ["mtsamples"]
    variants_per_seed: 1
    substitution_rate: 0.5
deduplication:
  strategy: "exact_match"
  priority_order: ["pmc_patients", "mtsamples"]
            ''')
            
    config = Config.from_yaml(config_path)
    builder = Builder(config, base_dir)
    
    if args.command == "build":
        try:
            builder.build(force=args.force)
        except Exception as e:
            print(f"Build failed: {e}")
            sys.exit(1)
    elif args.command == "validate":
        print("Validation standalone not fully implemented. Run build.")
    elif args.command == "stats":
        print("Stats viewer not implemented.")
    elif args.command == "manifest":
        print("Manifest viewer not implemented.")

if __name__ == "__main__":
    main()
