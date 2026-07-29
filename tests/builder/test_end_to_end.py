from pathlib import Path
from meditriage.builder.config import Config
from meditriage.builder.orchestrator import Builder

def test_builder_end_to_end(tmp_path):
    config_dict = {
        'random_seed': 42,
        'splits': {'train': 0.8, 'val': 0.1, 'test': 0.1},
        'active_datasets': ['mtsamples'],
        'augmentation': {
            'hinglish': {
                'enabled_for': ['mtsamples'],
                'variants_per_seed': 1,
                'substitution_rate': 0.5
            }
        },
        'deduplication': {
            'strategy': 'exact_match',
            'priority_order': ['mtsamples']
        }
    }
    
    config = Config(config_dict, raw_yaml="")
    # Use real base_dir to find raw datasets
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # We will override out_dir to be tmp_path
    builder = Builder(config, base_dir)
    builder.out_dir = tmp_path
    builder.processed_dir = tmp_path / "processed"
    builder.build_dir = tmp_path / "build_temp"
    builder.processed_dir.mkdir(parents=True, exist_ok=True)
    builder.build_dir.mkdir(parents=True, exist_ok=True)
    
    builder.build(force=True)
    
    assert (tmp_path / "processed" / "dataset.csv").exists()
    assert (tmp_path / "processed" / "build_manifest.json").exists()
    assert (tmp_path / "processed" / "dataset_statistics.json").exists()
