import os
import torch
import warnings
from transformers import XLMRobertaConfig, XLMRobertaModel
from torch.utils.data import DataLoader
from src.dataset import MediTriageDataset
from src.model import MediTriageTransformer, JointLoss
from src.config_manager import TrainingConfig
from src.evaluation import EvaluationExporter, generate_training_report
from src.calibration import Calibrator
from src.explainability import ExplainabilityRegistry

warnings.filterwarnings("ignore")

def main():
    print("Starting Phase 4 Smoke Test...")
    
    # 1. Dummy config and data
    class DummyConfig:
        def __init__(self):
            self.optimizer = "adamw"
            self.scheduler = "cosine"
            self.batch_size = 2
            self.epochs = 1
            self.checkpoint_dir = "./smoke_test_results"
    config = DummyConfig()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    dummy_rows = [
        {"id": "1", "split": "val", "dataset_source": "amco", "language": "en", 
         "text": "text A", "label_specialist_id": 0, "label_severity_id": 0},
        {"id": "2", "split": "val", "dataset_source": "aces", "language": "en", 
         "text": "text B", "label_specialist_id": 1, "label_severity_id": 1}
    ]
    
    # Mock Tokenizer
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.zeros(1, 10, dtype=torch.long), 
                    "attention_mask": torch.ones(1, 10, dtype=torch.long)}
                    
    dataset = MediTriageDataset(dummy_rows, MockTokenizer(), max_length=10)
    loader = DataLoader(dataset, batch_size=2)
    
    # 2. Dummy model
    xlm_config = XLMRobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1, num_attention_heads=1)
    encoder = XLMRobertaModel(xlm_config)
    model = MediTriageTransformer(encoder)
    model.eval()
    
    # 3. Evaluation Exporter
    exporter = EvaluationExporter(config.checkpoint_dir)
    
    spec_logits_list = []
    sev_logits_list = []
    spec_labels_list = []
    sev_labels_list = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            spec_labels = batch["labels_specialist"]
            sev_labels = batch["labels_severity"]
            
            spec_logits, sev_logits = model(input_ids, attention_mask)
            
            exporter.add_batch(
                batch["id"], batch["split"], batch["dataset_source"], batch["language"],
                spec_logits, sev_logits, spec_labels, sev_labels
            )
            
            spec_logits_list.append(spec_logits)
            sev_logits_list.append(sev_logits)
            spec_labels_list.append(spec_labels)
            sev_labels_list.append(sev_labels)
            
    print("Exporting Predictions...")
    exporter.export()
    
    # 4. Reports
    print("Generating Reports...")
    generate_training_report(
        config.checkpoint_dir, config, "smoke_123", "dummy_commit", 5.0, 0.99, "dummy_hash"
    )
    
    # 5. Calibration
    print("Running Calibration...")
    calibrator = Calibrator()
    calibrator.fit(
        torch.cat(spec_logits_list), torch.cat(spec_labels_list),
        torch.cat(sev_logits_list), torch.cat(sev_labels_list),
        config.checkpoint_dir
    )
    
    # 6. Explainability
    print("Running Explainability...")
    registry = ExplainabilityRegistry(model)
    ig = registry.get_hook("ig")
    ig.enable()
    attr = ig.analyze(torch.zeros(1, 10, dtype=torch.long), torch.ones(1, 10, dtype=torch.long), target_class=0)
    assert attr["method"] == "IntegratedGradients"
    
    print("Smoke Test Complete! Artifacts generated successfully.")

if __name__ == "__main__":
    main()
