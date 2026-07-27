import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, accuracy_score
import time
import json
import warnings
warnings.filterwarnings('ignore')

SUPERGROUPS = {
    'SURGERY': 'SURGICAL_SCIENCES',
    'ORTHO': 'SURGICAL_SCIENCES',
    'ENT_OPHTHALMO': 'SURGICAL_SCIENCES',
    'GEN_MED': 'INTERNAL_MED_CARDIO',
    'CARDIO_PULM': 'INTERNAL_MED_CARDIO',
    'GI': 'INTERNAL_MED_CARDIO',
    'RENAL_URO': 'INTERNAL_MED_CARDIO',
    'ONCOLOGY_HEME': 'INTERNAL_MED_CARDIO',
    'NEURO': 'NEURO_PSYCH',
    'PSYCH': 'NEURO_PSYCH',
    'ED': 'EMERGENCY_TRAUMA',
    'PEDS': 'PEDS_WOMEN',
    'OBGYN': 'PEDS_WOMEN'
}

class MedDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))

def train_eval_loop(df_train, df_val, df_test, label_col, model_name, patience=3, max_epochs=15):
    label_list = sorted(df_train[label_col].unique())
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')
    train_dataset = MedDataset(df_train['text'].tolist(), [label2id[l] for l in df_train[label_col]], tokenizer)
    val_dataset = MedDataset(df_val['text'].tolist(), [label2id[l] for l in df_val[label_col]], tokenizer)
    test_dataset = MedDataset(df_test['text'].tolist(), [label2id[l] for l in df_test[label_col]], tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model = DistilBertClassifier(len(label_list)).to(device)
    
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            head_params.append(param)
        else:
            encoder_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 2e-5},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    loss_history = []
    first_step = True
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if first_step:
                print(f"[{model_name}] Step 0 LRs: Encoder={optimizer.param_groups[0]['lr']:.2e}, Classifier={optimizer.param_groups[1]['lr']:.2e}")
                first_step = False
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"[{model_name}] Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        loss_history.append((train_loss, val_loss))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"best_{model_name}.pt")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch+1}")
                break
                
    # Evaluate
    model.load_state_dict(torch.load(f"best_{model_name}.pt", weights_only=True))
    model.eval()
    all_preds = []
    all_labels = []
    top3_correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Top-3 Accuracy
            top3 = torch.topk(logits, k=min(3, len(label_list)), dim=-1).indices
            for i in range(len(labels)):
                if labels[i] in top3[i]:
                    top3_correct += 1
                    
    top3_acc = top3_correct / len(all_labels)
    from src.metrics import compute_macro_f1
    macro_f1 = compute_macro_f1(all_labels, all_preds, "specialist")
    report = classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(len(label_list))], output_dict=True)
    dist = {id2label[i]: int(sum([p == i for p in all_preds])) for i in range(len(label_list))}
    
    return {
        'loss_history': loss_history,
        'macro_f1': macro_f1,
        'top3_acc': top3_acc,
        'report': report,
        'distribution': dist
    }

def main():
    df = pd.read_csv('data/processed/enriched/dataset_enriched.csv')
    df['text'] = df['text'].fillna('')
    df['supergroup'] = df['department_code'].map(SUPERGROUPS)
    
    # Optional: split train into train/val if needed, or just use 10% of train as val
    train_df = df[df['split'] == 'train']
    # Let's carve out 10% of train for validation (approx 1600 rows)
    val_df = train_df.sample(frac=0.1, random_state=42)
    train_df = train_df.drop(val_df.index)
    test_df = df[df['split'] == 'test']
    
    print("Training 13-Class Original")
    res_13 = train_eval_loop(train_df, val_df, test_df, 'department_code', 'orig_13')
    
    print("Training 5-Class Supergroups")
    res_5 = train_eval_loop(train_df, val_df, test_df, 'supergroup', 'consol_5')
    
    with open('v7_results.json', 'w') as f:
        json.dump({'orig': res_13, 'consol': res_5}, f, indent=2)
        
    print("All done!")

if __name__ == '__main__':
    main()
