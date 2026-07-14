# MediTriageAI - V6 Transformer Attempt (DistilBERT-multilingual)

## Setup and Fixes Applied
- **Model**: DistilBERT-multilingual
- **Encoder Freezing**: Encoder was frozen for Epoch 1 and unfrozen for Epoch 2.
- **Learning Rate Schedule**: Linear warmup (10% of total steps) applied.
- **Differential Learning Rates**: Encoder initialized to `2e-5`, Classifier heads initialized to `1e-4` (confirmed correctly routed to parameter groups, starting at `0.00e+00` due to warmup).
- **Max Length**: Reduced to `64` to meet time constraints.
- **Environment**: Intel GPU via `torch_directml`.

## Results
- **Specialist Routing Macro-F1**: 1.13% (0.0113)
- **Severity Macro-F1**: 2.76% (0.0276)

### Did it resolve Catastrophic Forgetting?
No. The performance actually dropped compared to the 3.96% V5 number. The combination of short training (2 epochs), reducing max context length to 64, and fixing the encoder freezing resulted in a severely underfit model that still exhibits strong mode collapse. 

### Label Distribution & Collapse Check
Despite the learning rate warmup and differential rates, the model collapsed entirely into predicting a single minority class. 
- **Class 10** has a recall of 56.25% but a precision of just 0.57%. This indicates the model predicted Class 10 approximately 1,578 times out of the 1,999 test samples.
- **8 out of 13** specialist classes had 0.0 precision and 0.0 recall.
- There is severe mode collapse; the model did not learn a distributed representation.

### Specialist Per-Class Precision & Recall
- **Class 0**: Prec: 0.000, Rec: 0.000
- **Class 1**: Prec: 0.000, Rec: 0.000
- **Class 2**: Prec: 0.000, Rec: 0.000
- **Class 3**: Prec: 0.000, Rec: 0.000
- **Class 4**: Prec: 0.000, Rec: 0.000
- **Class 5**: Prec: 0.036, Rec: 0.047
- **Class 6**: Prec: 0.015, Rec: 0.023
- **Class 7**: Prec: 0.000, Rec: 0.000
- **Class 8**: Prec: 0.000, Rec: 0.000
- **Class 9**: Prec: 0.045, Rec: 0.042
- **Class 10**: Prec: 0.006, Rec: 0.562
- **Class 11**: Prec: 0.031, Rec: 0.036
- **Class 12**: Prec: 0.000, Rec: 0.000

## Conclusion
This was the final scheduled attempt to patch the transformers with standard optimization fixes (warmup, differential LRs, progressive unfreezing). The failure to learn meaningful representations despite these safeguards strongly suggests that transformers are fundamentally struggling with the dataset properties (e.g., severe class imbalance, label noise, or text lengths), or that a small model like DistilBERT simply cannot converge effectively on this specific multilingual triage task under the current hyperparameters and dataset size.

As requested, we will lock in this finding and refrain from further transformer fine-tuning iterations.
