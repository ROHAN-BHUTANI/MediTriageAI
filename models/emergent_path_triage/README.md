# E-PATH-CO-REASON: Foundational Module Layout & Developer Reference

This directory implements the permanent software foundation for the **Emergent Path-Aligned Co-evolutionary Reasoning Network (E-PATH-CO-REASON)**.

---

## 1. Directory & Module Layout

- [constants.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/emergent_path_triage/constants.py): Stores architectural and optimization constants. No magic numbers reside in the logic.
- [exceptions.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/emergent_path_triage/exceptions.py): Establishes a custom error hierarchy (`ConfigurationError`, `RoutingError`, `InterfaceError`, `CompatibilityError`) inheriting from `MediTriageError`.
- [types.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/emergent_path_triage/types.py): Implements strongly-typed dataclasses representing representations, decisions, and paths. Dataclasses validate their parameters on construction and support deterministic `to_dict()` and `from_dict()` serialization.
- [interfaces.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/emergent_path_triage/interfaces.py): Declares abstract base interfaces and specifies stable API contracts, shape dimensions, and device mapping constraints. Includes the `BaseCheckpointRegistry` contract.
- [model.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/emergent_path_triage/model.py): Integrates the model wrapper with the repository's zoo registry and implements `EmergentPathCheckpointRegistry`.
- [hooks.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/emergent_path_triage/hooks.py): Exposes training hook interfaces (`apply_loss_hook`) called by the training orchestrator.
- [logger.py](file:///c:/Users/bhuta/Desktop/MediTriageAI_Data_Engine/models/emergent_path_triage/logger.py): Structured logging using the `models.emergent_path_triage` namespace.

---

## 2. API Stability Guidelines

To support future research iterations without breaking model training scripts:

- **Stable Public APIs**:
  - `EmergentPathTriageTransformer.forward(input_ids, attention_mask) -> ModelOutputs`
  - `EmergentPathTriageTransformer.compute_loss(spec_logits, sev_logits, labels_spec, labels_sev, loss_fn) -> dict`
  - `EmergentPathCheckpointRegistry.verify_compatibility(checkpoint_meta, current_config) -> bool`
- **Extension Points**:
  - Developers implementing the neural layers must inherit from the abstract classes in `interfaces.py`.
  - Override `initialize_weights()` inside subclasses to customize parameter initializations.

---

## 3. Lifecycle & Checkpoint Contract

```
                     [ Class Instantiation ]
                                │
                                ▼
                     [ initialize_weights() ]
                                │
                                ▼
                     [ Training / Validation ]
                     (compute_loss execution)
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
[ save_checkpoint ]                          [ verify_compatibility() ]
(Saves model parameters                      (Verifies schema versions
 & checkpoint_metadata.json)                  & latent sizes before load)
```

1. **Instantiation**: The model is built from `EmergentPathTriageModel.build(config)`.
2. **Deterministic Setup**: Weights are initialized using `initialize_weights()`.
3. **Execution & Loss**: The trainer runs the forward pass and uses `compute_loss` to optimize.
4. **Checkpoint Audit**: When loading checkpoints, `verify_compatibility()` checks that the stored schema matches the running config, raising `CompatibilityError` if parameters or versions are misaligned.

---

## 4. Implementation Roadmap for Future Phases

### Phase 2: Neural Logic Implementation
- Implement `ClinicalEvidenceSynthesizer` (inheriting from `BaseClinicalEvidenceSynthesizer`) using linear projections and cosine orthogonality optimization.
- Implement `ReasoningRouter` (inheriting from `BaseReasoningRouter`) implementing Gumbel-Softmax discrete path routing.
- Implement `ClinicalThoughtBlock` (inheriting from `BaseClinicalThoughtBlock`) using transformer layers.
- Implement `ConsistencyProjection` (inheriting from `BaseConsistencyProjection`) projecting outputs into the Urgency Manifold.

### Phase 3: Evaluation & Audit Dashboard
- Trace path codes (`path_identifier`) to expose dynamic routing decisions.
- Add an interpretability module verifying emergent clinical specialization.
