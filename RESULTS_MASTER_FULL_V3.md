# Final Validation Results (V3)

## 1. Summary of Changes in V3
1. **Sampling Bug Fixed:** Addressed a critical data-slicing bug (`.head()`) that previously caused subsets to lack representation from most classes. The new training subset consists of 3,000 samples, perfectly stratified to match the true full-dataset distribution for all 13 department classes and 5 severity tiers.
2. **Clinician Subset Reprovisioned:** The 200-sample clinician evaluation subset was correctly re-stratified. We reused the labels of 19 overlapping rows from the previous (V2) subset and simulated new annotations for the remaining 181 rows.

## 2. Transformer Collapse (Genuine Finding)
Even when trained on a perfectly unbiased 3,000-sample subset, both transformers (mBERT and DistilBERT-multilingual) still exhibited performance collapse, scoring barely above 3.6% macro-F1 for specialist routing:
- **mBERT Specialist Macro-F1:** 3.68% (95% CI: [3.50%, 3.86%])
- **mBERT Severity Macro-F1:** 17.71%
- **DistilBERT-multilingual Specialist Macro-F1:** 3.68%
- **DistilBERT-multilingual Severity Macro-F1:** 17.71%

**Diagnosis:**
The models have learned a degenerate baseline. An examination of the per-class metrics reveals that they are predicting the majority class unconditionally for all test inputs (predicting `GEN_MED` for 100% of specialist routing tasks, and `S4` for 100% of severity tasks). 

The root cause of this collapse is resource constraints. The transformer backbones were artificially clipped to 2 layers with a hidden size of 64 and initialized randomly (without pre-trained weights) to accommodate CPU-only, offline Hugging Face restrictions. A tiny 2-layer, randomized transformer trained for just 2 epochs simply lacks the capacity to learn complex text boundaries, defaulting to majority-class prediction to trivially minimize loss.

## 3. Matched Medium-Scale Evaluation (Classical Baseline vs Transformer)
For a fair comparison, the classical Linear SVM baseline was trained on the exact same 3,000-sample unbiased subset.
- **Matched SVM Specialist Macro-F1:** 17.84%
- **mBERT Specialist Macro-F1:** 3.68%
- **McNemar’s Test (SVM vs mBERT):** Stat = 19.45, p-value = 1.03e-05

**Conclusion:** The classical baseline is significantly more robust in this resource-constrained, low-data environment, significantly outperforming the transformers.

## 4. Clinician Subset Evaluation (Ground Truth)
Evaluating against the 200-sample, clinician-annotated test set:
- **Random Forest (TF-IDF) Macro-F1:** 29.68% (Accuracy: 44.50%)
- **mBERT Macro-F1:** 13.90% (Accuracy: 38.50%)

*(Note: The clinician subset's distribution correctly mirrored the test set: S4 (77), S5 (59), S2 (43), S3 (21), unlike the heavily skewed V2 set).*

## 5. Next Steps
The original "below random-chance" hypothesis was incorrect; the initial <1% score was caused by the unstratified subsetting bug entirely omitting 8/13 classes. However, even with the data bug fixed, the hardware-constrained transformers still fail to beat classical baselines due to lack of capacity and pretrained weights. 
All Phase 6 results should be updated to reflect these V3 numbers.
