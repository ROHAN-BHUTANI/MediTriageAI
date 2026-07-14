# MediTriageAI — Literature Survey (Grounding Document)

**Purpose:** Feed this into Antigravity/Gemini as context so it cites real work instead of inventing references, and so it understands where MediTriageAI's actual novelty sits relative to prior art. Organized by theme, matching a typical Related Work section structure. Every entry below is from a live search — verify each URL yourself before final citation in the paper, and pull full bibliographic details (venue, page numbers) from the source page itself.

---

## 1. Clinical Triage & Severity Classification with Transformers

- **Few-Shot LLMs for Actionable Triage Categorization of Online Patient Inquiries** (arXiv, 2026) — https://arxiv.org/pdf/2605.15680
  Surveys triage NLP moving from clinician-authored documentation to patient-authored messages; cites Gatto et al. (2022) on perceived severity in COVID telemedicine queries and Si et al. (2020) on BERT-based patient-message triage in small-data settings. **Directly relevant**: your Hinglish patient-input framing sits in this exact sub-line of work.

- **Deep learning-based NLP for detecting medical symptoms and histories in emergency patient triage** (Lee et al., Emergency Medicine, March 2024) — via physiciansweekly.com summary
  Fine-tuned BERT (KLUE-RoBERTa) to identify 12 symptoms + 2 histories from simulated ED conversations; used SHAP for explainability, validated via Turing test. Strong precedent for pairing severity classification with an explainability layer.

- **Reliability-Oriented Multilingual Orthopedic Diagnosis: Domain-Adaptive Modeling and Conceptual Validation Framework** (arXiv, 2026) — https://arxiv.org/pdf/2605.02266
  References mDeBERTa for low-resource multilingual pretraining and multilingual medical text simplification corpora (MultiMSD). Useful for framing "reliability" as a first-class evaluation axis, not just accuracy.

- **Domain-Specific Multilingual Strategies for Medical NLP: Cross-Lingual Analysis of Orthographic and Phonemic Representations** (IEEE EMBC, 2025) — Kim et al.
  Nearly identical framing to your phonetic perturbation engine — cross-lingual orthographic/phonemic representation analysis in medical NLP. **Cite this explicitly** to show you know the closest prior art, then differentiate on the Hinglish-specific deterministic perturbation method.

---

## 2. Multilingual / Low-Resource Clinical NLP

- **Recognition and normalization of multilingual symptom entities using in-domain-adapted BERT models** (PMC, 2024) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11352596/
  SympTEMIST shared task (BioCreative VIII): symptom/sign/finding detection and normalization across multilingual clinical text. Notes clinical NLP's data scarcity problem is worse for low-resource languages — a good citation for motivating why Hindi/Hinglish clinical NLP is underserved.

- **Multilingual and Cross-Linguistic Challenges in NLP** (Jain, Springer, 2025) — in *Transformative NLP: Bridging Ambiguity in Healthcare, Legal, and Financial Applications*
  Book chapter directly on cross-linguistic healthcare NLP challenges.

---

## 3. Hinglish / Code-Mixed NLP (benchmarks and datasets)

- **COMI-LINGUA: Expert Annotated Large-Scale Dataset for Multitask NLP in Hindi-English Code-Mixing** (arXiv, 2025) — https://arxiv.org/pdf/2503.21670
  Reviews the code-mixed NLP landscape: LinCE benchmark (11 corpora, 4 language pairs), GLUECoS (multilingual model fine-tuning across code-switched tasks), L3Cube-HingCorpus, and translation work by Dhar et al. (2018) and Srivastava & Singh (2020). This is your single best Related-Work anchor for the Hinglish-NLP-infrastructure paragraph.

- **AI4Bharat IndicNLP Catalog** — https://github.com/AI4Bharat/indicnlp_catalog
  Catalogs the IIIT-H English-Hindi code-mixed gold-standard corpus (6,096 sentences), CALCS 2021 Eng-Hinglish dataset (10k pairs), and multiple Hindi-English hate-speech/offensive-language datasets. Useful as a "prior datasets we considered and why MTSamples + our own perturbation pipeline was necessary instead" citation.

- **Sociolinguistically Informed Interpretability: Hinglish Emotion Classification** (arXiv, 2024) — https://arxiv.org/pdf/2402.03137
  Studies how multilingual PLMs learn (and overgeneralize) sociolinguistic associations between language choice and emotional expression in Hinglish — relevant if you discuss language-wise error analysis.

---

## 4. Multi-Task / Dual-Head Architectures for Severity + Category Classification

- **RecallRisk-BERT: A Multi-Task Framework for Post-Report Medical Device Recall Triage** (arXiv, 2026) — https://arxiv.org/pdf/2606.27174
  Near-identical architecture to yours: shared transformer encoder + hard parameter sharing, separate task-specific output heads, weighted linear combination of cross-entropy losses (L_total = λ1·L_severity + λ2·L_category), with inverse-frequency class weighting for imbalance. **This is your strongest direct architectural precedent** — cite it explicitly when justifying your loss formulation.

- **Multitask and Transfer Learning Approach for Joint Classification and Severity Estimation of Dysphonia** (PMC) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10776101/
  Shows MTL reduces computational cost vs. training separate models per task, and that auxiliary tasks (e.g. demographic prediction) can improve main-task generalization — relevant if you consider adding an auxiliary task (e.g., language-ID) to your architecture.

- **CMHL: Contrastive Multi-Head Learning for Emotionally Consistent Text Classification** (arXiv, 2026) — https://arxiv.org/pdf/2603.14078
  Frames joint diagnosis + severity optimization as replicating clinical reasoning, where the two judgments are "inextricably connected" — good framing language for your Introduction.

---

## 5. Robustness to Noisy / Code-Mixed / Typo'd Text

- **Exploring Robustness of Multilingual LLMs on Real-World Noisy Data** (arXiv, 2025) — https://arxiv.org/html/2501.08322
  Key finding: prior robustness studies on BERT/XLM-R/XLNet mostly used **simulated** noise, not real-world noise, and mostly evaluated small (<0.3B parameter) models. **This is your novelty wedge** — if your phonetic perturbation engine is modeling authentic Hinglish typing/transliteration error patterns rather than generic noise injection, say so explicitly and cite this gap.

- **Noisy Text Data: Achilles' Heel of BERT** (Srivastava, Makhija & Gupta, W-NUT 2020) — https://arxiv.org/pdf/2003.12932
  Foundational paper on BERT's noise sensitivity; still widely cited as the starting point for this line of work.

---

## 6. Confidence Calibration for Clinical Decision Support

- **A Critical Perspective on Finite Sample Conformal Prediction Theory in Medical Applications** (arXiv, 2025) — https://arxiv.org/pdf/2512.14727
  Cites the **FUTURE-AI consensus guideline**, which explicitly requires calibrated uncertainty outputs as a traceability requirement for trustworthy clinical AI. This is your best citation for justifying *why* you built a calibration/confidence layer at all — it's not a nice-to-have, it's an emerging regulatory/best-practice expectation.

- **A Comparative Study of Confidence Calibration in Deep Learning: From Computer Vision to Medical Imaging** (arXiv) — https://arxiv.org/pdf/2206.08833
  Standard reference for Expected Calibration Error (ECE) methodology, useful for your Phase 4 statistical validation section.

- **Improving Reliability of Clinical Models Using Prediction Calibration** (Springer) — https://link.springer.com/chapter/10.1007/978-3-030-60365-6_8
  Introduces reliability plots to quantify trade-off between model autonomy and generalization — directly usable methodology for your confidence-calibration evaluation.

---

## 7. Real-World Deployment Precedent (for the "major project" / product framing)

- **eSanjeevani (India's national telemedicine platform, MoHFW)** — via medrxiv.org/content/10.1101/2025.11.22.25340800
  World's largest documented telemedicine implementation in primary healthcare, ~0.4–0.45 million consultations/day. Its AI Clinical Decision Support System was validated by independent physicians across clarity, relevance, and diagnostic logic, and its symptom repository has been translated into 12 regional Indian languages including Hindi. **This is your strongest "this is a real, funded, national-scale problem" citation** for the Introduction/Motivation section of both the paper and any product pitch deck.

- **ASHABot (Khushi Baby + Microsoft Research India)** — via borgenproject.org
  WhatsApp-based, GPT-4-powered chatbot launched 2024, multilingual (Hindi/English/Hinglish), serving India's ~1 million ASHA community health workers who reach 800–900 million people. Directly validates the Hinglish-multilingual angle as a live deployment need, not a toy problem.

- **Revolutionizing Rural Healthcare in India: AI-Powered Chatbots for Affordable Symptom Analysis** (IEEE, 2024) — https://ieeexplore.ieee.org/document/10544758/
  IEEE-published precedent for exactly this problem framing (rural India, symptom-checker chatbot, cost/access barriers) — useful as a direct comparator your paper should distinguish itself from methodologically.

---

## 8. Dataset Foundation (MTSamples)

- **An empirical evaluation of dimensionality reduction and class balancing for medical text classification** (Nature Scientific Reports, 2025) — https://www.nature.com/articles/s41598-025-30537-w
  Uses 5,046 de-identified MTSamples notes across 31 specialties; PCA + SMOTE achieved 91.2% accuracy with 5-fold CV using ClinicalBERT as the transformer baseline. Good comparison point for your own MTSamples-derived accuracy numbers.

- **BRIDGE: Benchmarking LLMs for Understanding Real-World Clinical Practice Text** (arXiv, 2025) — https://arxiv.org/pdf/2504.19467
  Formally defines the MTSamples classification task (classify transcription into clinical specialty/document type) as a benchmark task — useful for framing your task setup as benchmark-aligned rather than ad hoc.

---

## How to use this file
- **Phase 5 (Novelty Check)** prompt: feed this file in and ask Gemini to position your specific contribution (Hinglish phonetic perturbation + dual-head architecture + calibration) against sections 4, 5, and 6 above specifically — those are your closest and most dangerous prior-art overlaps.
- **Phase 6 (Related Work)** prompt: feed this file in section-by-section (matching the numbered themes above) so each paragraph of Related Work is grounded in a real, verifiable source rather than a hallucinated one.
- **Introduction/Motivation**, for both paper and product framing: use section 7 (eSanjeevani, ASHABot) to justify real-world relevance and scale.
