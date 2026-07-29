# Pretrained Weights Feasibility Report

## 1. Environment Assessment
I have successfully tested the connectivity to the Hugging Face model hub from within this environment. 
- **Network Connectivity:** Successful. We can reach `huggingface.co`.
- **Download Capability:** Successful. I successfully downloaded the tokenizer and full model weights for `distilbert-base-multilingual-cased` using the standard `transformers` library, which automatically cached them in the local `.cache/huggingface/hub` directory.

## 2. Feasibility Conclusion
**Yes, it is entirely feasible.** 
While the deployment environment may ultimately be fully offline, the *training/evaluation* environment currently has the network access necessary to download and cache pretrained weights. By downloading the weights once while online, we can completely bypass the artificial `ZooConfig` restrictions (which were forcing a 2-layer, 64-hidden-dimension random initialization) and use the *actual* pretrained transformer backbones.

## 3. Impact on Research Narrative
Because it is feasible, we do **not** need to reframe the paper to focus solely on low-resource, untrained neural nets. We can proceed with evaluating genuine, pretrained transformers against the classical baseline. This will finally answer the core research question: *Does a pretrained multilingual language model outperform classical TF-IDF + SVM baselines on this task?*

## 4. Recommendation
I recommend immediately updating the model initialization logic in `src/models/` to load the actual pretrained weights (via `AutoModel.from_pretrained`) rather than constructing untrained, miniature `DistilBertConfig` / `BertConfig` objects from scratch. 
Once updated, we should re-run the V3 training pipeline (on the unbiased 3,000-sample subset) and re-evaluate. This will likely yield a massive boost to the transformer's performance, fully resolving the degenerate majority-class collapse.
