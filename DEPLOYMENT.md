# Deployment Guide

Once you have a fully trained Multi-Task Triage model checkpoint (`checkpoint_epoch_X.pt`), you can serve it via a REST API.

## Starting the API Server
We provide a lightweight FastAPI wrapper to serve the model.

```bash
python scripts/serve_api.py --checkpoint /path/to/checkpoint_epoch_best.pt
```

## Sending Inference Requests
You can query the API using `curl` or any HTTP client.

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Patient presents with sudden onset left-sided chest pain radiating to the jaw, accompanied by diaphoresis."}'
```

### Response Format
The model will output dual-head classification logits wrapped in probabilities:
```json
{
  "department_predictions": {
    "Cardiology": 0.98,
    "Emergency": 0.01
  },
  "severity_prediction": 1,
  "severity_confidence": 0.95
}
```

## Exporting for Edge Devices
The architecture relies on the HuggingFace `AutoModel` abstractions. You can export the PyTorch `nn.Module` to ONNX format using standard conversion scripts provided in the HF optimum library if you require ultra-low latency CPU deployment.
