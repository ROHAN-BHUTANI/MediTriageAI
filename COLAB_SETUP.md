# Google Colab Training Setup (V7)

This guide provides step-by-step instructions for running the V7 training pass on a Google Colab instance utilizing the free T4 GPU.

## 1. Prepare Local Files for Upload

Zip the following essential files and upload them to your Google Drive, or keep them ready to upload directly into the Colab environment:
- `colab_train.py` (The updated training script)
- `requirements_colab.txt` (Dependencies required for the run)
- `meditriage/data/processed/dataset.csv` (The training data)

*Note: You must preserve the directory structure `meditriage/data/processed/` when uploading the dataset, as `colab_train.py` expects this path.*

## 2. Set Up the Colab Environment

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a **New Notebook**.
3. Navigate to **Runtime > Change runtime type**.
4. Select **T4 GPU** under Hardware accelerator and click Save.

## 3. Upload Files to Colab

Run the following cell to upload the required files (or use the File Explorer pane on the left to drag and drop them):

```python
from google.colab import files
uploaded = files.upload()
```

If you upload a ZIP file (e.g., `workspace.zip`), extract it:
```bash
!unzip workspace.zip
```

## 4. Install Dependencies

Install the required Python packages using the generated requirements file:

```bash
!pip install -r requirements_colab.txt
```

## 5. Execute the Training Script

Run the V7 training script. It is now configured to automatically detect the Colab GPU using standard `torch.cuda` device mapping, with `batch_size=32` and `max_length=128`.

```bash
!python colab_train.py
```

## 6. Retrieve the Results

Once training finishes, the script will output a JSON file containing the loss curves, macro-F1 scores, top-3 accuracy, and classification reports. Download the results and the trained model weights:

```python
from google.colab import files

# Download the results metrics
files.download('v7_results.json')

# Download the best model weights
files.download('best_orig_13.pt')
files.download('best_consol_5.pt')
```

After retrieving `v7_results.json`, you can share the file and we can parse it to generate the final `RESULTS_MASTER_FULL_V7.md` report locally.
