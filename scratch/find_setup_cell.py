import json
from pathlib import Path

notebook_path = Path("EPATH_CO_REASON_Training.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "git clone" in src or "requirements.txt" in src or "pip install" in src:
            print(f"Cell {i} matches setup keyword!")
            print(src[:200])
