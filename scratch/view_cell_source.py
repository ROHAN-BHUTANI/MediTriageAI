import json
from pathlib import Path

notebook_path = Path("EPATH_CO_REASON_Training.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "REPOSITORY DETECT & IMPORT CHECKS" in src:
            print(f"Cell {i}:")
            print(src)
            print("-" * 50)
