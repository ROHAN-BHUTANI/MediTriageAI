import json
from pathlib import Path

notebook_path = Path("EPATH_CO_REASON_Training.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(21, 27):
    cell = nb["cells"][idx]
    print(f"=== Cell {idx} ({cell.get('cell_type')}) ===")
    print("".join(cell.get("source", [])))
