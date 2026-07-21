import json
from pathlib import Path

notebook_path = Path("EPATH_CO_REASON_Training.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        source_lines = cell.get("source", [])
        snippet = "".join(source_lines[:3]).strip()
        print(f"Cell {i}: {snippet[:120]}")
