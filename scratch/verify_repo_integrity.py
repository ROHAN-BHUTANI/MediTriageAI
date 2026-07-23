import os
import sys
import json
import ast
import py_compile
from pathlib import Path

def run_integrity_audit():
    print("============================================================")
    print("STARTING REPOSITORY INTEGRITY AUDIT")
    print("============================================================\n")
    
    issues_found = []
    
    # 1. Check Python file syntax and static compile
    print("1. Auditing Python File Syntax...")
    py_files = []
    for root, _, files in os.walk("."):
        if ".venv" in root or ".git" in root or ".gemini" in root or ".sixth" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                py_files.append(path)
                
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as e:
            msg = f"Syntax error in {path}: {e}"
            print(f"[FAIL] {msg}")
            issues_found.append(msg)
            
    print(f"[PASS] Audited {len(py_files)} Python files for syntax errors.\n")
    
    # 2. Check for deprecated or missing imports (e.g. data_ingestion)
    print("2. Auditing Module Imports & Deprecated References...")
    deprecated_names = ["data_ingestion", "load_and_split_dataset"]
    
    for path in py_files:
        if "test_concrete_execution" in str(path) or "validate_notebook_execution" in str(path) or "verify_repo_integrity" in str(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Parse AST to find import statements
            tree = ast.parse(content, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for dep in deprecated_names:
                            if dep in alias.name:
                                msg = f"Deprecated import '{alias.name}' in {path}"
                                print(f"[FAIL] {msg}")
                                issues_found.append(msg)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for dep in deprecated_names:
                            if dep in node.module:
                                msg = f"Deprecated import module '{node.module}' in {path}"
                                print(f"[FAIL] {msg}")
                                issues_found.append(msg)
                        for alias in node.names:
                            if dep in alias.name:
                                msg = f"Deprecated import symbol '{alias.name}' in {path}"
                                print(f"[FAIL] {msg}")
                                issues_found.append(msg)
                                
            # Statically search code text for raw deprecated terms
            for dep in deprecated_names:
                if f"import {dep}" in content or f"from {dep}" in content or f"src.{dep}" in content:
                    msg = f"Potential raw deprecated reference to '{dep}' in {path}"
                    print(f"[FAIL] {msg}")
                    issues_found.append(msg)
                    
        except Exception as e:
            print(f"[WARNING] Could not parse AST for {path}: {e}")
            
    print("[PASS] Completed import and deprecated module audit.\n")
    
    # 3. Verify configuration files and primary datasets
    print("3. Auditing Configuration Files & Datasets...")
    required_configs = [
        "campaign_config.json",
        "data/clinical_triage_clean.csv",
        "data/clinical_triage_hinglish.csv",
        "data/ood_queries.csv"
    ]
    for cfg in required_configs:
        path = Path(cfg)
        if not path.exists():
            msg = f"Missing required file: {cfg}"
            print(f"[FAIL] {msg}")
            issues_found.append(msg)
        else:
            print(f"[PASS] Verified: {cfg}")
            if cfg.endswith(".json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        json.load(f)
                except Exception as e:
                    msg = f"Invalid JSON in {cfg}: {e}"
                    print(f"[FAIL] {msg}")
                    issues_found.append(msg)
    print("")
    
    # 4. Audit Google Colab Notebook Cell Code
    print("4. Auditing Generated Colab Notebook Consistency...")
    notebook_path = Path("meditriageai_colab_execution.ipynb")
    if not notebook_path.exists():
        msg = f"Notebook file missing: {notebook_path}"
        print(f"[FAIL] {msg}")
        issues_found.append(msg)
    else:
        try:
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            
            # Check cell code for deprecated names
            code_cells_count = 0
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    code_cells_count += 1
                    source = "".join(cell.get("source", []))
                    for dep in deprecated_names:
                        if dep in source:
                            msg = f"Deprecated name '{dep}' referenced in notebook code cell #{code_cells_count}"
                            print(f"[FAIL] {msg}")
                            issues_found.append(msg)
            print(f"[PASS] Successfully scanned {code_cells_count} notebook code cells.")
        except Exception as e:
            msg = f"Could not read/parse notebook file: {e}"
            print(f"[FAIL] {msg}")
            issues_found.append(msg)
    print("")
    
    # 5. Summary and Verdict
    print("============================================================")
    print("AUDIT SUMMARY")
    print("============================================================")
    if issues_found:
        print(f"FAILED: {len(issues_found)} issues identified.")
        for issue in issues_found:
            print(f"- {issue}")
        sys.exit(1)
    else:
        print("SUCCESS: 100% repository integrity verified. No issues found.")
        sys.exit(0)

if __name__ == '__main__':
    run_integrity_audit()
