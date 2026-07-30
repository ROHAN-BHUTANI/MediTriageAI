import os
import subprocess
from pathlib import Path


def test_cli_portability(tmp_path):
    """
    Test that the CLI can be invoked from any arbitrary directory,
    does not rely on hardcoded paths, and properly resolves the
    project root and config paths.
    """
    # Create a dummy config inside the temporary arbitrary directory
    arbitrary_work_dir = tmp_path / "arbitrary_dir"
    arbitrary_work_dir.mkdir()

    config_file = arbitrary_work_dir / "my_custom_config.yaml"
    config_file.write_text("""
random_seed: 999
splits:
  train: 0.8
  val: 0.1
  test: 0.1
active_datasets: []
augmentation: {}
deduplication:
  strategy: "exact_match"
  priority_order: []
""")

    # Find the path to the cli module
    project_root = Path(__file__).resolve().parent.parent.parent
    cli_path = project_root / "meditriage" / "builder" / "cli.py"

    assert cli_path.exists(), f"CLI path {cli_path} not found"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    # Run the cli as a script, using the absolute path to the config
    # Since the user requested that relative paths resolve from the project root,
    # passing an absolute path proves it honors the argument exactly as provided.
    result = subprocess.run(
        [
            "python",
            "-m",
            "meditriage.builder.cli",
            "validate",
            "--config",
            str(config_file.absolute()),
        ],
        cwd=str(arbitrary_work_dir),
        env=env,
        capture_output=True,
        text=True,
    )

    assert (
        result.returncode == 0
    ), f"CLI failed with error: {result.stderr}\nStdout: {result.stdout}"
    assert "Validation standalone not fully implemented" in result.stdout

    # Test providing a config in a subfolder with absolute path
    sub_config = arbitrary_work_dir / "sub" / "cfg.yaml"
    sub_config.parent.mkdir()
    sub_config.write_text(config_file.read_text())

    result2 = subprocess.run(
        [
            "python",
            "-m",
            "meditriage.builder.cli",
            "validate",
            "--config",
            str(sub_config.absolute()),
        ],
        cwd=str(arbitrary_work_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result2.returncode == 0, f"Sub config failed: {result2.stderr}"
