"""Pytest configuration and shared fixtures for MediTriageAI test suite."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (e.g. full model training on CPU).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (requires --run-slow to run)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --run-slow given: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="Slow test – pass --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
