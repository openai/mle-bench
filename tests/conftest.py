"""Pytest configuration file which adds custom command line options and markers."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-cache",
        action="store_true",
        default=False,
        help="Download and prepare all datasets from scratch.",
    )
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return  # --slow given in cli: do not skip slow tests

    skip_slow = pytest.mark.skip(reason="Need --slow option to run")

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
