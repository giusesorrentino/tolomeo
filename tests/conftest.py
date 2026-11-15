import matplotlib
import pytest

# Force matplotlib to use the non-interactive Agg backend before any test
# imports it. Without this, matplotlib defaults to a platform-specific GUI
# backend that requires a display. CI runners are headless and have no display,
# so creating a figure would fail. Agg renders figures in memory only: all
# Figure methods behave identically to the interactive backends, just without
# opening any window.

matplotlib.use("Agg")

# Source - https://stackoverflow.com/a/61193490
# Posted by Manu J4, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-15, License - CC BY-SA 4.0


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
