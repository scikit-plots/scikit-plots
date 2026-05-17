"""Pytest configuration."""

from pathlib import Path

import pytest

# https://docs.pytest.org/en/stable/reference/reference.html#globalvar-pytest_plugins
# pytest_plugins = "sphinx.testing.fixtures"


@pytest.fixture(scope="session")
def rootdir():
    """Get the root directory for the whole test session."""
    return Path(__file__).parent.absolute() / "roots"
