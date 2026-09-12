"""Pytest configuration."""

from pathlib import Path

import pytest

# scikit-plots local patch: the whole suite was unrunnable.
#
# Sphinx's `app`, `status` and `warning` fixtures ship in
# `sphinx.testing.fixtures`, which was registered here via `pytest_plugins`
# -- but the line was commented out, so every test errored at setup with
# "fixture 'app' not found". A test suite that cannot execute reports no
# failures, which reads exactly like a passing suite in CI.
#
# `pytest_plugins` is only honoured in a *rootdir* conftest on modern pytest,
# and this one is nested, so re-enabling that line would not have worked
# either. Importing the fixtures directly is the supported way to make them
# available from a nested conftest.
from sphinx.testing.fixtures import *  # noqa: F401,F403,E402


@pytest.fixture(scope="session")
def rootdir():
    """Get the root directory for the whole test session."""
    return Path(__file__).parent.absolute() / "roots"
