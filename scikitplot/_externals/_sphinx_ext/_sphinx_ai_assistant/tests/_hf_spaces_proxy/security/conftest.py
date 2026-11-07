"""
Guard the shared release-security stages (``_stages.py``) after every test.

A test that writes into a stage it took with ``_stages.shared`` fails here,
named, instead of changing the inputs of the tests that run after it.
"""

from __future__ import annotations

import pytest

from . import _stages


@pytest.fixture(autouse=True)
def _shared_stages_unchanged(request):
    yield
    _stages.verify_all(test=request.node.nodeid)
