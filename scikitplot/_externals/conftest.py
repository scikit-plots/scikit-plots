"""
Shared fixtures.

Notes
-----
**Developer notes.** Shared setup lives here, in one tunable place, rather than
being copied into each module. Every fixture is function-scoped: these tests
assert that the pipeline holds no state between documents, so a session-scoped
object would undermine the property under test.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _library_default_log_level():
    """
    Run every test at the level the library has when nobody configures it.

    Notes
    -----
    **Developer notes.** The library logs its audit trail at ``INFO`` and is
    silent by default, because an unconfigured logger inherits ``WARNING``
    from the root. The project's pytest configuration lowers the root to
    ``INFO`` and prints records live, so every ``encode`` and ``decode`` in
    this suite became a line of output: 4 413 of them, between and across the
    result lines (``CP-091``). Holding the namespace at ``WARNING`` restores
    the condition a user starts from. A test about the audit trail asks for
    the level it needs, as an application would, and gets it back here.
    """
    import logging

    logger = logging.getLogger("scikitplot._externals")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)
