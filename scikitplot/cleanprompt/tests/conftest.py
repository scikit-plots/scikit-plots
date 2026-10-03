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

from .. import DEFAULT_POLICY, Redactor, default_registry


@pytest.fixture()
def policy():
    """Return the default policy."""
    return DEFAULT_POLICY


@pytest.fixture()
def redactor():
    """Return a redactor over the full default pattern set."""
    return Redactor()


@pytest.fixture()
def email_redactor():
    """Return a redactor restricted to email detection."""
    return Redactor(
        policy=DEFAULT_POLICY.evolve(kinds=("EMAIL",)),
        registry=default_registry(kinds=("EMAIL",)),
    )


@pytest.fixture()
def sample_text():
    """Return a text exercising several detector kinds at once."""
    return (
        "Ann met Anna at Acme. Mail a.b+tag@example.co.uk or ada@example.com, "
        "see https://example.com/x?u=z#f, call +1 555 010 4477, "
        "card 4242 4242 4242 4242, host 192.168.1.10, "
        "iban GB82 WEST 1234 5698 7654 32, ssn 123-45-6789, "
        "released 2024-01-15 with numpy 1.26.4 and order 12345678."
    )


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

    from .._logging import LOGGER_NAME

    logger = logging.getLogger(LOGGER_NAME)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)
