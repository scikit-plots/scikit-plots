# scikitplot/_cli/logging.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""CLI logging policy: configured once, routed to stderr.

Logging never touches stdout, so machine output (``--format json``) stays clean
(invariant: stdout is the result channel; see FINDINGS CLI-FE-005).
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_HANDLER: logging.Handler | None = None

# Net verbosity -> level. 0 is the WARNING baseline; +v lowers, -q raises.
_LEVELS: Final = {
    -2: logging.CRITICAL,
    -1: logging.ERROR,
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def level_for(verbosity: int) -> int:
    """Map a net verbosity to a :mod:`logging` level (clamped)."""
    clamped = max(-2, min(2, verbosity))
    return _LEVELS[clamped]


def resolve(verbose: int = 0, quiet: int = 0) -> int:
    """Return net verbosity from ``-v`` and ``-q`` counts.

    A single source of truth so the argparse and click frontends compute the
    same net level for the same arguments (frontend parity).

    Parameters
    ----------
    verbose : int
        Number of ``-v`` occurrences (increase verbosity).
    quiet : int
        Number of ``-q`` occurrences (decrease verbosity).

    Returns
    -------
    int
        ``verbose - quiet`` (unclamped; :func:`level_for` clamps for logging).
    """
    return int(verbose) - int(quiet)


def configure(verbosity: int = 0) -> None:
    """Install (once) a stderr handler and set the level for ``verbosity``.

    Idempotent: repeated calls adjust the level and re-point the handler at the
    current ``sys.stderr`` (so diagnostics follow stream redirection), without
    stacking duplicate handlers. Diagnostics never touch stdout.
    """
    global _HANDLER  # ruff: ignore[global-statement]
    # It is set to the absolute name of the module as imported.
    logger = logging.getLogger("scikitplot")  # __name__ "scikitplot._cli.logging"
    logger.setLevel(level_for(verbosity))
    if _HANDLER is None:
        _HANDLER = logging.StreamHandler(sys.stderr)
        _HANDLER.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(_HANDLER)
        logger.propagate = False
    else:
        # Follow the current stderr without flushing the previous stream
        # (which may already be closed, e.g. across test cases).
        _HANDLER.stream = sys.stderr


__all__ = ["configure", "level_for", "resolve"]
