# scikitplot/_cli/exit_codes.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Centralized, stable, semantic CLI exit codes.

References
----------
* BSD/GNU convention: 0 success, 1 general error, 2 usage error.
* sysexits.h : codes 64-78 for finer categories.
"""

from __future__ import annotations

from typing import Final

OK: Final = 0
ERROR: Final = 1  # general, unclassified runtime failure
USAGE: Final = 2  # bad arguments / unknown command (argparse default)
UNAVAILABLE: Final = 69  # optional capability/dependency missing (EX_UNAVAILABLE)
SOFTWARE: Final = 70  # internal programming error (EX_SOFTWARE)
INTERRUPTED: Final = 130  # 128 + SIGINT(2); Ctrl+C

__all__ = [
    "ERROR",
    "INTERRUPTED",
    "OK",
    "SOFTWARE",
    "UNAVAILABLE",
    "USAGE",
]
