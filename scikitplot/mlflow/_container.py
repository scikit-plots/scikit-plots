# scikitplot/mlflow/_container.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_container.
"""

from __future__ import annotations

from pathlib import Path


def running_in_docker() -> bool:
    """
    Detect whether the current process is running in a Docker container.

    Returns
    -------
    bool
        True if the file ``/.dockerenv`` exists, else False.

    Notes
    -----
    This check is intentionally strict and deterministic:
    - It relies only on ``/.dockerenv`` (common in Docker containers).
    - It does not attempt cgroup inference or other heuristics.
    """
    return Path("/.dockerenv").exists()
