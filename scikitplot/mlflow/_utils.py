# scikitplot/mlflow/_utils.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
_utils.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MlflowVersion:
    """
    Parsed MLflow version.

    Parameters
    ----------
    raw : str
        Raw version string from package metadata.
    major : int
        Major component.
    minor : int
        Minor component.
    patch : int
        Patch component.

    Notes
    -----
    This parser is intentionally conservative: it extracts the first three numeric
    components from the version string. Pre-release/build metadata is ignored for
    compatibility checks.
    """

    raw: str = ""
    major: int = 0
    minor: int = 0
    patch: int = 0

    @property
    def triple(self) -> tuple[int, int, int]:
        """Return (major, minor, patch)."""
        return (self.major, self.minor, self.patch)


def is_mlflow_installed() -> bool:
    """
    Check whether MLflow is installed in the current Python environment.

    Returns
    -------
    bool
        True if the `mlflow` module is importable.

    Notes
    -----
    This does not import MLflow; it checks import metadata only.
    """
    return importlib.util.find_spec("mlflow") is not None


def mlflow_version() -> MlflowVersion | None:
    """
    Retrieve the installed MLflow version (if available).

    Returns
    -------
    MlflowVersion or None
        Parsed version if MLflow is installed, otherwise None.

    Notes
    -----
    Uses package metadata (`importlib.metadata.version`).
    """
    if not is_mlflow_installed():
        return None
    raw = importlib.metadata.version("mlflow")
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", raw)
    if not m:
        # Fallback to 0.0.0 for unexpected version strings; callers should treat raw string.
        return MlflowVersion(raw=raw, major=0, minor=0, patch=0)
    return MlflowVersion(
        raw=raw, major=int(m.group(1)), minor=int(m.group(2)), patch=int(m.group(3))
    )
