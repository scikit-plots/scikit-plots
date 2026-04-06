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
import importlib.util
import re
import sys
from dataclasses import dataclass

from ._custom import get_provider

__all__ = [
    "MlflowVersion",
    "is_mlflow_installed",
    "mlflow_version",
]


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
    try:
        return importlib.util.find_spec("mlflow") is not None
    except ValueError:
        # importlib.util.find_spec raises ValueError when the module is already in
        # sys.modules but has __spec__ = None (e.g., types.ModuleType stubs or
        # partially-initialised modules).  In this case the module *is* present.
        return "mlflow" in sys.modules and sys.modules["mlflow"] is not None


def _parse_version(raw: str) -> MlflowVersion:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", raw)
    if not m:
        # Fallback to 0.0.0 for unexpected version strings; callers should treat raw string.
        return MlflowVersion(raw=raw, major=0, minor=0, patch=0)
    return MlflowVersion(
        raw=raw,
        major=int(m.group(1)),
        minor=int(m.group(2)),
        patch=int(m.group(3)),
    )


def mlflow_version() -> MlflowVersion | None:
    """
    Retrieve the installed MLflow version (if available).

    Returns
    -------
    MlflowVersion or None
        Parsed version if MLflow is importable and version can be resolved, otherwise None.

    Notes
    -----
    Uses package metadata (`importlib.metadata.version`).
    Prefers module attribute `__version__` to support mocked or vendored MLflow.
    Falls back to package metadata when available.
    """
    provider = get_provider()
    if provider is not None and provider.version is not None:
        return _parse_version(provider.version)

    if not is_mlflow_installed():
        return None

    # --- Try module-based version (works with mocks) ---
    try:
        mlflow = importlib.import_module("mlflow")
        raw = getattr(mlflow, "__version__", None)
        if raw:
            return _parse_version(raw)
    except Exception:  # noqa: BLE001
        # Fail closed, continue to metadata fallback
        pass

    # --- Fallback to package metadata ---
    try:
        raw = importlib.metadata.version("mlflow")
        return _parse_version(raw)
    except importlib.metadata.PackageNotFoundError:
        return None
