# scikitplot/corpus/_capabilities.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Runtime capability snapshot for reproducibility (CORPUS-PKG-001).

Capability discovery in the corpus is *distributed*: ANN backends probe imports
via ``ANNBackend.is_available``, chunkers/enrichers keep per-module ``_HAS_*`` /
``_AVAILABLE`` flags, and ~30 modules carry optional-import fallbacks. That makes
it hard to record, for a given run, *which* optional components were actually
present and at what version.

:func:`capability_snapshot` consolidates that into a single, read-only structure
suitable for embedding in a run/build manifest: the Python/platform identity, the
availability of each registered ANN backend, and the installed version (or
``None``) of the optional distributions that affect corpus behaviour. It changes
no state and imports nothing heavy, so it is safe to call anywhere.

Notes
-----
This is the reproducibility *snapshot*; wiring it into result objects as an
ordered provenance manifest is tracked separately (CORPUS-OBS-001).
"""

from __future__ import annotations

import importlib.metadata as _md
import platform
from typing import Any, Iterable

__all__ = ["capability_snapshot", "distribution_version"]

#: Optional distributions whose presence/version affects corpus behaviour.
#: Grouped by role for readability; order is not significant.
_RELEVANT_DISTRIBUTIONS = (
    # core numerical / ML stack
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    # ANN / similarity backends
    "annoy",
    "faiss-cpu",
    "voyager",
    # embedding backends
    "sentence-transformers",
    "torch",
    "transformers",
    # readers / export / IO
    "lxml",
    "joblib",
    "polars",
    "pyarrow",
    "requests",
    "beautifulsoup4",
    # chunkers / enrichers
    "nltk",
    "regex",
    "jieba",
)

#: Best-effort map from ANN backend name to its PyPI distribution, used only to
#: annotate the snapshot with a version. Unmapped backends report ``None``.
_BACKEND_DISTRIBUTION = {
    "annoy": "annoy",
    "faiss": "faiss-cpu",
    "voyager": "voyager",
    "bruteforce": "numpy",
}


def distribution_version(name: str) -> str | None:
    """
    Return the installed version of a distribution, or ``None`` if absent.

    Parameters
    ----------
    name : str
        Distribution (PyPI) name, e.g. ``"numpy"``.

    Returns
    -------
    str or None
        The version string, or ``None`` when the distribution is not installed
        (or its metadata cannot be read).
    """
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - metadata backends can raise oddly
        return None


def _ann_backends() -> dict[str, dict[str, Any]]:
    """Availability + version for each registered ANN backend (read-only)."""
    try:
        from ._similarity._backends import _BACKENDS  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - snapshot must never fail on import issues
        return {}

    result: dict[str, dict[str, Any]] = {}
    for name, cls in sorted(_BACKENDS.items()):
        try:
            available = bool(cls.is_available())
        except Exception:  # noqa: BLE001
            available = False
        dist = _BACKEND_DISTRIBUTION.get(name)
        result[name] = {
            "available": available,
            "version": distribution_version(dist) if dist else None,
        }
    return result


def capability_snapshot(
    extra_distributions: Iterable[str] = (),
) -> dict[str, Any]:
    """
    Return a read-only snapshot of the runtime corpus capabilities.

    Parameters
    ----------
    extra_distributions : iterable of str, optional
        Additional distribution names to record beyond the built-in set.

    Returns
    -------
    dict
        A mapping with keys:

        ``python`` : str
            Python version (e.g. ``"3.11.6"``).
        ``implementation`` : str
            Interpreter implementation (e.g. ``"CPython"``).
        ``platform`` : str
            Platform identifier from :func:`platform.platform`.
        ``ann_backends`` : dict
            ``{name: {"available": bool, "version": str | None}}`` for every
            registered ANN backend.
        ``distributions`` : dict
            ``{distribution: version | None}`` for the relevant optional
            packages (plus any *extra_distributions*).

    Notes
    -----
    Purely observational — it acquires no locks, mutates no state, and never
    raises for a missing optional component (absent components report ``None`` /
    ``False``), so it is safe to embed in a run/build manifest.
    """
    dists: dict[str, str | None] = {}
    for dist in tuple(_RELEVANT_DISTRIBUTIONS) + tuple(extra_distributions):
        dists[dist] = distribution_version(dist)

    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "ann_backends": _ann_backends(),
        "distributions": dists,
    }
