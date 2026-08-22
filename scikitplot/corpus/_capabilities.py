# scikitplot/corpus/_capabilities.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Runtime capability snapshot for reproducibility (CORPUS-PKG-001).

Capability discovery in the corpus is *distributed*: ANN backends probe imports
via ``VectorIndexBackend.is_available``, chunkers/enrichers keep per-module ``_HAS_*`` /
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
from enum import unique
from typing import Any, Iterable

from ._schema import _StrEnumBase

__all__ = [
    "CapabilityStatus",
    "capability_snapshot",
    "distribution_version",
    "probe_backend",
]

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


@unique
class CapabilityStatus(_StrEnumBase):
    """Why a capability is or is not usable.

    Notes
    -----
    **User-focused.**  ``AVAILABLE`` means usable now.  Everything else explains
    *why not*, which determines what to do about it: ``ABSENT`` means install
    something, ``BROKEN`` means an installed component is failing, and
    ``MISCONFIGURED`` means the Python package is present but a non-Python
    prerequisite is not.

    **Developer-focused.**  A boolean was not enough.  Finding F-R02-05 measured
    that a backend whose ``is_available()`` *raised* -- an installed but corrupt,
    mis-linked or ABI-incompatible native library -- reported exactly the same as
    one that was never installed::

        BROKEN backend (is_available raises) -> {'available': False, ...}
        ABSENT backend (not installed)       -> {'available': False, ...}
        indistinguishable: True

    Those need opposite responses, so they are now different states.
    """

    AVAILABLE = "available"
    """Present, importable and reporting itself usable."""

    ABSENT = "absent"
    """Not installed. Install the relevant extra."""

    BROKEN = "broken"
    """Installed but failing: its availability probe raised."""

    INCOMPATIBLE = "incompatible"
    """Installed at a version this build does not support."""

    MISCONFIGURED = "misconfigured"
    """Importable, but a non-Python prerequisite is missing.

    The motivating case is ``pytesseract``, which needs a Tesseract *binary*
    that pip cannot supply -- so the extra installs "successfully" while the
    capability remains unusable (finding F-R14-01).
    """

    UNREACHABLE = "unreachable"
    """A remote dependency could not be contacted."""

    UNKNOWN = "unknown"
    """Not probed. Never a guess -- see :func:`probe_backend`."""


def probe_backend(cls: Any) -> tuple[CapabilityStatus, str | None]:
    """Classify one backend's availability.

    Parameters
    ----------
    cls : type
        A backend class exposing ``is_available()``.

    Returns
    -------
    tuple of (CapabilityStatus, str or None)
        The status and a stable machine-readable reason code.  The reason is
        ``None`` when the status is ``AVAILABLE``.

    Notes
    -----
    **Developer.**  The distinction rests on *how* the probe answers: returning
    ``False`` means the backend knows it is not installed, whereas *raising*
    means it is installed enough to try and failed -- which is ``BROKEN``.
    Review disproof D-13 confirmed the probes themselves are correctly
    fail-safe; the defect was that both outcomes collapsed to one boolean.
    """
    try:
        usable = bool(cls.is_available())
    except ImportError as exc:
        return CapabilityStatus.ABSENT, f"import_failed: {type(exc).__name__}"
    except Exception as exc:  # noqa: BLE001 - probe must never propagate
        return CapabilityStatus.BROKEN, f"probe_raised: {type(exc).__name__}"
    if usable:
        return CapabilityStatus.AVAILABLE, None
    return CapabilityStatus.ABSENT, "not_installed"


def _ann_backends() -> dict[str, dict[str, Any]]:
    """Availability + version for each registered ANN backend (read-only)."""
    try:
        from ._similarity._backends import (  # noqa: PLC0415
            _BACKENDS,
            backend_aliases,
        )
    except Exception:  # noqa: BLE001 - snapshot must never fail on import issues
        return {}

    result: dict[str, dict[str, Any]] = {}
    for name, cls in sorted(_BACKENDS.items()):
        status, reason = probe_backend(cls)
        dist = _BACKEND_DISTRIBUTION.get(name)
        result[name] = {
            "status": status.value,
            "reason_code": reason,
            # Derived from `status`; the most common question deserves a direct
            # answer, but `status` is the authority.
            "available": status is CapabilityStatus.AVAILABLE,
            "version": distribution_version(dist) if dist else None,
            # F-R02-06: aliases are reported as a FIELD, never as extra
            # entries, so consumers counting backends get the true count.
            "aliases": backend_aliases(name),
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
