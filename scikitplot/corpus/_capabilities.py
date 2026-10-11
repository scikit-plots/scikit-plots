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

import dataclasses
import importlib.metadata as _md
import importlib.util as _iu
import platform
import shutil
from collections.abc import Callable, Iterable
from enum import unique
from typing import Any

from ._schema import _StrEnumBase

__all__ = [
    "CapabilityRegistry",
    "CapabilityReport",
    "CapabilitySpec",
    "CapabilityStatus",
    "capability_report",
    "capability_snapshot",
    "component_capabilities",
    "default_capability_registry",
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
    # media / document readers
    "faster-whisper",
    "openai-whisper",
    "pytesseract",
    "easyocr",
    "pdfminer.six",
    "PyMuPDF",
    "pypdf",
    "pdfplumber",
    "mutagen",
    "librosa",
    "soundfile",
    # optional edit-distance acceleration
    "RapidFuzz",
    "Levenshtein",
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


# =============================================================================
# Component readiness registry
# =============================================================================


@dataclasses.dataclass(frozen=True)
class CapabilitySpec:
    """Declarative probe specification for one optional component.

    A spec deliberately separates *installation* from *readiness*.  A Python
    distribution can be installed while a model, corpus, executable, shared
    library, credential, or other runtime asset is still missing.

    Parameters
    ----------
    name : str
        Stable capability identifier, for example ``"ocr:pytesseract"``.
    role : str
        Human-readable role such as ``"ocr"`` or ``"asr"``.
    module : str or None, optional
        Importable module used as a lightweight installation probe.
    distributions : tuple[str, ...], optional
        Distribution names used only for version/provenance discovery.
    assets_required : bool, optional
        Whether usability depends on an asset beyond the Python package.
    asset_probe : callable or None, optional
        Lightweight, side-effect-free probe.  Return ``True`` when assets are
        ready, ``False`` when definitely missing, and ``None`` when readiness
        cannot be known without loading/downloading the component.
    may_download : bool, optional
        Whether first use may download models/data.
    requires_network : bool, optional
        Whether normal operation requires network access.
    remedy : str or None, optional
        User-facing guidance for the common unavailable case.

    Notes
    -----
    Probes must not download assets, initialise models, or contact the network.
    ``UNKNOWN`` is preferable to inventing readiness.
    """

    name: str
    role: str
    module: str | None = None
    distributions: tuple[str, ...] = ()
    assets_required: bool = False
    asset_probe: Callable[[], bool | None] | None = None
    may_download: bool = False
    requires_network: bool = False
    remedy: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("CapabilitySpec.name must be a non-empty string")
        if not self.role or not isinstance(self.role, str):
            raise ValueError("CapabilitySpec.role must be a non-empty string")
        if self.asset_probe is not None and not callable(self.asset_probe):
            raise TypeError("CapabilitySpec.asset_probe must be callable or None")


@dataclasses.dataclass(frozen=True)
class CapabilityReport:
    """Read-only readiness state for one optional component.

    ``installed`` means Python can locate the package/module. ``assets_ready``
    answers the second question when it can be observed cheaply. ``ready`` is
    the combined preflight answer. ``selected`` and ``active`` are contextual:
    selection means policy chose the component; active means the caller has
    evidence it actually completed work successfully.
    """

    name: str
    role: str
    status: CapabilityStatus
    installed: bool
    assets_ready: bool | None
    ready: bool | None
    selected: bool = False
    active: bool = False
    version: str | None = None
    reason_code: str | None = None
    remedy: str | None = None
    may_download: bool = False
    requires_network: bool = False

    @property
    def available(self) -> bool:
        """Whether this capability is known usable now."""
        return self.ready is True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status.value,
            "installed": self.installed,
            "assets_ready": self.assets_ready,
            "ready": self.ready,
            "selected": self.selected,
            "active": self.active,
            "available": self.available,
            "version": self.version,
            "reason_code": self.reason_code,
            "remedy": self.remedy,
            "may_download": self.may_download,
            "requires_network": self.requires_network,
        }


def _module_present(module: str | None) -> tuple[bool, str | None]:
    if not module:
        return True, None
    try:
        return _iu.find_spec(module) is not None, None
    except Exception as exc:  # noqa: BLE001 - readiness probe must be fail-safe
        return False, f"module_probe_raised:{type(exc).__name__}"


def _first_distribution_version(names: tuple[str, ...]) -> str | None:
    for name in names:
        version = distribution_version(name)
        if version is not None:
            return version
    return None


def _probe_spec(
    spec: CapabilitySpec,
    *,
    selected: bool = False,
    active: bool = False,
) -> CapabilityReport:
    installed, module_reason = _module_present(spec.module)
    version = _first_distribution_version(spec.distributions)

    # Runtime success is stronger evidence than a conservative preflight probe.
    # It must not trigger installation/network activity merely to prove itself.
    if active:
        return CapabilityReport(
            name=spec.name,
            role=spec.role,
            status=CapabilityStatus.AVAILABLE,
            installed=True,
            assets_ready=True if spec.assets_required else None,
            ready=True,
            selected=selected,
            active=True,
            version=version,
            remedy=spec.remedy,
            may_download=spec.may_download,
            requires_network=spec.requires_network,
        )

    if not installed:
        return CapabilityReport(
            name=spec.name,
            role=spec.role,
            status=(
                CapabilityStatus.BROKEN
                if module_reason is not None
                else CapabilityStatus.ABSENT
            ),
            installed=False,
            assets_ready=False if spec.assets_required else None,
            ready=False,
            selected=selected,
            active=False,
            version=version,
            reason_code=module_reason or "not_installed",
            remedy=spec.remedy,
            may_download=spec.may_download,
            requires_network=spec.requires_network,
        )

    if not spec.assets_required:
        return CapabilityReport(
            name=spec.name,
            role=spec.role,
            status=CapabilityStatus.AVAILABLE,
            installed=True,
            assets_ready=None,
            ready=True,
            selected=selected,
            active=False,
            version=version,
            remedy=spec.remedy,
            may_download=spec.may_download,
            requires_network=spec.requires_network,
        )

    if spec.asset_probe is None:
        return CapabilityReport(
            name=spec.name,
            role=spec.role,
            status=CapabilityStatus.UNKNOWN,
            installed=True,
            assets_ready=None,
            ready=None,
            selected=selected,
            active=False,
            version=version,
            reason_code="assets_not_probed",
            remedy=spec.remedy,
            may_download=spec.may_download,
            requires_network=spec.requires_network,
        )

    try:
        assets_ready = spec.asset_probe()
    except Exception as exc:  # noqa: BLE001 - report, never crash discovery
        return CapabilityReport(
            name=spec.name,
            role=spec.role,
            status=CapabilityStatus.BROKEN,
            installed=True,
            assets_ready=None,
            ready=False,
            selected=selected,
            active=False,
            version=version,
            reason_code=f"asset_probe_raised:{type(exc).__name__}",
            remedy=spec.remedy,
            may_download=spec.may_download,
            requires_network=spec.requires_network,
        )

    if assets_ready is True:
        status = CapabilityStatus.AVAILABLE
        reason = None
        ready: bool | None = True
    elif assets_ready is False:
        status = CapabilityStatus.MISCONFIGURED
        reason = "required_asset_missing"
        ready = False
    else:
        status = CapabilityStatus.UNKNOWN
        reason = "asset_readiness_unknown"
        ready = None

    return CapabilityReport(
        name=spec.name,
        role=spec.role,
        status=status,
        installed=True,
        assets_ready=assets_ready,
        ready=ready,
        selected=selected,
        active=False,
        version=version,
        reason_code=reason,
        remedy=spec.remedy,
        may_download=spec.may_download,
        requires_network=spec.requires_network,
    )


class CapabilityRegistry:
    """Explicit, side-effect-free registry for component readiness probes.

    The registry is intentionally an object rather than hidden global plugin
    magic.  Applications may copy/extend the default registry or construct a
    fully custom registry for private backends.
    """

    def __init__(self, specs: Iterable[CapabilitySpec] = ()) -> None:
        self._specs: dict[str, CapabilitySpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: CapabilitySpec, *, replace: bool = False) -> None:
        """Register *spec*; refuse accidental replacement by default."""
        if spec.name in self._specs and not replace:
            raise ValueError(f"capability {spec.name!r} is already registered")
        self._specs[spec.name] = spec

    def unregister(self, name: str) -> None:
        """Remove *name*; raise ``KeyError`` when it is unknown."""
        del self._specs[name]

    def get(self, name: str) -> CapabilitySpec:
        """Return a registered spec."""
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown capability {name!r}; known={sorted(self._specs)}"
            ) from exc

    def names(self, *, role: str | None = None) -> tuple[str, ...]:
        """Return stable capability names, optionally filtered by role."""
        return tuple(
            sorted(
                name
                for name, spec in self._specs.items()
                if role is None or spec.role == role
            )
        )

    def roles(self) -> tuple[str, ...]:
        """Return the stable set of registered capability roles."""
        return tuple(sorted({spec.role for spec in self._specs.values()}))

    def probe(
        self,
        name: str,
        *,
        selected: bool = False,
        active: bool = False,
    ) -> CapabilityReport:
        """Probe one capability without loading/downloading heavy assets."""
        return _probe_spec(self.get(name), selected=selected, active=active)

    def snapshot(
        self,
        names: Iterable[str] | None = None,
        *,
        selected: Iterable[str] = (),
        active: Iterable[str] = (),
    ) -> dict[str, CapabilityReport]:
        """Probe several capabilities in stable name order."""
        selected_set = set(selected)
        active_set = set(active)
        wanted = self.names() if names is None else tuple(sorted(set(names)))
        return {
            name: self.probe(
                name,
                selected=name in selected_set,
                active=name in active_set,
            )
            for name in wanted
        }

    def copy(self) -> CapabilityRegistry:
        """Return an independent registry containing the same immutable specs."""
        return CapabilityRegistry(self._specs.values())


def _binary_probe(name: str) -> Callable[[], bool]:
    return lambda: shutil.which(name) is not None


def _builtin_capability_specs() -> tuple[CapabilitySpec, ...]:
    """Return built-in lightweight specs without importing optional packages."""
    return (
        CapabilitySpec(
            "asr:faster-whisper",
            "asr",
            module="faster_whisper",
            distributions=("faster-whisper",),
            assets_required=True,
            # Model availability depends on model name/cache and is intentionally
            # UNKNOWN until runtime rather than triggering Hugging Face access.
            may_download=True,
            remedy="Install faster-whisper and pre-cache the requested model, or allow model download at runtime.",
        ),
        CapabilitySpec(
            "asr:openai-whisper",
            "asr",
            module="whisper",
            distributions=("openai-whisper",),
            assets_required=True,
            may_download=True,
            remedy="Install openai-whisper and pre-cache the requested model, or allow model download at runtime.",
        ),
        CapabilitySpec(
            "ocr:pytesseract",
            "ocr",
            module="pytesseract",
            distributions=("pytesseract",),
            assets_required=True,
            asset_probe=_binary_probe("tesseract"),
            remedy="Install both pytesseract and the system Tesseract executable.",
        ),
        CapabilitySpec(
            "ocr:easyocr",
            "ocr",
            module="easyocr",
            distributions=("easyocr",),
            assets_required=True,
            may_download=True,
            remedy="Install easyocr and make its model assets available locally, or allow model download.",
        ),
        CapabilitySpec(
            "xml:lxml",
            "xml",
            module="lxml",
            distributions=("lxml",),
        ),
        CapabilitySpec(
            "xml:stdlib",
            "xml",
            remedy="The Python standard-library XML fallback is always available.",
        ),
        CapabilitySpec(
            "pdf:pdfminer",
            "pdf",
            module="pdfminer",
            distributions=("pdfminer.six",),
            remedy="Install pdfminer.six for the primary PDFReader extraction backend.",
        ),
        CapabilitySpec(
            "pdf:pypdf",
            "pdf",
            module="pypdf",
            distributions=("pypdf",),
            remedy="Install pypdf for the PDFReader fallback backend.",
        ),
        CapabilitySpec(
            "audio:mutagen",
            "audio-duration",
            module="mutagen",
            distributions=("mutagen",),
        ),
        CapabilitySpec(
            "audio:librosa",
            "audio-duration",
            module="librosa",
            distributions=("librosa",),
        ),
        CapabilitySpec(
            "audio:soundfile",
            "audio-duration",
            module="soundfile",
            distributions=("soundfile",),
        ),
        CapabilitySpec(
            "distance:internal",
            "edit-distance",
            module="scikitplot.cexternals._editdistance",
            remedy="Build/install the Scikit-Plots native extension, or use RapidFuzz/pure Python.",
        ),
        CapabilitySpec(
            "distance:rapidfuzz",
            "edit-distance",
            module="rapidfuzz",
            distributions=("RapidFuzz", "rapidfuzz"),
        ),
        CapabilitySpec(
            "distance:levenshtein",
            "edit-distance",
            module="Levenshtein",
            distributions=("Levenshtein", "python-Levenshtein"),
            remedy="Install Levenshtein only when its GPL-2.0-or-later licensing is acceptable for your application.",
        ),
        CapabilitySpec(
            "distance:python",
            "edit-distance",
            remedy="The dependency-free Python fallback is always available.",
        ),
    )


_DEFAULT_COMPONENT_REGISTRY: CapabilityRegistry | None = None


def default_capability_registry(*, copy: bool = True) -> CapabilityRegistry:
    """Return the built-in component readiness registry.

    ``copy=True`` (default) prevents one caller's registrations from mutating
    process-wide discovery for unrelated callers.
    """
    global _DEFAULT_COMPONENT_REGISTRY  # ruff: ignore[global-statement]
    if _DEFAULT_COMPONENT_REGISTRY is None:
        _DEFAULT_COMPONENT_REGISTRY = CapabilityRegistry(_builtin_capability_specs())
    return _DEFAULT_COMPONENT_REGISTRY.copy() if copy else _DEFAULT_COMPONENT_REGISTRY


def capability_report(
    name: str,
    *,
    selected: bool = False,
    active: bool = False,
    registry: CapabilityRegistry | None = None,
) -> CapabilityReport:
    """Probe one registered component capability."""
    reg = registry or default_capability_registry(copy=False)
    return reg.probe(name, selected=selected, active=active)


def component_capabilities(
    names: Iterable[str] | None = None,
    *,
    role: str | None = None,
    selected: Iterable[str] = (),
    active: Iterable[str] = (),
    registry: CapabilityRegistry | None = None,
) -> dict[str, dict[str, Any]]:
    """Return JSON-compatible readiness records for components.

    Unlike :func:`capability_snapshot`, this API distinguishes package
    installation from asset readiness and contextual selection/activity.
    ``role`` provides discoverability without requiring callers to know every
    concrete capability identifier. When both ``names`` and ``role`` are given,
    the role filters the explicitly requested names.
    """
    reg = registry or default_capability_registry(copy=False)
    if names is None:
        wanted: Iterable[str] | None = (
            reg.names(role=role) if role is not None else None
        )
    elif role is None:
        wanted = names
    else:
        wanted = tuple(name for name in names if reg.get(name).role == role)
    return {
        name: report.to_dict()
        for name, report in reg.snapshot(
            wanted,
            selected=selected,
            active=active,
        ).items()
    }
