"""
Capability truth for the optional tiers of :mod:`scikitplot.cleanprompt`.

Reports whether an optional dependency tier is usable, using the project's
canonical seven-state vocabulary, and never importing the dependency it reports
on.

Notes
-----
**User notes.** :func:`capabilities` answers "what can this installation do?"
without side effects. It is safe to call at import time, in a notebook, or in a
``--version``-style banner.

**Developer notes.** Three rules govern this module, each of which exists
because collapsing it produced a real defect elsewhere in this project.

1. ``BROKEN`` is not ``ABSENT``. A distribution that is installed but raises on
   import (corrupt, mis-linked, ABI-incompatible) must not report the same as
   one that was never installed. A boolean cannot express this.
2. Presence is probed with :func:`importlib.metadata.version`, not
   :func:`importlib.util.find_spec`. ``find_spec`` answers "is there a module of
   this name on the path", which is true for a shadowing stub, an empty
   namespace package, and a half-removed install. It also yields no version, so
   a compatibility range cannot be checked.
3. The vocabulary is *consumed by value*, not imported. ``CapabilityStatus`` is
   owned by ``scikitplot.corpus``; this submodule carries a local ``str``-enum
   copy so that it stays independently importable, in line with this project's
   submodule-independence rule. Do not define a parallel vocabulary with
   different member names.

Probing never imports the target distribution, so a call to :func:`capabilities`
cannot pull ``spacy`` into a process that only wanted the base tier.

See Also
--------
scikitplot.cleanprompt._exceptions.CapabilityError : Raised from these reports.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import NamedTuple

from ._exceptions import CapabilityError

__all__ = [
    "TIERS",
    "CapabilityReport",
    "CapabilityStatus",
    "capabilities",
    "probe",
    "require",
]


class CapabilityStatus(str, Enum):
    """
    Seven-state usability vocabulary for an optional dependency tier.

    This is a by-value copy of the project's canonical vocabulary. Members are
    ``str`` so they serialize to JSON and compare to plain strings without
    conversion.

    Attributes
    ----------
    AVAILABLE : str
        Installed, version-compatible, usable.
    ABSENT : str
        Not installed.
    BROKEN : str
        Installed but the probe raised; installed-and-failing.
    INCOMPATIBLE : str
        Installed at a version outside the supported range.
    MISCONFIGURED : str
        Installed and compatible, but a required resource is missing (for
        example a spaCy model that has not been downloaded).
    UNREACHABLE : str
        Requires a network or service endpoint that cannot be reached.
    UNKNOWN : str
        Not determined.
    """

    AVAILABLE = "AVAILABLE"
    ABSENT = "ABSENT"
    BROKEN = "BROKEN"
    INCOMPATIBLE = "INCOMPATIBLE"
    MISCONFIGURED = "MISCONFIGURED"
    UNREACHABLE = "UNREACHABLE"
    UNKNOWN = "UNKNOWN"


class _TierSpec(NamedTuple):
    """Static description of one optional tier."""

    distribution: str
    minimum: tuple[int, ...]
    below: tuple[int, ...]
    extra: str
    purpose: str


#: Static tier table. ``minimum`` is inclusive, ``below`` is exclusive.
#: Ranges rather than pins: a library must remain usable across the span its
#: consumers legitimately install. Each upper bound is a major boundary at which
#: the distribution has historically made breaking changes.
TIERS: dict[str, _TierSpec] = {
    "ner": _TierSpec(
        distribution="spacy",
        minimum=(3, 4),
        below=(5,),
        extra="cleanprompt",
        purpose="named-entity detection",
    ),
    "nltk": _TierSpec(
        distribution="nltk",
        minimum=(3, 6),
        below=(4,),
        extra="cleanprompt",
        purpose="named-entity detection without spaCy (English only)",
    ),
    "web": _TierSpec(
        distribution="flask",
        minimum=(2, 2),
        below=(4,),
        extra="cleanprompt",
        purpose="the local web interface",
    ),
    "crypto": _TierSpec(
        distribution="cryptography",
        minimum=(41,),
        below=(50,),
        extra="cleanprompt",
        # Not "vault encryption" any more: the base tier encrypts vaults with
        # the standard library alone. This tier adds the reviewed AES
        # primitive for anyone who prefers it, which is a different and much
        # smaller claim than the one this row used to make.
        purpose="the Fernet vault cipher (--cipher fernet); encryption itself needs no tier",
    ),
}


class CapabilityReport(NamedTuple):
    """
    Outcome of probing one tier.

    Attributes
    ----------
    tier : str
        Tier name, one of the keys of :data:`TIERS`.
    status : CapabilityStatus
        Seven-state result.
    distribution : str
        The distribution that was probed.
    version : str or None
        Installed version, when it could be determined.
    supported : str
        Human-readable supported range.
    detail : str
        Why this status was reached.
    install_hint : str
        The exact command that would make the tier available.
    """

    tier: str
    status: CapabilityStatus
    distribution: str
    version: str | None
    supported: str
    detail: str
    install_hint: str

    @property
    def available(self) -> bool:
        """bool: ``True`` only when :attr:`status` is ``AVAILABLE``."""
        return self.status is CapabilityStatus.AVAILABLE


def _parse_release(version: str) -> tuple[int, ...] | None:
    """
    Parse the numeric release prefix of a version string.

    Parameters
    ----------
    version : str
        A version string such as ``"3.7.2"``, ``"4.0.0rc1"`` or ``"2.2"``.

    Returns
    -------
    tuple of int or None
        The leading numeric components, or ``None`` when no leading numeric
        component could be read.

    Notes
    -----
    **Developer notes.** Deliberately minimal and total: it reads the leading
    dot-separated integer run and stops at the first component that is not a
    plain integer. It is used only for range comparison against the small,
    fully numeric bounds in :data:`TIERS`, so PEP 440 epochs, local versions and
    pre-release ordering do not affect the decision. A pre-release such as
    ``"4.0.0rc1"`` reads as ``(4, 0, 0)`` and is therefore treated as the
    release it precedes, which is the conservative reading for an upper bound.
    """
    parts: list = []
    for chunk in version.split("."):
        head = ""
        for char in chunk:
            if char.isdigit():
                head += char
            else:
                break
        if not head:
            break
        parts.append(int(head))
        if head != chunk:
            break
    if not parts:
        return None
    return tuple(parts)


def _installed_version(distribution: str) -> str | None:
    """
    Return the installed version of ``distribution``, or ``None``.

    Raises
    ------
    Exception
        Propagates anything other than the "not installed" signal, so the caller
        can classify it as ``BROKEN`` rather than ``ABSENT``.
    """
    if sys.version_info >= (3, 8):  # ruff: ignore[outdated-version-block]
        from importlib.metadata import (  # ruff: ignore[import-outside-top-level]
            PackageNotFoundError,
            version,
        )
    else:  # pragma: no cover - the package floor is 3.8
        from importlib_metadata import (  # type: ignore[]  # ruff: ignore[import-outside-top-level]
            PackageNotFoundError,
            version,
        )

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def probe(tier: str) -> CapabilityReport:
    """
    Report whether one optional tier is usable.

    Parameters
    ----------
    tier : str
        Tier name; one of the keys of :data:`TIERS`.

    Returns
    -------
    CapabilityReport
        The seven-state report. Never raises for an unusable tier.

    Raises
    ------
    KeyError
        If ``tier`` is not a known tier name.

    Notes
    -----
    **Developer notes.** The target distribution is never imported here. Metadata
    lookup is a filesystem operation; it answers the presence and version
    questions without executing third-party code, which is what keeps
    ``import scikitplot.cleanprompt`` free of optional dependencies.

    Examples
    --------
    >>> report = probe("ner")
    >>> report.tier
    'ner'
    >>> report.status in set(CapabilityStatus)
    True
    """
    spec = TIERS[tier]
    supported = "{}>={},<{}".format(
        spec.distribution,
        ".".join(str(part) for part in spec.minimum),
        ".".join(str(part) for part in spec.below),
    )
    hint = f'pip install "{supported}"'

    try:
        found = _installed_version(spec.distribution)
    except Exception as exc:  # noqa: BLE001 - classification, not suppression
        return CapabilityReport(
            tier=tier,
            status=CapabilityStatus.BROKEN,
            distribution=spec.distribution,
            version=None,
            supported=supported,
            detail=(
                f"metadata lookup for {spec.distribution!r} raised {type(exc).__name__}: {exc}"
            ),
            install_hint=hint,
        )

    if found is None:
        return CapabilityReport(
            tier=tier,
            status=CapabilityStatus.ABSENT,
            distribution=spec.distribution,
            version=None,
            supported=supported,
            detail=f"{spec.distribution} is not installed",
            install_hint=hint,
        )

    release = _parse_release(found)
    if release is None:
        return CapabilityReport(
            tier=tier,
            status=CapabilityStatus.UNKNOWN,
            distribution=spec.distribution,
            version=found,
            supported=supported,
            detail=f"installed version {found!r} has no numeric release",
            install_hint=hint,
        )

    if release < spec.minimum or release >= spec.below:
        return CapabilityReport(
            tier=tier,
            status=CapabilityStatus.INCOMPATIBLE,
            distribution=spec.distribution,
            version=found,
            supported=supported,
            detail=f"installed {found} is outside {supported}",
            install_hint=hint,
        )

    return CapabilityReport(
        tier=tier,
        status=CapabilityStatus.AVAILABLE,
        distribution=spec.distribution,
        version=found,
        supported=supported,
        detail=f"{spec.distribution} {found} satisfies {supported}",
        install_hint=hint,
    )


def capabilities() -> dict[str, CapabilityReport]:
    """
    Probe every optional tier.

    Returns
    -------
    dict of str to CapabilityReport
        One report per tier, keyed by tier name, in the order of :data:`TIERS`.

    Examples
    --------
    >>> sorted(capabilities())
    ['crypto', 'ner', 'nltk', 'web']
    """
    return {name: probe(name) for name in TIERS}


def require(tier: str) -> CapabilityReport:
    """
    Return the report for ``tier``, raising when it is not usable.

    Parameters
    ----------
    tier : str
        Tier name; one of the keys of :data:`TIERS`.

    Returns
    -------
    CapabilityReport
        The report, guaranteed to be ``AVAILABLE``.

    Raises
    ------
    CapabilityError
        If the tier is not usable. The message names the tier's purpose, the
        observed status, and the exact install command.

    Notes
    -----
    **Developer notes.** Call this *before* importing the tier's dependency, not
    after. An actionable message placed below a module-scope third-party import
    is unreachable, because the import raises first.
    """
    report = probe(tier)
    if report.available:
        return report
    raise CapabilityError(
        f"{TIERS[tier].purpose.capitalize()} requires {report.supported} ({report.status.value}: {report.detail}); install it with: {report.install_hint}",
        tier=tier,
        status=report.status.value,
        install_hint=report.install_hint,
    )
