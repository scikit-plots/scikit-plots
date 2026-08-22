# scikitplot/corpus/_retrieval.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Operation outcomes for retrieval: :class:`RetrievalResponse` and per-leg status.

A retrieval operation can run several *independent evidence paths* -- lexical,
dense, graph -- and any one of them can fail while the others succeed.  This
module supplies the envelope that reports that, so a caller can tell a complete
result from a partial one.

Notes
-----
**User-focused.**  ``index.search(...)`` returns a :class:`RetrievalResponse`.
It behaves like the list of hits it replaced -- iterate it, take its length,
index into it -- and additionally answers *how the search went*::

    response = index.search("query", config=cfg)
    for hit in response:  # same as before
        ...
    if response.status is RetrievalStatus.DEGRADED:
        for record in response.error_records():
            print(record)  # which leg did not run, and why

**Developer-focused.**  This type exists because of a measured defect
(F-R09-01).  A hybrid search invoked without a query embedding used to skip its
dense leg, fuse the lexical leg alone, and return the result **still labelled**
``match_mode="hybrid"``.  On a three-document corpus that meant 2 hits instead
of 3, with every score exactly halved by the missing ``hybrid_alpha``
contribution -- and no exception, no warning and no status.

That is the campaign's recurring shape: an operation completes, returns
plausible output, and does not report that it ran on partial evidence.  The same
shape was found in filter handling (F-R07-01) and in the document hierarchy
(F-R08-02).  The envelope is the single mechanism that closes it.

The overall :attr:`RetrievalResponse.status` is **derived**, never set by hand,
so the rule "``EMPTY`` requires that every requested leg actually ran" is
mechanically enforced rather than left to convention.

See Also
--------
scikitplot.corpus._diagnostics.ErrorRecord : the diagnostic each degraded leg carries.
"""

from __future__ import annotations

import dataclasses
from enum import unique
from typing import Any, Iterator

from ._diagnostics import ErrorRecord
from ._schema import _StrEnumBase

__all__: list[str] = [
    "LegKind",
    "LegOutcome",
    "LegStatus",
    "RetrievalResponse",
    "RetrievalStatus",
]


@unique
class RetrievalStatus(_StrEnumBase):
    """Outcome of a retrieval *operation* as a whole.

    Examples
    --------
    >>> RetrievalStatus.DEGRADED == "degraded"
    True
    """

    SUCCESS = "success"
    """Every requested leg ran and at least one hit was found."""

    EMPTY = "empty"
    """Every requested leg ran fully and correctly found zero hits.

    This value is only reachable when *all* requested legs actually ran.  A
    query that returned nothing because a leg could not run is ``DEGRADED``, not
    ``EMPTY`` -- that distinction is the whole point of the envelope.
    """

    DEGRADED = "degraded"
    """At least one requested leg did not run, or ran at reduced capability.

    Hits may be incomplete.  A ``DEGRADED`` response always carries at least one
    :class:`~scikitplot.corpus._diagnostics.ErrorRecord`; a degradation with no
    explanation is invalid.
    """

    FAILED = "failed"
    """The operation could not be performed at all."""

    CANCELLED = "cancelled"
    """Aborted by budget, deadline or explicit cancellation."""


@unique
class LegStatus(_StrEnumBase):
    """Outcome of one evidence path.

    Notes
    -----
    ``SKIPPED`` exists only at the leg level and deliberately has no
    operation-level counterpart.  A leg the caller never requested is not a
    failure and must not force ``DEGRADED``; a leg that *was* requested and could
    not run must.  Without the distinction, either every non-hybrid search would
    report ``DEGRADED``, or F-R09-01 would stay invisible.
    """

    SUCCESS = "success"
    """The leg ran and contributed hits."""

    EMPTY = "empty"
    """The leg ran fully and contributed no hits."""

    DEGRADED = "degraded"
    """The leg ran at reduced capability."""

    FAILED = "failed"
    """The leg was requested but could not run."""

    SKIPPED = "skipped"
    """The leg was not requested by this query."""


@unique
class LegKind(_StrEnumBase):
    """Which evidence path a leg represents."""

    LEXICAL = "lexical"
    """Keyword / BM25 matching."""

    DENSE = "dense"
    """Vector similarity over embeddings."""

    GRAPH = "graph"
    """Relationship traversal."""

    METADATA = "metadata"
    """Attribute filtering."""


@dataclasses.dataclass(frozen=True)
class LegOutcome:
    """How one evidence path fared.

    Parameters
    ----------
    leg : LegKind
        Which evidence path this describes.
    status : LegStatus
        Outcome for this leg.
    hit_count : int, optional
        Hits contributed *before* fusion.
    generation : int or None, optional
        Index generation the leg ran against, for provenance.
    backend : str or None, optional
        Backend that served the leg, when applicable.
    error : ErrorRecord or None, optional
        Required when ``status`` is ``DEGRADED`` or ``FAILED``.

    Raises
    ------
    ValueError
        If ``status`` is ``DEGRADED`` or ``FAILED`` and ``error`` is ``None``.

    Notes
    -----
    **Developer.**  The ``error``-required invariant is enforced in
    ``__post_init__`` rather than documented, because an unexplained degradation
    is exactly as unhelpful as no degradation signal at all.
    """

    leg: LegKind
    status: LegStatus
    hit_count: int = 0
    generation: int | None = None
    backend: str | None = None
    error: ErrorRecord | None = None

    def __post_init__(self) -> None:
        """Enforce that a degraded or failed leg explains itself."""
        if self.status in (LegStatus.DEGRADED, LegStatus.FAILED) and self.error is None:
            raise ValueError(
                f"LegOutcome({self.leg.value}) is {self.status.value} but carries "
                "no ErrorRecord; a degradation without an explanation is invalid."
            )

    @property
    def ran(self) -> bool:
        """Whether the leg actually executed."""
        return self.status in (LegStatus.SUCCESS, LegStatus.EMPTY)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "leg": self.leg.value,
            "status": self.status.value,
            "hit_count": self.hit_count,
            "generation": self.generation,
            "backend": self.backend,
            "error": self.error.to_dict() if self.error is not None else None,
        }


@dataclasses.dataclass(frozen=True)
class RetrievalResponse:
    """Hits plus the story of how they were obtained.

    Parameters
    ----------
    hits : list
        Fused, ranked hits.
    legs : list of LegOutcome, optional
        One entry per evidence path considered, including skipped ones.
    query : str or None, optional
        The query text, for provenance.
    status : RetrievalStatus or None, optional
        Normally left ``None`` so it is **derived** from ``legs`` and ``hits``.
        Pass a value only for terminal outcomes the legs cannot express, such as
        ``CANCELLED``.

    Notes
    -----
    **User.**  This object stands in for the plain list it replaced: ``len()``,
    iteration and indexing all address the hits.

    **Developer.**  Deriving :attr:`status` rather than accepting it is what
    makes the ``EMPTY``-requires-all-legs-ran rule enforceable.  See
    :meth:`_derive_status`.

    Examples
    --------
    >>> from scikitplot.corpus._diagnostics import ErrorCategory, ErrorRecord
    >>> ok = LegOutcome(LegKind.LEXICAL, LegStatus.EMPTY)
    >>> RetrievalResponse(hits=[], legs=[ok]).status
    <RetrievalStatus.EMPTY: 'empty'>

    A leg that could not run makes an empty result ``DEGRADED``, not ``EMPTY``:

    >>> bad = LegOutcome(
    ...     LegKind.DENSE,
    ...     LegStatus.FAILED,
    ...     error=ErrorRecord(
    ...         code="NO_QUERY_EMBEDDING",
    ...         category=ErrorCategory.CAPABILITY,
    ...         message="no query embedding supplied",
    ...     ),
    ... )
    >>> RetrievalResponse(hits=[], legs=[ok, bad]).status
    <RetrievalStatus.DEGRADED: 'degraded'>
    """

    hits: list = dataclasses.field(default_factory=list)
    legs: list[LegOutcome] = dataclasses.field(default_factory=list)
    query: str | None = None
    status: RetrievalStatus | None = None

    def __post_init__(self) -> None:
        """Derive :attr:`status` unless a terminal value was supplied."""
        if self.status is None:
            object.__setattr__(self, "status", self._derive_status())

    def _derive_status(self) -> RetrievalStatus:
        """Compute the operation status from the leg outcomes.

        Returns
        -------
        RetrievalStatus

        Notes
        -----
        The rule, in order:

        1. every requested leg ``FAILED``            -> ``FAILED``
        2. any requested leg ``FAILED``/``DEGRADED`` -> ``DEGRADED``
        3. all requested legs ran, zero hits         -> ``EMPTY``
        4. otherwise                                 -> ``SUCCESS``

        Legs with status ``SKIPPED`` are not "requested" and are excluded from
        every clause.
        """
        requested = [leg for leg in self.legs if leg.status is not LegStatus.SKIPPED]
        if requested and all(leg.status is LegStatus.FAILED for leg in requested):
            return RetrievalStatus.FAILED
        if any(
            leg.status in (LegStatus.FAILED, LegStatus.DEGRADED) for leg in requested
        ):
            return RetrievalStatus.DEGRADED
        if not self.hits:
            return RetrievalStatus.EMPTY
        return RetrievalStatus.SUCCESS

    # -- sequence-like access over the hits ----------------------------------

    def __iter__(self) -> Iterator:
        """Iterate the hits."""
        return iter(self.hits)

    def __len__(self) -> int:
        """Return the number of hits."""
        return len(self.hits)

    def __getitem__(self, index):
        """Index into the hits."""
        return self.hits[index]

    def __bool__(self) -> bool:
        """Truthy when there are hits.

        Notes
        -----
        **Developer.**  Deliberately mirrors the list this replaced, so
        ``if results:`` keeps its meaning.  Callers that care about *why* a
        response is falsy must consult :attr:`status`, which is the point.
        """
        return bool(self.hits)

    # -- diagnostics ----------------------------------------------------------

    def leg(self, kind: LegKind | str) -> LegOutcome | None:
        """Return the outcome for one evidence path, if present."""
        kind = LegKind(kind)
        for outcome in self.legs:
            if outcome.leg is kind:
                return outcome
        return None

    def error_records(self) -> list[ErrorRecord]:
        """Return every diagnostic attached to a leg."""
        return [leg.error for leg in self.legs if leg.error is not None]

    @property
    def degraded_legs(self) -> list[LegOutcome]:
        """Legs that were requested but did not run normally."""
        return [
            leg
            for leg in self.legs
            if leg.status in (LegStatus.FAILED, LegStatus.DEGRADED)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of the outcome (not the hits)."""
        return {
            "status": self.status.value,
            "query": self.query,
            "hit_count": len(self.hits),
            "legs": [leg.to_dict() for leg in self.legs],
        }
