# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Neutral retrieval outcome envelope for the MCP retrieval tier.

This module exists to satisfy one invariant required by the MCP review campaign
(run M04)::

    FAILED retrieval != EMPTY retrieval

Before this envelope, every retriever in :mod:`scikitplot.mcp` returned a bare
``list[RetrievedChunk]``. A list cannot distinguish *"the corpus contains no
match"* from *"every backend failed"*, so a total retrieval failure reached the
wire as ``"No matching documentation was found for this query."`` — a confident,
wrong answer.

Notes
-----
**User-focused.** :class:`RetrievalOutcome` *is* a :class:`list`. Existing code
that calls ``len()``, iterates, indexes, slices or tests ``if results:`` keeps
working unchanged; the outcome simply carries extra provenance alongside the
hits. Nothing needs to be rewritten to adopt it.

**Developer-focused.** Subclassing :class:`list` rather than wrapping it is a
deliberate minimal-impact choice. ``DocsRetriever.search`` is annotated
``-> list[RetrievedChunk]`` and that annotation stays *true*, so the Protocol
does not change and no caller breaks. This mirrors the decision Corpus already
made for :class:`~scikitplot.corpus.RetrievalResponse`, whose docstring records
that it "stands in for the plain list it replaced".

**Vocabulary is consumed, not redefined.** The status strings here are exactly
the values of :class:`~scikitplot.corpus.RetrievalStatus`
(``success``/``empty``/``degraded``/``failed``/``cancelled``), and
:func:`to_corpus_status` resolves them to the real Corpus enum on demand. The
values are duplicated as plain strings *only* because the MCP retrieval tier must
import without ``scikitplot.corpus`` installed — the module-scope boundary rule
enforced by ``_maintenance/check_trackers.py``. :func:`to_corpus_status` performs
the import lazily, so the boundary holds and the two vocabularies cannot drift
without :func:`assert_matches_corpus` failing.

See Also
--------
scikitplot.corpus.RetrievalStatus : The authoritative status vocabulary.
scikitplot.corpus.RetrievalResponse : The Corpus-side envelope this mirrors.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CANCELLED",
    "DEGRADED",
    "EMPTY",
    "FAILED",
    "SUCCESS",
    "LegRecord",
    "RetrievalOutcome",
    "assert_matches_corpus",
    "to_corpus_status",
]

#: Mirror of :class:`scikitplot.corpus.RetrievalStatus` values. See module notes
#: for why these are strings here rather than an imported enum.
SUCCESS = "success"
EMPTY = "empty"
DEGRADED = "degraded"
FAILED = "failed"
CANCELLED = "cancelled"

_TERMINAL = (SUCCESS, EMPTY, DEGRADED, FAILED, CANCELLED)


class LegRecord:
    """
    How one retrieval leg fared.

    Parameters
    ----------
    leg : str
        Identifier for the evidence path, e.g. ``"dense"`` or ``"lexical"``.
    status : str
        One of :data:`SUCCESS`, :data:`EMPTY`, :data:`DEGRADED`, :data:`FAILED`
        or :data:`CANCELLED`.
    hit_count : int, optional
        Hits contributed by this leg *before* fusion.
    error : str or None, optional
        Human-readable reason. Required when ``status`` is :data:`DEGRADED` or
        :data:`FAILED`.

    Raises
    ------
    ValueError
        If ``status`` is not a known status, or if a degraded/failed leg is
        recorded without an ``error``.

    Notes
    -----
    **Developer-focused.** The error-required invariant is enforced here rather
    than merely documented, matching
    :class:`scikitplot.corpus.LegOutcome`: an unexplained degradation is as
    unhelpful as no degradation signal at all.
    """

    __slots__ = ("error", "hit_count", "leg", "status")

    def __init__(
        self,
        leg: str,
        status: str,
        hit_count: int = 0,
        error: str | None = None,
    ) -> None:
        if status not in _TERMINAL:
            raise ValueError(
                f"unknown retrieval leg status {status!r}; expected one of {_TERMINAL}"
            )
        if status in (DEGRADED, FAILED) and not error:
            raise ValueError(
                f"leg {leg!r} reported {status!r} without an error explanation"
            )
        self.leg = str(leg)
        self.status = status
        self.hit_count = int(hit_count)
        self.error = error

    def as_dict(self) -> dict[str, Any]:
        """
        Return a JSON-safe mapping describing this leg.

        Returns
        -------
        dict
            Keys ``leg``, ``status``, ``hit_count`` and — only when set —
            ``error``.
        """
        record: dict[str, Any] = {
            "leg": self.leg,
            "status": self.status,
            "hit_count": self.hit_count,
        }
        if self.error:
            record["error"] = self.error
        return record

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LegRecord(leg={self.leg!r}, status={self.status!r})"


class RetrievalOutcome(list):
    """
    Ranked hits plus the story of how they were obtained.

    Parameters
    ----------
    hits : iterable, optional
        The fused, ranked hits. Copied into this list.
    legs : list of LegRecord, optional
        One entry per evidence path considered.
    status : str or None, optional
        Normally left ``None`` so it is **derived** from ``legs`` and ``hits``.
        Pass a value only for terminal outcomes the legs cannot express, such as
        :data:`CANCELLED`.

    Notes
    -----
    **User-focused.** This is a real :class:`list` of hits. ``len(outcome)``,
    ``for hit in outcome``, ``outcome[0]`` and ``if outcome:`` all behave exactly
    as they did before. Read :attr:`status` when you need to know *why* the list
    is the length it is.

    **Developer-focused.** Deriving :attr:`status` rather than accepting it is
    what makes the "``EMPTY`` requires every leg to have run" rule enforceable.
    The derivation order mirrors
    :meth:`scikitplot.corpus.RetrievalResponse._derive_status`:

    1. every recorded leg :data:`FAILED` -> :data:`FAILED`
    2. any leg :data:`FAILED` or :data:`DEGRADED` -> :data:`DEGRADED`
    3. all legs ran, zero hits -> :data:`EMPTY`
    4. otherwise -> :data:`SUCCESS`

    With no legs recorded the outcome degrades to the historical behaviour
    (:data:`EMPTY` when there are no hits, :data:`SUCCESS` otherwise), so a
    retriever that has not yet been updated keeps its previous meaning.

    Examples
    --------
    >>> RetrievalOutcome([]).status
    'empty'
    >>> failed = LegRecord("dense", FAILED, error="index unavailable")
    >>> RetrievalOutcome([], legs=[failed]).status
    'failed'
    >>> ok = LegRecord("lexical", EMPTY)
    >>> RetrievalOutcome([], legs=[ok, failed]).status
    'degraded'
    """

    __slots__ = ("legs", "status")

    def __init__(
        self,
        hits: Any = (),
        *,
        legs: list[LegRecord] | None = None,
        status: str | None = None,
    ) -> None:
        super().__init__(hits or ())
        self.legs: list[LegRecord] = list(legs or ())
        if status is not None and status not in _TERMINAL:
            raise ValueError(
                f"unknown retrieval status {status!r}; expected one of {_TERMINAL}"
            )
        self.status: str = status if status is not None else self._derive_status()

    def _derive_status(self) -> str:
        """
        Compute the operation status from the recorded leg outcomes.

        Returns
        -------
        str
            One of the module-level status constants.
        """
        if self.legs:
            if all(leg.status == FAILED for leg in self.legs):
                return FAILED
            if any(leg.status in (FAILED, DEGRADED) for leg in self.legs):
                return DEGRADED
        return EMPTY if not len(self) else SUCCESS

    @property
    def failed(self) -> bool:
        """bool: ``True`` when no evidence path produced a usable answer."""
        return self.status == FAILED

    @property
    def degraded(self) -> bool:
        """bool: ``True`` when at least one evidence path did not run cleanly."""
        return self.status == DEGRADED

    def errors(self) -> list[str]:
        """
        Return the explanations from every leg that did not run cleanly.

        Returns
        -------
        list of str
        """
        return [leg.error for leg in self.legs if leg.error]

    def as_dict(self) -> dict[str, Any]:
        """
        Return a JSON-safe mapping describing the outcome, without the hits.

        Returns
        -------
        dict
            Keys ``status`` and ``legs``.
        """
        return {"status": self.status, "legs": [leg.as_dict() for leg in self.legs]}


def status_of(chunks: Any) -> str:
    """
    Return the retrieval status for ``chunks``, tolerating a plain list.

    Parameters
    ----------
    chunks : iterable or None
        Either a :class:`RetrievalOutcome` or any legacy sequence of hits.

    Returns
    -------
    str
        The recorded status when ``chunks`` is a :class:`RetrievalOutcome`;
        otherwise :data:`EMPTY` or :data:`SUCCESS` inferred from length, which
        reproduces the behaviour that predates this module.

    Notes
    -----
    **Developer-focused.** This keeps every existing caller correct without
    modification: a retriever that still returns a bare list is treated exactly
    as it was before.
    """
    status = getattr(chunks, "status", None)
    if isinstance(status, str) and status in _TERMINAL:
        return status
    try:
        return SUCCESS if len(chunks or ()) else EMPTY
    except TypeError:  # pragma: no cover - non-sized iterable
        return SUCCESS


def to_corpus_status(status: str) -> Any:
    """
    Resolve a status string to :class:`scikitplot.corpus.RetrievalStatus`.

    Parameters
    ----------
    status : str
        One of the module-level status constants.

    Returns
    -------
    scikitplot.corpus.RetrievalStatus

    Raises
    ------
    RuntimeError
        If ``scikitplot.corpus`` is not installed.
    ValueError
        If ``status`` is not a known status value.

    Notes
    -----
    **Developer-focused.** The import is performed here, at call time, so that
    importing :mod:`scikitplot.mcp` never pulls in ``scikitplot.corpus``. That
    module-scope boundary is enforced by
    ``scikitplot/mcp/_maintenance/check_trackers.py``.
    """
    if status not in _TERMINAL:
        raise ValueError(
            f"unknown retrieval status {status!r}; expected one of {_TERMINAL}"
        )
    try:
        from scikitplot.corpus import RetrievalStatus  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "scikitplot.corpus is required to map MCP retrieval statuses onto "
            "RetrievalStatus; install the corpus extras "
            "(pip install scikit-plots[corpus])."
        ) from exc
    return RetrievalStatus(status)


def assert_matches_corpus() -> None:
    """
    Verify this module's vocabulary still matches Corpus's.

    Raises
    ------
    RuntimeError
        If ``scikitplot.corpus`` is not installed.
    AssertionError
        If Corpus's :class:`~scikitplot.corpus.RetrievalStatus` has gained or
        renamed a member relative to the constants defined here.

    Notes
    -----
    **Developer-focused.** This is the drift gate for the one piece of duplicated
    vocabulary in the MCP tier. It is called by the test suite whenever
    ``scikitplot.corpus`` is importable, so the strings above cannot silently
    diverge from the enum they mirror.
    """
    try:
        from scikitplot.corpus import RetrievalStatus  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "scikitplot.corpus is required to verify the retrieval status "
            "vocabulary; install the corpus extras."
        ) from exc
    corpus_values = {member.value for member in RetrievalStatus}
    mirrored = set(_TERMINAL)
    if corpus_values != mirrored:
        raise AssertionError(
            "MCP retrieval status vocabulary drifted from "
            f"scikitplot.corpus.RetrievalStatus: corpus={sorted(corpus_values)} "
            f"mcp={sorted(mirrored)}"
        )
