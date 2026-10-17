# scikitplot/corpus/_execution.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Retrieval as typed, bounded stages: :class:`RetrievalExecutionPlan` (R4-3).

A plan runs a query through five stages, always in this order::

    transform -> retrieve -> fuse -> rerank -> compress

Each stage is optional except ``retrieve``, and each reports a
:class:`StageOutcome`: what ran, on how many items, how many calls and seconds
it took, under which budget, and why it did not finish when it did not.

Notes
-----
**User.**

.. code-block:: python

    plan = RetrievalExecutionPlan(
        transform=lambda q: [q, q + " definition"],  # multi-query
        rerank=my_cross_encoder,  # (query, hits) -> scores
        compress=CharBudgetCompressor(4000),  # context within 4000 characters
        budgets={"retrieve": StageBudget(max_calls=4, max_seconds=5.0)},
        deadline_seconds=10.0,  # no stage starts after 10 s
    )
    result = plan.run(index, "what is a span?")
    for hit in result:  # ranked hits, like a RetrievalResponse
        ...
    result.status  # derived, as for a RetrievalResponse
    result.stage("rerank").seconds
    result.passages  # the compressed context, as spans of each hit's text

``evaluate_retrieval(plan.bind(index), queries, qrels)`` measures a plan with
the same harness as a bare index.

**Developer.** Three rules from the earlier runs shape this module.

* *No new retrieval path.* ``retrieve`` calls ``index.search`` once per query
  and keeps every response; fusion across queries is the one
  :class:`~scikitplot.corpus.FusionPolicy` (R2-1). An empty plan returns
  exactly the hits ``index.search`` returns.
* *No silent partial result.* A stage that could not run, ran on part of its
  input, or stopped at a budget says so with an
  :class:`~scikitplot.corpus.ErrorRecord`, and the result's status is derived
  from the stages and the responses, never set by hand (the rule of
  :class:`~scikitplot.corpus.RetrievalResponse`).
* *Budgets are arithmetic.* A budget is checked before each call; a component
  never decides whether it may run. A call that is already running is not
  interrupted: its time is reported, and the budget stops the next call. So
  only ``retrieve``, which makes one call per query, takes a
  :class:`StageBudget`; the plan's ``deadline_seconds`` stops any later stage
  from starting. A single call is bounded by the component's own timeout.

Compression returns character spans of each hit's ``doc.text``, never
rewritten text, so a passage stays a citation of its source (R3-3).
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import operator
import time
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from enum import unique
from typing import Any, Callable, Iterator

from ._diagnostics import ErrorCategory, ErrorRecord
from ._fusion import FusionPolicy
from ._retrieval import RetrievalStatus
from ._retrievers import _fusion_key
from ._schema import _StrEnumBase

__all__: list[str] = [
    "CharBudgetCompressor",
    "ContextPassage",
    "ExecutionResult",
    "PlannedHit",
    "RetrievalExecutionPlan",
    "StageBudget",
    "StageKind",
    "StageOutcome",
    "StageStatus",
]

#: Accepted values of ``RetrievalExecutionPlan.on_error``.
ON_ERROR: tuple[str, ...] = ("degrade", "raise")


@unique
class StageKind(_StrEnumBase):
    """One stage of a plan, in execution order."""

    TRANSFORM = "transform"
    """Query to one or more queries (rewrite, multi-query, decomposition)."""

    RETRIEVE = "retrieve"
    """``index.search`` for each query."""

    FUSE = "fuse"
    """The queries' rankings fused into one (only with two or more queries)."""

    RERANK = "rerank"
    """New scores for the first ``rerank_depth`` hits."""

    COMPRESS = "compress"
    """Spans of each hit's text kept as context."""


#: Execution order of the stages.
STAGE_ORDER: tuple[StageKind, ...] = tuple(StageKind)


@unique
class StageStatus(_StrEnumBase):
    """Outcome of one stage."""

    SUCCESS = "success"
    """The stage ran on all of its input and produced output."""

    EMPTY = "empty"
    """The stage ran on all of its input and produced nothing."""

    DEGRADED = "degraded"
    """The stage ran on part of its input, or produced part of its output."""

    FAILED = "failed"
    """The stage was configured but did not produce output."""

    SKIPPED = "skipped"
    """The stage was not configured, or had nothing to do."""

    CANCELLED = "cancelled"
    """A budget or the plan's deadline stopped the stage before it ran, or
    (``retrieve``) the index cancelled every query that ran."""


_EXPLAINED = (StageStatus.DEGRADED, StageStatus.FAILED, StageStatus.CANCELLED)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value!r}.")
    return value


def _positive_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    return value


@dataclasses.dataclass(frozen=True)
class StageBudget:
    """Deterministic limits on the ``retrieve`` stage.

    Parameters
    ----------
    max_calls : int or None, default=None
        Queries ``retrieve`` may start (one ``index.search`` call each).
        ``None`` is no limit.
    max_seconds : float or None, default=None
        Seconds after which ``retrieve`` starts no further query. A running
        call is not interrupted: its time is reported, and an overrun is in
        the stage's details. ``None`` is no limit.

    Raises
    ------
    TypeError
        If a limit has the wrong type (``bool`` included).
    ValueError
        If a limit is not positive (or not finite).
    """

    max_calls: int | None = None
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate the limits."""
        if self.max_calls is not None:
            object.__setattr__(
                self,
                "max_calls",
                _positive_int(self.max_calls, "StageBudget.max_calls"),
            )
        if self.max_seconds is not None:
            object.__setattr__(
                self,
                "max_seconds",
                _positive_seconds(self.max_seconds, "StageBudget.max_seconds"),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {"max_calls": self.max_calls, "max_seconds": self.max_seconds}


@dataclasses.dataclass(frozen=True)
class StageOutcome:
    """How one stage fared.

    Parameters
    ----------
    stage : StageKind
        Which stage.
    status : StageStatus
        Its outcome.
    component : str or None, default=None
        Name of the configured callable, for provenance.
    items_in, items_out : int, default=0
        Queries (``transform``, ``retrieve``) or hits (the other stages) the
        stage received and produced.
    calls : int, default=0
        Calls the stage started.
    seconds : float, default=0.0
        Time the stage took.
    budget : StageBudget or None, default=None
        The budget it ran under.
    error : ErrorRecord or None, default=None
        Required when ``status`` is ``DEGRADED``, ``FAILED`` or ``CANCELLED``.
    details : tuple of (str, object) pairs, default=()
        Stage-specific facts (queries dropped, rerank depth, ...).

    Raises
    ------
    ValueError
        If ``status`` is ``DEGRADED``, ``FAILED`` or ``CANCELLED`` and
        ``error`` is ``None``.
    """

    stage: StageKind
    status: StageStatus
    component: str | None = None
    items_in: int = 0
    items_out: int = 0
    calls: int = 0
    seconds: float = 0.0
    budget: StageBudget | None = None
    error: ErrorRecord | None = None
    details: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Enforce that a stage that did not finish explains itself."""
        if self.status in _EXPLAINED and self.error is None:
            raise ValueError(
                f"StageOutcome({self.stage.value}) is {self.status.value} but carries "
                "no ErrorRecord; an unexplained outcome is invalid."
            )

    @property
    def ran(self) -> bool:
        """Whether the stage produced its output in full."""
        return self.status in (StageStatus.SUCCESS, StageStatus.EMPTY)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "component": self.component,
            "items_in": self.items_in,
            "items_out": self.items_out,
            "calls": self.calls,
            "seconds": self.seconds,
            "budget": self.budget.to_dict() if self.budget is not None else None,
            "error": self.error.to_dict() if self.error is not None else None,
            "details": dict(self.details),
        }


@dataclasses.dataclass(frozen=True)
class PlannedHit:
    """One ranked hit of a plan, with the score of each stage that ranked it.

    Parameters
    ----------
    hit : object
        The hit as ``index.search`` returned it (for a document several
        queries found: from the first such query).
    score : float
        The value the hit is ranked by: the rerank score when it was
        reranked, else the fused score, else the search score.
    key : str
        Its identity (``doc_id``, else ``node_id``), on which fusion keys.
    fused_score : float or None, default=None
        The cross-query fusion score, when two or more queries ran.
    rerank_score : float or None, default=None
        The reranker's score, when the hit was reranked.
    query_ranks : tuple of (int or None), default=()
        Zero-based rank of the hit in each query's ranking, ``None`` where a
        query did not return it; aligned with ``ExecutionResult.queries``.
    """

    hit: Any
    score: float
    key: str
    fused_score: float | None = None
    rerank_score: float | None = None
    query_ranks: tuple[int | None, ...] = ()

    @property
    def doc(self) -> Any:
        """The matched document (``hit.doc``, else the hit itself)."""
        return getattr(self.hit, "doc", self.hit)

    @property
    def doc_id(self) -> str:
        """The hit's identity."""
        return self.key


@dataclasses.dataclass(frozen=True)
class ContextPassage:
    """The context a compressor kept from one hit.

    Parameters
    ----------
    hit : PlannedHit
        The hit.
    spans : tuple of (int, int)
        Ordered, non-overlapping character ranges of ``hit.doc.text``.

    Notes
    -----
    **User.** :attr:`text` is the spans' text joined by newlines, so every
    character of a passage is a character of its source at a known offset.
    """

    hit: PlannedHit
    spans: tuple[tuple[int, int], ...]

    @property
    def text(self) -> str:
        """The kept text: each span of the source, joined by newlines."""
        source = _doc_text(self.hit)
        return "\n".join(source[start:end] for start, end in self.spans)

    @property
    def chars(self) -> int:
        """Characters of source kept (separators not counted)."""
        return sum(end - start for start, end in self.spans)


def _doc_text(hit: PlannedHit) -> str:
    text = getattr(hit.doc, "text", None)
    if not isinstance(text, str):
        raise TypeError(
            f"hit {hit.key!r}: compression needs doc.text as a str, got "
            f"{type(text).__name__}."
        )
    return text


@dataclasses.dataclass(frozen=True)
class ExecutionResult:
    """Ranked hits plus the account of every stage.

    Parameters
    ----------
    hits : tuple of PlannedHit
        Final ranking, best first, at most the plan's ``top_k``.
    stages : tuple of StageOutcome
        One per stage, in execution order.
    queries : tuple of str
        The queries ``retrieve`` was asked to run, the original first unless
        the transform dropped it.
    responses : tuple
        ``index.search``'s response for each query, aligned with
        ``queries``; ``None`` where the query was not started (a budget) or
        its search or embedding raised (see the ``retrieve`` stage's error).
    passages : tuple of ContextPassage or None, default=None
        The compressed context, when ``compress`` ran.
    query : str or None, default=None
        The query the plan was run with.
    status : RetrievalStatus or None, default=None
        Leave ``None``: it is derived (see :meth:`_derive_status`).

    Notes
    -----
    **User.** Like a :class:`~scikitplot.corpus.RetrievalResponse`, this
    iterates, indexes and lens like its hits, so code that consumed a
    response consumes a plan's result.
    """

    hits: tuple[PlannedHit, ...]
    stages: tuple[StageOutcome, ...]
    queries: tuple[str, ...]
    responses: tuple[Any, ...]
    passages: tuple[ContextPassage, ...] | None = None
    query: str | None = None
    status: RetrievalStatus | None = None

    def __post_init__(self) -> None:
        """Derive :attr:`status`."""
        if self.status is None:
            object.__setattr__(self, "status", self._derive_status())

    def _derive_status(self) -> RetrievalStatus:
        """Compute the status from the stages and the responses.

        Notes
        -----
        In order:

        1. ``retrieve`` cancelled (no query ran: stopped before any, or the
           index cancelled each it ran) -> ``CANCELLED``;
        2. ``retrieve`` failed -> ``FAILED``;
        3. any stage degraded, failed or cancelled, or any response that ran
           not ``success`` or ``empty`` -> ``DEGRADED``;
        4. no hits -> ``EMPTY``;
        5. otherwise ``SUCCESS``.
        """
        retrieve = self.stage(StageKind.RETRIEVE)
        if retrieve is not None and retrieve.status is StageStatus.CANCELLED:
            return RetrievalStatus.CANCELLED
        if retrieve is not None and retrieve.status is StageStatus.FAILED:
            return RetrievalStatus.FAILED
        if any(stage.status in _EXPLAINED for stage in self.stages):
            return RetrievalStatus.DEGRADED
        for response in self.responses:
            if getattr(response, "status", None) is None:
                continue  # not run, or a plain list of hits
            # A status that is not a RetrievalStatus (or its value) is not
            # known to be complete.
            if _response_status(response) not in (
                RetrievalStatus.SUCCESS,
                RetrievalStatus.EMPTY,
            ):
                return RetrievalStatus.DEGRADED
        if not self.hits:
            return RetrievalStatus.EMPTY
        return RetrievalStatus.SUCCESS

    def __iter__(self) -> Iterator[PlannedHit]:
        """Iterate the hits."""
        return iter(self.hits)

    def __len__(self) -> int:
        """Return the number of hits."""
        return len(self.hits)

    def __getitem__(self, index):
        """Index into the hits."""
        return self.hits[index]

    def __bool__(self) -> bool:
        """Truthy when there are hits (consult :attr:`status` for why not)."""
        return bool(self.hits)

    def stage(self, kind: StageKind | str) -> StageOutcome | None:
        """Return the outcome of one stage."""
        kind = StageKind(kind)
        return next((s for s in self.stages if s.stage is kind), None)

    def error_records(self) -> list[ErrorRecord]:
        """Return every diagnostic: the stages' first, then the responses'."""
        records = [s.error for s in self.stages if s.error is not None]
        for response in self.responses:
            getter = getattr(response, "error_records", None)
            if callable(getter):
                records.extend(getter())
        return records

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of the outcome (not the hits)."""
        return {
            "status": self.status.value,
            "query": self.query,
            "queries": list(self.queries),
            "hit_count": len(self.hits),
            "passage_count": len(self.passages) if self.passages is not None else None,
            "stages": [s.to_dict() for s in self.stages],
            "responses": [_response_dict(r) for r in self.responses],
        }


def _response_status(response: Any) -> RetrievalStatus | None:
    """Return a response's status as a :class:`RetrievalStatus`, else ``None``.

    A status given as its string value (``"cancelled"``) is the member; any
    other value, or none, is not a status the plan acts on.
    """
    status = getattr(response, "status", None)
    if isinstance(status, RetrievalStatus) or status is None:
        return status
    try:
        return RetrievalStatus(status)
    except (ValueError, TypeError):
        return None


def _response_dict(response: Any) -> dict[str, Any] | None:
    """One response for :meth:`ExecutionResult.to_dict`.

    A response with ``to_dict`` gives its own; any other gives its status
    value (``None`` if it has none) and its length (``None`` if unsized), so
    a reader can tell a failed or cancelled response from a list of hits.
    """
    if response is None:
        return None
    if hasattr(response, "to_dict"):
        return response.to_dict()
    status = getattr(response, "status", None)
    return {
        "status": getattr(status, "value", None if status is None else str(status)),
        "hit_count": len(response) if hasattr(response, "__len__") else None,
    }


@dataclasses.dataclass(frozen=True)
class CharBudgetCompressor:
    """Keep each hit's text, best first, until a character budget is spent.

    Parameters
    ----------
    max_chars : int
        Characters of source text the context may hold in total.
    whole_words : bool, default=True
        Cut a partly kept hit at its last whitespace character
        (:meth:`str.isspace`) within the budget, so no word is split; a hit
        whose first word does not fit keeps nothing. Text without whitespace
        (Chinese, Japanese) is then kept only whole: use ``False`` for it.

    Notes
    -----
    **User.** Deterministic and order-preserving: hits are kept in rank order
    and each keeps one span from the start of its text. Hits after the budget
    is spent keep nothing.

    Examples
    --------
    >>> class Doc:
    ...     def __init__(self, text):
    ...         self.text = text
    >>> class Hit:
    ...     def __init__(self, text):
    ...         self.doc = Doc(text)
    >>> CharBudgetCompressor(12)("q", [Hit("alpha beta"), Hit("gamma delta")])
    [((0, 10),), ()]
    """

    max_chars: int
    whole_words: bool = True

    def __post_init__(self) -> None:
        """Validate the budget."""
        _positive_int(self.max_chars, "CharBudgetCompressor.max_chars")
        if not isinstance(self.whole_words, bool):
            raise TypeError("CharBudgetCompressor.whole_words must be a bool.")

    def __call__(
        self, query: str, hits: Sequence[Any]
    ) -> list[tuple[tuple[int, int], ...]]:
        """Return one tuple of spans per hit."""
        left = self.max_chars
        out: list[tuple[tuple[int, int], ...]] = []
        for hit in hits:
            text = getattr(getattr(hit, "doc", hit), "text", "") or ""
            if left <= 0 or not text:
                out.append(())
                continue
            end = min(len(text), left)
            if self.whole_words and end < len(text):
                # text[end] is the first character not kept: cut at the last
                # whitespace at or before it.
                end = next((j for j in range(end, 0, -1) if text[j].isspace()), 0)
            while end > 0 and text[end - 1].isspace():
                end -= 1
            out.append(((0, end),) if end > 0 else ())
            left -= end
        return out


def _component_name(component: Any) -> str | None:
    if component is None:
        return None
    return getattr(component, "__name__", None) or type(component).__name__


def _record(
    stage: StageKind, code: str, category: ErrorCategory, message: str, **details: Any
) -> ErrorRecord:
    return ErrorRecord(
        code=code,
        category=category,
        message=message,
        stage=stage.value,
        details=details,
    )


def _failure(
    stage: StageKind, exc: BaseException, component: str | None
) -> ErrorRecord:
    return ErrorRecord.from_exception(
        exc,
        code="STAGE_FAILED",
        category=ErrorCategory.CAPABILITY,
        stage=stage.value,
        details={"component": component},
    )


class _Clock:
    """The plan's clock: per-stage and whole-plan elapsed time."""

    def __init__(self, tick: Callable[[], float], deadline: float | None) -> None:
        self._tick = tick
        self._start = tick()
        self._deadline = deadline

    def now(self) -> float:
        return self._tick()

    def deadline_passed(self) -> bool:
        return self._deadline is not None and self.now() - self._start >= self._deadline


@dataclasses.dataclass(frozen=True)
class RetrievalExecutionPlan:
    """Run retrieval through typed, bounded stages.

    Parameters
    ----------
    retrieval : object or None, default=None
        Passed to ``index.search`` as ``config`` (a
        :class:`~scikitplot.corpus.RetrievalConfig`); ``None`` uses the
        index's own.
    transform : callable or None, default=None
        ``transform(query) -> sequence of str``: rewrites, sub-queries or
        variants. Empty strings are refused, exact duplicates are run once.
    keep_original : bool, default=True
        Run the original query first, whatever ``transform`` returns.
    max_queries : int, default=8
        Queries ``retrieve`` runs at most; further ones are dropped and the
        ``transform`` stage reports ``degraded``.
    fusion : FusionPolicy or None, default=None
        Fuses the queries' rankings (the leg of query ``i`` is named
        ``"q{i}"``); ``None`` is ``FusionPolicy()``. Not used with one query.
        A policy with ``top_k=0`` is refused; one that weighs every query 0
        makes ``fuse`` report ``degraded``.
    rerank : callable or None, default=None
        ``rerank(query, hits) -> sequence of float``, one finite score per
        hit, higher is better; ties keep the fused order.
    rerank_depth : int or None, default=None
        Hits passed to ``rerank``; ``None`` passes all. When set, ``top_k``
        must be set too and be at most ``rerank_depth``, so no returned hit
        is ranked by two scales.
    compress : callable or None, default=None
        ``compress(query, hits) -> sequence``, one entry per hit, each a
        sequence of ``(start, end)`` character ranges of ``hit.doc.text``,
        ordered and non-overlapping; an empty entry keeps nothing of that hit
        (see :class:`CharBudgetCompressor`).
    top_k : int or None, default=None
        Hits returned; ``None`` returns every hit ``retrieve`` found.
    budgets : mapping of str to StageBudget, default=()
        ``{"retrieve": StageBudget(...)}``: the one stage that makes several
        calls. The other stages make one call each, which a budget checked
        before calls cannot bound; they are refused here.
    deadline_seconds : float or None, default=None
        Seconds after the plan starts beyond which no stage or query starts
        (``fuse``, an in-process computation, always runs).
    on_error : {"degrade", "raise"}, default="degrade"
        What an exception from a component (or from ``index.search``) does:
        ``"degrade"`` reports the stage ``failed`` and keeps the previous
        stage's output; ``"raise"`` propagates it.
    clock : callable or None, default=None
        ``clock() -> seconds``; ``None`` is :func:`time.perf_counter`.

    Raises
    ------
    TypeError, ValueError
        For a field of the wrong type or out of range.

    See Also
    --------
    scikitplot.corpus.RetrievalIndex.search : What ``retrieve`` calls.
    scikitplot.corpus.FusionPolicy : How the queries' rankings are fused.
    scikitplot.corpus.evaluate_retrieval : Measures ``plan.bind(index)``.

    Notes
    -----
    **User.** An empty plan, ``RetrievalExecutionPlan().run(index, q)``,
    returns the hits of ``index.search(q)`` in the same order, and its status.

    **Developer.** A stage with nothing to do is ``skipped`` (``fuse`` with
    one query, ``rerank`` with no hits), not ``success``, so a report never
    claims work that was not done.

    Examples
    --------
    >>> plan = RetrievalExecutionPlan(
    ...     top_k=3, budgets={"retrieve": StageBudget(max_calls=4)}
    ... )
    >>> plan.budget("retrieve")
    StageBudget(max_calls=4, max_seconds=None)
    """

    retrieval: Any = None
    transform: Callable[[str], Sequence[str]] | None = None
    keep_original: bool = True
    max_queries: int = 8
    fusion: FusionPolicy | None = None
    rerank: Callable[[str, Sequence[PlannedHit]], Sequence[float]] | None = None
    rerank_depth: int | None = None
    compress: Callable[[str, Sequence[PlannedHit]], Sequence[Any]] | None = None
    top_k: int | None = None
    budgets: Mapping[str, StageBudget] | tuple[tuple[str, StageBudget], ...] = ()
    deadline_seconds: float | None = None
    on_error: str = "degrade"
    clock: Callable[[], float] | None = None

    def __post_init__(self) -> None:
        """Validate every field and freeze the budgets."""
        for name in ("transform", "rerank", "compress", "clock"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(
                    f"RetrievalExecutionPlan.{name} must be callable or None."
                )
        if not isinstance(self.keep_original, bool):
            raise TypeError("RetrievalExecutionPlan.keep_original must be a bool.")
        _positive_int(self.max_queries, "RetrievalExecutionPlan.max_queries")
        if self.fusion is not None and not isinstance(self.fusion, FusionPolicy):
            raise TypeError(
                "RetrievalExecutionPlan.fusion must be a FusionPolicy or None."
            )
        if self.fusion is not None and self.fusion.top_k == 0:
            raise ValueError(
                "RetrievalExecutionPlan.fusion: top_k=0 would drop every hit; "
                "use the plan's top_k to cut the ranking."
            )
        if self.rerank_depth is not None:
            _positive_int(self.rerank_depth, "RetrievalExecutionPlan.rerank_depth")
        if self.top_k is not None:
            _positive_int(self.top_k, "RetrievalExecutionPlan.top_k")
        if (
            self.rerank is not None
            and self.rerank_depth is not None
            and (self.top_k is None or self.top_k > self.rerank_depth)
        ):
            raise ValueError(
                "RetrievalExecutionPlan: top_k must be set and at most rerank_depth, "
                "so every returned hit is ranked by the reranker's scale "
                f"(top_k={self.top_k!r}, rerank_depth={self.rerank_depth!r})."
            )
        if self.deadline_seconds is not None:
            object.__setattr__(
                self,
                "deadline_seconds",
                _positive_seconds(
                    self.deadline_seconds, "RetrievalExecutionPlan.deadline_seconds"
                ),
            )
        if self.on_error not in ON_ERROR:
            raise ValueError(
                f"RetrievalExecutionPlan.on_error must be one of {ON_ERROR}, "
                f"got {self.on_error!r}."
            )
        object.__setattr__(self, "budgets", self._checked_budgets(self.budgets))

    @staticmethod
    def _checked_budgets(budgets: Any) -> tuple[tuple[str, StageBudget], ...]:
        if isinstance(budgets, Mapping):
            pairs = list(budgets.items())
        elif isinstance(budgets, tuple):
            pairs = list(budgets)
        else:
            raise TypeError(
                "RetrievalExecutionPlan.budgets must be a mapping of stage name to "
                f"StageBudget, got {type(budgets).__name__}."
            )
        checked: dict[str, StageBudget] = {}
        for pair in pairs:
            if not (isinstance(pair, tuple) and len(pair) == 2):  # noqa: PLR2004
                raise TypeError(
                    "RetrievalExecutionPlan.budgets: each entry is (stage, budget)."
                )
            name, budget = pair
            try:
                kind = StageKind(name)
            except ValueError:
                raise ValueError(
                    f"RetrievalExecutionPlan.budgets: unknown stage {name!r}; stages are "
                    f"{[k.value for k in STAGE_ORDER]}."
                ) from None
            if kind is not StageKind.RETRIEVE:
                raise ValueError(
                    f"RetrievalExecutionPlan.budgets: {kind.value!r} makes one call, "
                    "which a budget checked before calls cannot bound; only "
                    "'retrieve' takes a StageBudget. Use deadline_seconds, or the "
                    "component's own timeout."
                )
            if not isinstance(budget, StageBudget):
                raise TypeError(
                    f"RetrievalExecutionPlan.budgets[{kind.value!r}] must be a StageBudget."
                )
            if kind.value in checked:
                raise ValueError(
                    f"RetrievalExecutionPlan.budgets: {kind.value!r} given twice."
                )
            checked[kind.value] = budget
        return tuple(sorted(checked.items()))

    def budget(self, stage: StageKind | str) -> StageBudget | None:
        """Return the budget of ``stage``, or ``None``."""
        return dict(self.budgets).get(StageKind(stage).value)

    # ------------------------------------------------------------------ run

    def bind(self, index: Any, **kwargs: Any) -> Callable[[str], ExecutionResult]:
        """Return ``query -> self.run(index, query, **kwargs)``.

        Parameters
        ----------
        index : object
            Anything with ``search``.
        **kwargs
            Passed to :meth:`run` on every call.

        Returns
        -------
        callable
            For :func:`~scikitplot.corpus.evaluate_retrieval`.
        """

        def run(query: str) -> ExecutionResult:
            return self.run(index, query, **kwargs)

        run.__name__ = f"plan({_component_name(index)})"
        return run

    def run(
        self,
        index: Any,
        query: str,
        *,
        query_embedding: Any = None,
        embed_query: Callable[[str], Any] | None = None,
        filter: Any = None,  # noqa: A002
    ) -> ExecutionResult:
        """Run the plan for ``query``.

        Parameters
        ----------
        index : object
            Anything with ``search(query, **kwargs)`` returning an iterable of
            hits (:class:`~scikitplot.corpus.RetrievalIndex`). Each hit carries
            an identity (``hit.doc.doc_id``, ``hit.doc_id`` or ``hit.node_id``)
            and a real ``score``; a query whose hits do not fails. Bare
            ``(id, score)`` pairs are not hits.
        query : str
            The query.
        query_embedding : array-like or None, default=None
            Embedding of ``query`` itself, passed with it to ``index.search``.
        embed_query : callable or None, default=None
            ``embed_query(text) -> embedding`` for the queries the transform
            produced (and for ``query`` when ``query_embedding`` is ``None``).
            Without it, those queries are searched with no embedding, and a
            dense leg reports why it could not run.
        filter : Filter, mapping or None, default=None
            Passed to every ``index.search`` call.

        Returns
        -------
        ExecutionResult

        Raises
        ------
        TypeError
            If ``query`` is not a ``str`` or ``index`` has no ``search``; with
            ``on_error="raise"``, a component's output of the wrong type.
        ValueError
            If ``query`` is empty; with ``on_error="raise"``, a component's
            output out of range.
        Exception
            With ``on_error="raise"``, whatever a component or
            ``index.search`` raised.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}.")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only.")
        search = getattr(index, "search", None)
        if not callable(search):
            raise TypeError(
                f"index must have a search method, got {type(index).__name__}."
            )
        if embed_query is not None and not callable(embed_query):
            raise TypeError("embed_query must be callable or None.")
        clock = _Clock(self.clock or time.perf_counter, self.deadline_seconds)

        stages: list[StageOutcome] = []
        queries, outcome = self._transform(query, clock)
        stages.append(outcome)
        responses, ranked, outcome = self._retrieve(
            search,
            query,
            queries,
            query_embedding=query_embedding,
            embed_query=embed_query,
            filter=filter,
            clock=clock,
        )
        stages.append(outcome)
        hits, outcome = self._fuse(queries, ranked, clock)
        stages.append(outcome)
        hits, outcome = self._rerank(query, hits, clock)
        stages.append(outcome)
        if self.top_k is not None:
            hits = hits[: self.top_k]
        passages, outcome = self._compress(query, hits, clock)
        stages.append(outcome)
        return ExecutionResult(
            hits=tuple(hits),
            stages=tuple(stages),
            queries=tuple(queries),
            responses=tuple(responses),
            passages=passages,
            query=query,
        )

    # --------------------------------------------------------------- stages

    def _stopped(
        self, kind: StageKind, clock: _Clock, started: float, calls: int
    ) -> ErrorRecord | None:
        """Return why ``kind`` may start no further call, or ``None``."""
        budget = self.budget(kind)
        if (
            budget is not None
            and budget.max_calls is not None
            and calls >= budget.max_calls
        ):
            return _record(
                kind,
                "BUDGET_EXHAUSTED",
                ErrorCategory.RESOURCE,
                f"{kind.value}: max_calls limit of {budget.max_calls} reached",
                budget="max_calls",
                limit=budget.max_calls,
            )
        if (
            budget is not None
            and budget.max_seconds is not None
            and clock.now() - started >= budget.max_seconds
        ):
            return _record(
                kind,
                "BUDGET_EXHAUSTED",
                ErrorCategory.RESOURCE,
                f"{kind.value}: max_seconds limit of {budget.max_seconds} reached",
                budget="max_seconds",
                limit=budget.max_seconds,
            )
        if clock.deadline_passed():
            return _record(
                kind,
                "DEADLINE_EXCEEDED",
                ErrorCategory.RESOURCE,
                f"{kind.value}: the plan's deadline of {self.deadline_seconds} s has passed",
                budget="deadline_seconds",
                limit=self.deadline_seconds,
            )
        return None

    def _outcome(
        self,
        kind: StageKind,
        status: StageStatus,
        started: float,
        clock: _Clock,
        **fields: Any,
    ) -> StageOutcome:
        details = fields.pop("details", {})
        return StageOutcome(
            stage=kind,
            status=status,
            seconds=max(0.0, clock.now() - started),
            budget=self.budget(kind),
            details=tuple(sorted(details.items())),
            **fields,
        )

    def _transform(self, query: str, clock: _Clock) -> tuple[list[str], StageOutcome]:
        kind = StageKind.TRANSFORM
        started = clock.now()
        name = _component_name(self.transform)
        if self.transform is None:
            return [query], self._outcome(
                kind, StageStatus.SKIPPED, started, clock, items_in=1, items_out=1
            )
        stop = self._stopped(kind, clock, started, 0)
        if stop is not None:
            return [query], self._outcome(
                kind,
                StageStatus.CANCELLED,
                started,
                clock,
                component=name,
                items_in=1,
                items_out=1,
                error=stop,
            )
        try:
            produced = self.transform(query)
            proposed = _check_queries(produced)
        except Exception as exc:  # noqa: BLE001 - reported, or re-raised below
            if self.on_error == "raise":
                raise
            return [query], self._outcome(
                kind,
                StageStatus.FAILED,
                started,
                clock,
                component=name,
                items_in=1,
                items_out=1,
                calls=1,
                error=_failure(kind, exc, name),
            )
        queries: list[str] = [query] if self.keep_original else []
        for candidate in proposed:
            if candidate not in queries:
                queries.append(candidate)
        added = len(queries) - (1 if self.keep_original else 0)
        details: dict[str, Any] = {
            "proposed": len(proposed),
            # Proposals not run because they repeat an earlier query (the
            # original included).
            "duplicates": len(proposed) - added,
        }
        if not queries:
            error = _record(
                kind,
                "NO_QUERY",
                ErrorCategory.VALIDATION,
                "transform: no query to run; the original is used",
            )
            return [query], self._outcome(
                kind,
                StageStatus.FAILED,
                started,
                clock,
                component=name,
                items_in=1,
                items_out=1,
                calls=1,
                error=error,
                details=details,
            )
        if len(queries) > self.max_queries:
            dropped = queries[self.max_queries :]
            queries = queries[: self.max_queries]
            details["dropped"] = len(dropped)
            error = _record(
                kind,
                "QUERIES_DROPPED",
                ErrorCategory.RESOURCE,
                f"transform: {len(dropped)} queries beyond max_queries={self.max_queries} not run",
                dropped=dropped,
            )
            return queries, self._outcome(
                kind,
                StageStatus.DEGRADED,
                started,
                clock,
                component=name,
                items_in=1,
                items_out=len(queries),
                calls=1,
                error=error,
                details=details,
            )
        return queries, self._outcome(
            kind,
            StageStatus.SUCCESS,
            started,
            clock,
            component=name,
            items_in=1,
            items_out=len(queries),
            calls=1,
            details=details,
        )

    def _search_one(
        self,
        search: Callable[..., Any],
        index: int,
        text: str,
        *,
        embedding: Any,
        embed_query: Callable[[str], Any] | None,
        filter: Any,  # noqa: A002
    ) -> tuple[Any, list[Any] | None, ErrorRecord | None, bool]:
        """Search one query: ``(response, hits, failure, cancelled)``.

        ``hits`` is ``None`` when the query failed: ``search`` or
        ``embed_query`` raised (``failure`` is that exception's record), or the
        response's status is ``failed`` or ``cancelled`` (``failure`` is the
        index's first :class:`ErrorRecord`, else the plan's own record).
        ``cancelled`` is ``True`` only for a ``cancelled`` response.
        """
        kind = StageKind.RETRIEVE
        kwargs: dict[str, Any] = {}
        if self.retrieval is not None:
            kwargs["config"] = self.retrieval
        if filter is not None:
            kwargs["filter"] = filter
        try:
            if embedding is not None:
                kwargs["query_embedding"] = embedding
            elif embed_query is not None:
                kwargs["query_embedding"] = embed_query(text)
            response = search(text, **kwargs)
            hits = list(response)
            for rank, hit in enumerate(hits, start=1):
                _fusion_key(hit, f"q{index}", rank)  # refuse a hit with no identity
                score = getattr(hit, "score", None)
                if isinstance(score, bool) or not isinstance(score, numbers.Real):
                    raise TypeError(
                        f"query {index} rank {rank}: a hit's score must be a real "
                        f"number, got {type(score).__name__}"
                    )
        except Exception as exc:  # noqa: BLE001 - reported, or re-raised below
            if self.on_error == "raise":
                raise
            return (
                None,
                None,
                ErrorRecord.from_exception(
                    exc,
                    code="QUERY_FAILED",
                    category=ErrorCategory.CAPABILITY,
                    stage=kind.value,
                    details={"query_index": index},
                ),
                False,
            )
        status = _response_status(response)
        if status not in (RetrievalStatus.FAILED, RetrievalStatus.CANCELLED):
            return response, hits, None, False
        records = getattr(response, "error_records", None)
        first = records()[:1] if callable(records) else []
        if first and isinstance(first[0], ErrorRecord):
            known = first[0].details
            known = dict(known) if isinstance(known, Mapping) else {}
            failure = dataclasses.replace(
                first[0], details={**known, "query_index": index}
            )
        else:  # no record, or not an ErrorRecord: the plan's own
            failure = _record(
                kind,
                "QUERY_FAILED",
                ErrorCategory.CAPABILITY,
                f"retrieve: query {index} returned a {status.value} response",
                query_index=index,
            )
        return response, None, failure, status is RetrievalStatus.CANCELLED

    def _retrieve(
        self,
        search: Callable[..., Any],
        query: str,
        queries: list[str],
        *,
        query_embedding: Any,
        embed_query: Callable[[str], Any] | None,
        filter: Any,  # noqa: A002
        clock: _Clock,
    ) -> tuple[list[Any], list[list[Any] | None], StageOutcome]:
        kind = StageKind.RETRIEVE
        started = clock.now()
        name = _component_name(getattr(search, "__self__", search))
        responses: list[Any] = [None] * len(queries)
        ranked: list[list[Any] | None] = [None] * len(queries)
        failures: list[ErrorRecord] = []
        cancelled = 0  # queries whose response the index itself cancelled
        stop: ErrorRecord | None = None
        calls = 0
        for i, text in enumerate(queries):
            stop = self._stopped(kind, clock, started, calls)
            if stop is not None:
                break
            calls += 1
            responses[i], ranked[i], failure, was_cancelled = self._search_one(
                search,
                i,
                text,
                embedding=query_embedding if text == query else None,
                embed_query=embed_query,
                filter=filter,
            )
            if failure is not None:
                failures.append(failure)
                cancelled += was_cancelled
        ran = [i for i, hits in enumerate(ranked) if hits is not None]
        details: dict[str, Any] = {
            "queries": len(queries),
            "ran": len(ran),
            "failed": len(failures),
            "not_started": len(queries) - calls,
        }
        if stop is not None:
            details["stopped_by"] = stop.details.get("budget")
        budget = self.budget(kind)
        elapsed = clock.now() - started
        if (
            budget is not None
            and budget.max_seconds is not None
            and elapsed > budget.max_seconds
        ):
            # Time spent past the budget: a call already running is not
            # interrupted, whether or not a stop followed it.
            details["seconds_over_budget"] = elapsed - budget.max_seconds
        if not ran and (
            (stop is not None and not failures)
            or (failures and cancelled == len(failures))
        ):
            status, error = StageStatus.CANCELLED, failures[0] if failures else stop
        elif not ran:
            status, error = StageStatus.FAILED, failures[0] if failures else stop
        elif failures or stop is not None:
            status, error = StageStatus.DEGRADED, failures[0] if failures else stop
        else:
            found = any(ranked[i] for i in ran)
            status, error = (StageStatus.SUCCESS if found else StageStatus.EMPTY), None
        return (
            responses,
            ranked,
            self._outcome(
                kind,
                status,
                started,
                clock,
                component=name,
                items_in=len(queries),
                items_out=len(ran),
                calls=calls,
                error=error,
                details=details,
            ),
        )

    def _fuse(
        self, queries: list[str], ranked: list[list[Any] | None], clock: _Clock
    ) -> tuple[list[PlannedHit], StageOutcome]:
        kind = StageKind.FUSE
        started = clock.now()
        ran = [i for i, hits in enumerate(ranked) if hits is not None]
        keys: dict[int, list[str]] = {
            i: [
                _fusion_key(hit, f"q{i}", r) for r, hit in enumerate(ranked[i], start=1)
            ]
            for i in ran
        }
        best: list[dict[str, int]] = []
        for i in range(len(queries)):
            first_rank: dict[str, int] = {}
            for r, key in enumerate(keys.get(i, ())):
                first_rank.setdefault(key, r)
            best.append(first_rank)

        def ranks_of(key: str) -> tuple[int | None, ...]:
            return tuple(b.get(key) for b in best)

        def one_ranking(i: int | None) -> list[PlannedHit]:
            # Exactly the ranking of query i, duplicates included.
            if i is None:
                return []
            return [
                PlannedHit(
                    hit=hit,
                    score=float(getattr(hit, "score", 0.0)),
                    key=key,
                    query_ranks=ranks_of(key),
                )
                for hit, key in zip(ranked[i], keys[i])
            ]

        items_in = sum(len(ranked[i]) for i in ran)
        if len(ran) <= 1:
            planned = one_ranking(ran[0] if ran else None)
            return planned, self._outcome(
                kind,
                StageStatus.SKIPPED,
                started,
                clock,
                items_in=items_in,
                items_out=len(planned),
                details={"reason": "fewer than two queries ran"},
            )
        policy = self.fusion or FusionPolicy()
        first: dict[str, Any] = {}
        for i in ran:
            for hit, key in zip(ranked[i], keys[i]):
                first.setdefault(key, hit)
        fused = policy.fuse((f"q{i}", keys[i]) for i in ran)
        planned = [
            PlannedHit(
                hit=first[item.key],
                score=item.score,
                key=item.key,
                fused_score=item.score,
                query_ranks=ranks_of(item.key),
            )
            for item in fused
        ]
        error = None
        if planned:
            status = StageStatus.SUCCESS
        elif items_in:
            # Hits went in and none came out: the policy weighs every query 0.
            status = StageStatus.DEGRADED
            error = _record(
                kind,
                "FUSION_DROPPED_ALL",
                ErrorCategory.VALIDATION,
                f"fuse: the fusion policy returned no hit for {items_in} retrieved; "
                "it gives weight 0 to every query that returned hits "
                "(legs 'q0', 'q1', ...)",
                weights=dict(policy.weights),
                default_weight=policy.default_weight,
            )
        else:
            status = StageStatus.EMPTY
        return planned, self._outcome(
            kind,
            status,
            started,
            clock,
            component="FusionPolicy",
            items_in=items_in,
            items_out=len(planned),
            calls=1,
            error=error,
            details={"k": policy.k, "legs": len(ran)},
        )

    def _rerank(
        self, query: str, hits: list[PlannedHit], clock: _Clock
    ) -> tuple[list[PlannedHit], StageOutcome]:
        kind = StageKind.RERANK
        started = clock.now()
        name = _component_name(self.rerank)
        if self.rerank is None:
            return hits, self._outcome(
                kind,
                StageStatus.SKIPPED,
                started,
                clock,
                items_in=len(hits),
                items_out=len(hits),
            )
        if not hits:
            return hits, self._outcome(
                kind,
                StageStatus.SKIPPED,
                started,
                clock,
                component=name,
                details={"reason": "no hits"},
            )
        depth = (
            len(hits)
            if self.rerank_depth is None
            else min(self.rerank_depth, len(hits))
        )
        head, tail = hits[:depth], hits[depth:]
        common = {
            "component": name,
            "items_in": depth,
            "details": {"depth": depth, "candidates": len(hits)},
        }
        stop = self._stopped(kind, clock, started, 0)
        if stop is not None:
            return hits, self._outcome(
                kind,
                StageStatus.CANCELLED,
                started,
                clock,
                items_out=len(hits),
                error=stop,
                **common,
            )
        try:
            scores = _check_scores(self.rerank(query, tuple(head)), len(head))
        except Exception as exc:  # noqa: BLE001 - reported, or re-raised below
            if self.on_error == "raise":
                raise
            return hits, self._outcome(
                kind,
                StageStatus.FAILED,
                started,
                clock,
                items_out=len(hits),
                calls=1,
                error=_failure(kind, exc, name),
                **common,
            )
        order = sorted(range(depth), key=lambda i: (-scores[i], i))
        reranked = [
            dataclasses.replace(head[i], score=scores[i], rerank_score=scores[i])
            for i in order
        ]
        return reranked + tail, self._outcome(
            kind,
            StageStatus.SUCCESS,
            started,
            clock,
            items_out=depth,
            calls=1,
            **common,
        )

    def _compress(
        self, query: str, hits: list[PlannedHit], clock: _Clock
    ) -> tuple[tuple[ContextPassage, ...] | None, StageOutcome]:
        kind = StageKind.COMPRESS
        started = clock.now()
        name = _component_name(self.compress)
        if self.compress is None:
            return None, self._outcome(
                kind,
                StageStatus.SKIPPED,
                started,
                clock,
                items_in=len(hits),
                items_out=len(hits),
            )
        stop = self._stopped(kind, clock, started, 0)
        if stop is not None:
            return None, self._outcome(
                kind,
                StageStatus.CANCELLED,
                started,
                clock,
                component=name,
                items_in=len(hits),
                error=stop,
            )
        try:
            spans = _check_spans(self.compress(query, tuple(hits)), hits)
        except Exception as exc:  # noqa: BLE001 - reported, or re-raised below
            if self.on_error == "raise":
                raise
            return None, self._outcome(
                kind,
                StageStatus.FAILED,
                started,
                clock,
                component=name,
                items_in=len(hits),
                calls=1,
                error=_failure(kind, exc, name),
            )
        passages = tuple(
            ContextPassage(hit=hit, spans=s) for hit, s in zip(hits, spans) if s
        )
        kept = sum(p.chars for p in passages)
        status = StageStatus.SUCCESS if passages else StageStatus.EMPTY
        return passages, self._outcome(
            kind,
            status,
            started,
            clock,
            component=name,
            items_in=len(hits),
            items_out=len(passages),
            calls=1,
            details={"chars": kept},
        )


def _check_queries(produced: Any) -> list[str]:
    """Validate a transform's output: a sequence of non-empty strings."""
    if isinstance(produced, (str, bytes)) or not isinstance(produced, Sequence):
        raise TypeError(
            "transform must return a sequence of str (a list or tuple), got "
            f"{type(produced).__name__}."
        )
    out = []
    for i, item in enumerate(produced):
        if not isinstance(item, str):
            raise TypeError(f"transform: item {i} is {type(item).__name__}, not str.")
        if not item.strip():
            raise ValueError(f"transform: item {i} is empty or whitespace-only.")
        out.append(item)
    return out


def _check_scores(produced: Any, expected: int) -> list[float]:
    """Validate a reranker's output: one finite real per hit."""
    ordered = isinstance(produced, Sequence) or hasattr(produced, "__array__")
    if (
        isinstance(produced, (str, bytes, AbstractSet, Mapping))
        or not ordered
        or not hasattr(produced, "__len__")
    ):
        raise TypeError(
            f"rerank must return a sequence of {expected} scores (a list, tuple or "
            f"array, in hit order), got {type(produced).__name__}."
        )
    scores = list(produced)
    if len(scores) != expected:
        raise ValueError(f"rerank returned {len(scores)} scores for {expected} hits.")
    out = []
    for i, raw in enumerate(scores):
        if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
            raise TypeError(
                f"rerank: score {i} is {type(raw).__name__}, not a real number."
            )
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"rerank: score {i} is {value!r}, not finite.")
        out.append(value)
    return out


def _check_spans(
    produced: Any, hits: Sequence[PlannedHit]
) -> list[tuple[tuple[int, int], ...]]:
    """Validate a compressor's output: per hit, ordered disjoint spans of its text."""
    if isinstance(produced, (str, bytes)) or not hasattr(produced, "__len__"):
        raise TypeError(
            f"compress must return one entry per hit, got {type(produced).__name__}."
        )
    entries = list(produced)
    if len(entries) != len(hits):
        raise ValueError(
            f"compress returned {len(entries)} entries for {len(hits)} hits."
        )
    out = []
    for hit, entry in zip(hits, entries):
        entry = list(entry)  # noqa: PLW2901
        if not entry:
            out.append(())  # nothing kept: a hit without text is fine
            continue
        text = _doc_text(hit)
        spans = []
        previous = 0
        for span in entry:
            if (
                not isinstance(span, (tuple, list))
                or len(span) != 2  # noqa: PLR2004
                or any(
                    isinstance(v, bool) or not isinstance(v, numbers.Integral)
                    for v in span
                )
            ):
                raise TypeError(
                    f"compress: hit {hit.key!r}: a span is not a pair of ints: {span!r}."
                )
            start, end = operator.index(span[0]), operator.index(span[1])
            if not previous <= start < end <= len(text):
                raise ValueError(
                    f"compress: hit {hit.key!r}: span {tuple(span)!r} is not inside its "
                    f"text (length {len(text)}), ordered and disjoint."
                )
            spans.append((start, end))
            previous = end
        out.append(tuple(spans))
    return out
