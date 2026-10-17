# scikitplot/corpus/_correction.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Bounded corrective retrieval: grade the evidence, then accept, retry or abstain.

:class:`CorrectiveRetrieval` runs a query, grades what came back with a
deterministic grader, and then:

* ``correct`` evidence is accepted;
* ``incorrect`` or ``ambiguous`` evidence is retried with a rewritten query,
  while rounds remain and the rewrite proposes a new query;
* when no round is accepted, the ``ambiguous`` rounds are fused and returned
  as partial evidence, or, when there are none, the retrieval abstains and
  returns no hit.

It is a stage around the existing retrieval contract, never a new index:
each round is one ``index.search`` (or one
:class:`~scikitplot.corpus.RetrievalExecutionPlan` run), and the result is
measured by :func:`~scikitplot.corpus.evaluate_retrieval` like an index
(``evaluate_retrieval(corrective.bind(index), ...)``).

.. code-block:: python

    from scikitplot.corpus import CorrectiveRetrieval, TermCoverageGrader

    corrective = CorrectiveRetrieval(
        grader=TermCoverageGrader(top_k=3, accept=0.8, reject=0.3),
        rewrite=lambda query, rounds: my_rewriter(query, rounds),  # or None
        max_rounds=3,
    )
    result = corrective.run(index, "what keeps a chunk's span?")
    result.verdict, [r.grade.grade for r in result.rounds], list(result)

Notes
-----
**Developer.** The grader decides from the hits alone, by a rule whose
thresholds the caller sets; :class:`TermCoverageGrader` measures how many of
the query's words the top hits contain. It is a deterministic signal, not a
judgement of truth: a hit can hold every query word and still not answer the
query. A model grader is any callable with the same signature, and its
identity is recorded with every grade. The number of rounds is the call
budget (Rule 127): it is checked before each round, and so is
``deadline_seconds``; a round already running is not interrupted.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import re
import time
from enum import unique
from typing import Any, Callable, Iterator, Mapping

from ._diagnostics import ErrorCategory, ErrorRecord
from ._fusion import FusionPolicy
from ._retrieval import RetrievalStatus
from ._retrievers import _fusion_key
from ._schema import _StrEnumBase

__all__ = [
    "CorrectionResult",
    "CorrectionRound",
    "CorrectionVerdict",
    "CorrectiveRetrieval",
    "EvidenceGrade",
    "GradeResult",
    "TermCoverageGrader",
]

_WORD = re.compile(r"\w+")


@unique
class EvidenceGrade(_StrEnumBase):
    """How a grader judged one round's hits."""

    CORRECT = "correct"
    """The hits are evidence for the query: accept them."""

    AMBIGUOUS = "ambiguous"
    """Partly: retry if rounds remain, else keep them as partial evidence."""

    INCORRECT = "incorrect"
    """Not evidence: retry if rounds remain, else abstain."""


@unique
class CorrectionVerdict(_StrEnumBase):
    """What the corrective retrieval came to."""

    ACCEPTED = "accepted"
    """The first round was graded correct."""

    CORRECTED = "corrected"
    """A later round, with a rewritten query, was graded correct."""

    PARTIAL = "partial"
    """No round was correct; the ambiguous rounds' hits, fused, are returned."""

    ABSTAINED = "abstained"
    """No round was correct or ambiguous; no hit is returned."""


@dataclasses.dataclass(frozen=True)
class GradeResult:
    """
    One grader's judgement of one round.

    Parameters
    ----------
    grade : EvidenceGrade or str
        The judgement.
    score : float
        In ``[0, 1]``; the grader's measure (coverage, for
        :class:`TermCoverageGrader`). Any real number type (``numpy.float32``
        included) is accepted and stored as ``float``.
    reason : str, default=""
        Why, in words.
    grader : str or None, default=None
        The grader's identity. A grader may set it (a model and its version);
        when it is ``None``, :class:`CorrectiveRetrieval` fills in the
        grader's name.

    Raises
    ------
    TypeError, ValueError
        If ``grade`` is not an :class:`EvidenceGrade` (or its value) or
        ``score`` is not a finite number in ``[0, 1]``.
    """

    grade: EvidenceGrade
    score: float
    reason: str = ""
    grader: str | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "grade", EvidenceGrade(self.grade))
        except ValueError:
            raise ValueError(
                f"GradeResult.grade: unknown grade {self.grade!r}"
            ) from None
        if isinstance(self.score, bool) or not isinstance(self.score, numbers.Real):
            raise TypeError("GradeResult.score must be a real number.")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"GradeResult.score must be in [0, 1], got {self.score!r}."
            )
        object.__setattr__(self, "score", float(self.score))
        if not isinstance(self.reason, str):
            raise TypeError("GradeResult.reason must be a string.")
        if self.grader is not None and not isinstance(self.grader, str):
            raise TypeError("GradeResult.grader must be a string or None.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "grade": self.grade.value,
            "score": self.score,
            "reason": self.reason,
            "grader": self.grader,
        }


def _hit_text(hit: Any) -> str:
    text = getattr(getattr(hit, "doc", hit), "text", None)
    return text if isinstance(text, str) else ""


@dataclasses.dataclass(frozen=True)
class TermCoverageGrader:
    r"""
    Grade hits by the share of the query's words their text contains.

    ``coverage = |query words found in the top hits| / |query words|``, with
    words ``\w+`` lower-cased and counted once. Coverage at or above
    ``accept`` is ``correct``; at or below ``reject`` is ``incorrect``;
    between is ``ambiguous``. No hits, or a query without a word, is
    ``incorrect`` with score 0.

    Parameters
    ----------
    top_k : int, default=3
        Hits whose text is read.
    accept : float, default=0.8
        Coverage from which the evidence is ``correct``.
    reject : float, default=0.3
        Coverage up to which it is ``incorrect``; below ``accept``.

    Raises
    ------
    TypeError, ValueError
        If ``top_k`` is not a positive int or the thresholds are not
        ``0 <= reject < accept <= 1``.

    Examples
    --------
    >>> from types import SimpleNamespace as NS
    >>> from scikitplot.corpus import TermCoverageGrader
    >>> hit = NS(doc=NS(text="chunk spans are slices of the source"))
    >>> TermCoverageGrader()("chunk spans", [hit]).grade.value
    'correct'
    >>> TermCoverageGrader()("chunk embeddings cache", [hit]).grade.value
    'ambiguous'
    """

    top_k: int = 3
    accept: float = 0.8
    reject: float = 0.3

    def __post_init__(self) -> None:
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 1
        ):
            raise ValueError("TermCoverageGrader.top_k must be a positive int.")
        for name in ("accept", "reject"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"TermCoverageGrader.{name} must be a real number.")
            object.__setattr__(self, name, float(value))
        if not 0.0 <= self.reject < self.accept <= 1.0:
            raise ValueError(
                "TermCoverageGrader: thresholds must satisfy 0 <= reject < accept <= 1, "
                f"got reject={self.reject!r}, accept={self.accept!r}."
            )

    def __call__(self, query: str, hits: Any) -> GradeResult:
        """
        Grade ``hits`` for ``query``.

        Parameters
        ----------
        query : str
            The query the hits are evidence for.
        hits : iterable
            Ranked hits; ``hit.doc.text`` (or ``hit.text``) is read, and a hit
            without text contributes no word.

        Returns
        -------
        GradeResult
        """
        words = set(_WORD.findall(query.lower()))
        top = list(hits)[: self.top_k]
        if not words or not top:
            why = "the query has no word" if not words else "no hit"
            return GradeResult(EvidenceGrade.INCORRECT, 0.0, why)
        found: set[str] = set()
        for hit in top:
            found |= set(_WORD.findall(_hit_text(hit).lower()))
        coverage = len(words & found) / len(words)
        if coverage >= self.accept:
            grade = EvidenceGrade.CORRECT
        elif coverage <= self.reject:
            grade = EvidenceGrade.INCORRECT
        else:
            grade = EvidenceGrade.AMBIGUOUS
        return GradeResult(
            grade,
            coverage,
            f"{len(words & found)} of {len(words)} query words in the top "
            f"{len(top)} hits",
        )


@dataclasses.dataclass(frozen=True)
class CorrectionRound:
    """
    One round: the query run, how it was graded, what came back.

    Parameters
    ----------
    index : int
        0-based round number.
    query : str
        The query this round ran (the original, or a rewrite).
    grade : GradeResult
        The grade of its hits, against the original query.
    status : RetrievalStatus or None
        The response's status; ``None`` for a response without one.
    hit_count : int
        Hits the round returned.
    seconds : float
        Time the round took (search and grading).
    error : ErrorRecord or None, default=None
        The round's first error (its search failed, its response was not
        complete, or its grader failed).
    """

    index: int
    query: str
    grade: GradeResult
    status: RetrievalStatus | None
    hit_count: int
    seconds: float
    error: ErrorRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "index": self.index,
            "query": self.query,
            "grade": self.grade.to_dict(),
            "status": (
                None if self.status is None else RetrievalStatus(self.status).value
            ),
            "hit_count": self.hit_count,
            "seconds": self.seconds,
            "error": None if self.error is None else self.error.to_dict(),
        }


@dataclasses.dataclass(frozen=True)
class CorrectionResult:
    """
    The returned hits, the verdict, and every round.

    Iterating it yields the hits, so it is read like a search response.

    Parameters
    ----------
    query : str
        The query as given.
    verdict : CorrectionVerdict
        What the retrieval came to.
    rounds : tuple of CorrectionRound
        Every round run, in order.
    hits : tuple
        The accepted round's hits; the fused ambiguous rounds' hits for
        ``partial``; none for ``abstained``.
    stop_reason : str
        ``"accepted"``, ``"max_rounds"``, ``"no_rewrite"`` (the rewrite
        proposed no new query), ``"deadline"`` or ``"rewrite_failed"``.
    errors : tuple of ErrorRecord, default=()
        Every error: a round whose search failed or whose response was not
        complete, a grader or rewrite that failed, a fusion that could not
        run, the deadline.
    """

    query: str
    verdict: CorrectionVerdict
    rounds: tuple[CorrectionRound, ...]
    hits: tuple[Any, ...]
    stop_reason: str
    errors: tuple[ErrorRecord, ...] = ()

    @property
    def status(self) -> RetrievalStatus:
        """
        Return the status a search response would have.

        Returns
        -------
        RetrievalStatus
            ``cancelled`` when every round was cancelled (no hit is then
            returned: a cancelled round is not graded); ``degraded`` when any
            error was recorded (every incomplete round response records
            one); otherwise ``success`` with hits and ``empty`` without.
        """
        if self.rounds and all(
            r.status is RetrievalStatus.CANCELLED for r in self.rounds
        ):
            return RetrievalStatus.CANCELLED
        if self.errors:
            return RetrievalStatus.DEGRADED
        return RetrievalStatus.SUCCESS if self.hits else RetrievalStatus.EMPTY

    def __iter__(self) -> Iterator[Any]:
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, index: int) -> Any:
        return self.hits[index]

    def error_records(self) -> list[ErrorRecord]:
        """Return the errors, as a search response does."""
        return list(self.errors)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "query": self.query,
            "verdict": self.verdict.value,
            "status": self.status.value,
            "stop_reason": self.stop_reason,
            "hit_count": len(self.hits),
            "rounds": [r.to_dict() for r in self.rounds],
            "errors": [e.to_dict() for e in self.errors],
        }


def _name(component: Any) -> str:
    return getattr(component, "__name__", None) or type(component).__name__


@dataclasses.dataclass(frozen=True)
class CorrectiveRetrieval:
    """
    Grade, then accept, retry with a rewritten query, or abstain (CRAG).

    Parameters
    ----------
    plan : RetrievalExecutionPlan or None, default=None
        How one round retrieves: ``plan.run(index, query)``. ``None`` is
        ``index.search(query, **search_kwargs)``.
    grader : callable or None, default=None
        ``grader(query, hits) -> GradeResult``. ``None`` is
        ``TermCoverageGrader()``.
    rewrite : callable or None, default=None
        ``rewrite(query, rounds) -> str or None``: the next round's query,
        given the original query and the rounds so far. The proposal is
        stripped of surrounding whitespace; ``None`` (or a rewrite returning
        ``None`` or a query already run) means no retry. It is called only
        while rounds remain and the deadline has not passed.
    max_rounds : int, default=2
        Rounds at most, the first included: the call budget.
    fusion : FusionPolicy or None, default=None
        Fuses the ambiguous rounds (one or more) for a ``partial`` result;
        ``None`` is ``FusionPolicy()``. The leg of round ``i`` is named
        ``"r{i}"``. A fusion that keeps no hit (every leg weighed 0) makes
        the result ``abstained``. Fusing several rounds needs each hit's
        identity (``doc_id``); without one, the first ambiguous round is
        returned unfused and ``FUSE_FAILED`` is recorded. One round is cut
        and weighed by the policy without needing an identity.
    deadline_seconds : float or None, default=None
        No rewrite is asked for and no further round starts after this many
        seconds (checked before the rewrite and again before the round); a
        stop by the deadline is recorded (``DEADLINE_EXCEEDED``), so the
        result is ``degraded``. Without a rewrite there is no further round,
        and no deadline stop.
    on_error : {"degrade", "raise"}, default="degrade"
        A search, grader or rewrite that raises, whatever the reason (a
        keyword argument the index does not take included): recorded, with
        the round graded ``incorrect`` (``"degrade"``), or propagated. A
        failed or cancelled round is never graded: it is not evidence.
    clock : callable or None, default=None
        Seconds; ``None`` is :func:`time.perf_counter`.

    Raises
    ------
    TypeError, ValueError
        If a field is of the wrong type or out of range.

    Examples
    --------
    An index that returns only documents holding every query word finds
    nothing for the first query; the rewrite drops a word:

    >>> from types import SimpleNamespace as NS
    >>> from scikitplot.corpus import CorrectiveRetrieval, TermCoverageGrader
    >>> docs = {"a": "spans are slices", "b": "the cache key is a digest"}
    >>> class Index:
    ...     def search(self, query):
    ...         return [
    ...             NS(doc=NS(doc_id=k, text=t), score=1.0)
    ...             for k, t in docs.items()
    ...             if set(query.split()) <= set(t.split())
    ...         ]
    >>> corrective = CorrectiveRetrieval(
    ...     grader=TermCoverageGrader(accept=0.6),
    ...     rewrite=lambda query, rounds: " ".join(query.split()[:-1]) or None,
    ... )
    >>> result = corrective.run(Index(), "cache key size")
    >>> result.verdict.value, [r.grade.grade.value for r in result.rounds]
    ('corrected', ['incorrect', 'correct'])
    >>> [h.doc.doc_id for h in result]
    ['b']
    """

    plan: Any = None
    grader: Callable[[str, Any], GradeResult] | None = None
    rewrite: Callable[[str, tuple[CorrectionRound, ...]], str | None] | None = None
    max_rounds: int = 2
    fusion: FusionPolicy | None = None
    deadline_seconds: float | None = None
    on_error: str = "degrade"
    clock: Callable[[], float] | None = None

    def __post_init__(self) -> None:
        if self.plan is not None and not callable(getattr(self.plan, "run", None)):
            raise TypeError(
                "CorrectiveRetrieval.plan must have a run method or be None."
            )
        for name in ("grader", "rewrite", "clock"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"CorrectiveRetrieval.{name} must be callable or None.")
        if (
            isinstance(self.max_rounds, bool)
            or not isinstance(self.max_rounds, int)
            or self.max_rounds < 1
        ):
            raise ValueError("CorrectiveRetrieval.max_rounds must be a positive int.")
        if self.fusion is not None and not isinstance(self.fusion, FusionPolicy):
            raise TypeError(
                "CorrectiveRetrieval.fusion must be a FusionPolicy or None."
            )
        if self.deadline_seconds is not None and (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or self.deadline_seconds <= 0
        ):
            raise ValueError(
                "CorrectiveRetrieval.deadline_seconds must be a positive finite number."
            )
        if self.on_error not in ("degrade", "raise"):
            raise ValueError(
                'CorrectiveRetrieval.on_error must be "degrade" or "raise".'
            )

    def bind(
        self, index: Any, **search_kwargs: Any
    ) -> Callable[[str], CorrectionResult]:
        """
        Return ``query -> self.run(index, query, **search_kwargs)``.

        Parameters
        ----------
        index : object
            What every round searches.
        **search_kwargs
            Passed to every round.

        Returns
        -------
        callable
            For :func:`~scikitplot.corpus.evaluate_retrieval`.
        """

        def run(query: str) -> CorrectionResult:
            return self.run(index, query, **search_kwargs)

        run.__name__ = f"corrective({_name(index)})"
        return run

    def run(  # noqa: PLR0912 - one branch per verdict and stop reason
        self, index: Any, query: str, **search_kwargs: Any
    ) -> CorrectionResult:
        """
        Run up to ``max_rounds`` graded rounds for ``query``.

        Parameters
        ----------
        index : object
            Anything with ``search`` (or what ``plan.run`` takes).
        query : str
            A non-empty query.
        **search_kwargs
            Passed to ``index.search`` (or ``plan.run``) in every round.

        Returns
        -------
        CorrectionResult
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("CorrectiveRetrieval.run needs a non-empty query string.")
        grader = TermCoverageGrader() if self.grader is None else self.grader
        grader_name = _name(grader)
        tick = time.perf_counter if self.clock is None else self.clock
        started = tick()
        rounds: list[CorrectionRound] = []
        ranked: list[list[Any]] = []
        errors: list[ErrorRecord] = []
        seen = {query.strip()}
        current = query
        stop = "max_rounds"
        for number in range(self.max_rounds):
            round_start = tick()
            hits, status, problems = self._retrieve(index, current, search_kwargs)
            grade: GradeResult | None = None
            if status not in (RetrievalStatus.FAILED, RetrievalStatus.CANCELLED):
                # A failed or cancelled round is not evidence. Graded against the original query: a rewrite that drifted
                # from it is not accepted for it.
                grade, failure = self._grade(grader, query, hits)
                if failure is not None:
                    problems.append(failure)
            errors.extend(problems)
            if grade is None:
                why = (
                    f"the search {status.value}"
                    if status in (RetrievalStatus.FAILED, RetrievalStatus.CANCELLED)
                    else "the grader failed"
                )
                grade = GradeResult(EvidenceGrade.INCORRECT, 0.0, why)
            if grade.grader is None:
                grade = dataclasses.replace(grade, grader=grader_name)
            rounds.append(
                CorrectionRound(
                    index=number,
                    query=current,
                    grade=grade,
                    status=status,
                    hit_count=len(hits),
                    seconds=max(0.0, tick() - round_start),
                    error=problems[0] if problems else None,
                )
            )
            ranked.append(hits)
            if grade.grade is EvidenceGrade.CORRECT:
                verdict = (
                    CorrectionVerdict.ACCEPTED
                    if number == 0
                    else CorrectionVerdict.CORRECTED
                )
                return CorrectionResult(
                    query,
                    verdict,
                    tuple(rounds),
                    tuple(hits),
                    "accepted",
                    tuple(errors),
                )
            if number + 1 == self.max_rounds:
                break
            if self.rewrite is None:
                stop = "no_rewrite"  # no further round is possible
                break
            if self._past_deadline(tick, started, number + 1, errors):
                # Checked before the rewrite: a model rewrite is a call too.
                stop = "deadline"
                break
            proposal, failure = self._propose(query, tuple(rounds))
            if failure is not None:
                errors.append(failure)
                stop = "rewrite_failed"
                break
            if proposal is None or proposal in seen:
                stop = "no_rewrite"
                break
            if self._past_deadline(tick, started, number + 1, errors):
                # And again before the next round: the rewrite took time.
                stop = "deadline"
                break
            seen.add(proposal)
            current = proposal
        return self._settle(query, rounds, ranked, stop, errors)

    def _past_deadline(
        self,
        tick: Callable[[], float],
        started: float,
        rounds: int,
        errors: list[ErrorRecord],
    ) -> bool:
        """Whether the deadline has passed; if so, record it in ``errors``."""
        if self.deadline_seconds is None or tick() - started < self.deadline_seconds:
            return False
        errors.append(
            ErrorRecord(
                code="DEADLINE_EXCEEDED",
                category=ErrorCategory.RESOURCE,
                message=(
                    f"corrective retrieval stopped after {rounds} round(s): "
                    f"deadline_seconds={self.deadline_seconds!r}"
                ),
                stage="correct",
                details={"rounds": rounds},
            )
        )
        return True

    def _grade(
        self, grader: Callable[[str, Any], GradeResult], query: str, hits: list[Any]
    ) -> tuple[GradeResult | None, ErrorRecord | None]:
        try:
            produced = grader(query, tuple(hits))
            if not isinstance(produced, GradeResult):
                raise TypeError(
                    "the grader must return a GradeResult, got "
                    f"{type(produced).__name__}"
                )
            # Rebuilt as the base class, so its checks run whatever the
            # grader's subclass does.
            grade = GradeResult(
                produced.grade, produced.score, produced.reason, produced.grader
            )
        except Exception as exc:  # noqa: BLE001 - recorded, or re-raised below
            if self.on_error == "raise":
                raise
            return None, ErrorRecord.from_exception(
                exc,
                code="GRADE_FAILED",
                category=ErrorCategory.CAPABILITY,
                stage="correct",
            )
        return grade, None

    def _retrieve(
        self, index: Any, query: str, search_kwargs: Mapping[str, Any]
    ) -> tuple[list[Any], RetrievalStatus | None, list[ErrorRecord]]:
        """Run one round's search: ``(hits, status, errors)``.

        A response that is not complete (``degraded``, ``failed``,
        ``cancelled``, or a status this module cannot read) contributes its
        own error records, or one naming its status when it has none, so a
        degraded result always says why.
        """
        try:
            if self.plan is None:
                response = index.search(query, **search_kwargs)
            else:
                response = self.plan.run(index, query, **search_kwargs)
            hits = list(response)
        except Exception as exc:  # noqa: BLE001 - recorded, or re-raised below
            if self.on_error == "raise":
                raise
            return (
                [],
                RetrievalStatus.FAILED,
                [
                    ErrorRecord.from_exception(
                        exc,
                        code="ROUND_FAILED",
                        category=ErrorCategory.CAPABILITY,
                        stage="correct",
                        details={"query": query},
                    )
                ],
            )
        raw = getattr(response, "status", None)
        try:
            status = None if raw is None else RetrievalStatus(raw)
        except ValueError:
            return (
                hits,
                RetrievalStatus.DEGRADED,
                [
                    ErrorRecord(
                        code="STATUS_UNKNOWN",
                        category=ErrorCategory.VALIDATION,
                        message=f"a round's response has an unknown status {raw!r}",
                        stage="correct",
                        details={"query": query},
                    )
                ],
            )
        if status not in (
            RetrievalStatus.DEGRADED,
            RetrievalStatus.FAILED,
            RetrievalStatus.CANCELLED,
        ):
            return hits, status, []
        records = getattr(response, "error_records", None)
        found = [
            r
            for r in (records() if callable(records) else [])
            if isinstance(r, ErrorRecord)
        ]
        if not found:
            found = [
                ErrorRecord(
                    code="ROUND_INCOMPLETE",
                    category=ErrorCategory.CAPABILITY,
                    message=f"a round's response is {status.value} without a record",
                    stage="correct",
                    details={"query": query},
                )
            ]
        return hits, status, found

    def _propose(
        self, query: str, rounds: tuple[CorrectionRound, ...]
    ) -> tuple[str | None, ErrorRecord | None]:
        if self.rewrite is None:
            return None, None
        try:
            proposal = self.rewrite(query, rounds)
        except Exception as exc:  # noqa: BLE001 - recorded, or re-raised below
            if self.on_error == "raise":
                raise
            return None, ErrorRecord.from_exception(
                exc,
                code="REWRITE_FAILED",
                category=ErrorCategory.CAPABILITY,
                stage="correct",
            )
        if isinstance(proposal, str) and proposal.strip():
            return proposal.strip(), None
        if proposal is not None:
            error = TypeError(
                f"rewrite must return a non-empty str or None, got {proposal!r}"
            )
            if self.on_error == "raise":
                raise error
            return None, ErrorRecord.from_exception(
                error,
                code="REWRITE_FAILED",
                category=ErrorCategory.VALIDATION,
                stage="correct",
            )
        return None, None

    def _settle(
        self,
        query: str,
        rounds: list[CorrectionRound],
        ranked: list[list[Any]],
        stop: str,
        errors: list[ErrorRecord],
    ) -> CorrectionResult:
        """Return the ambiguous rounds fused (``partial``), or ``abstained``."""
        ambiguous = [
            i for i, r in enumerate(rounds) if r.grade.grade is EvidenceGrade.AMBIGUOUS
        ]
        hits: tuple[Any, ...] = ()
        policy = self.fusion or FusionPolicy()
        if ambiguous:
            first: dict[str, Any] = {}
            legs = []
            try:
                for i in ambiguous:
                    keys = []
                    for rank, hit in enumerate(ranked[i], start=1):
                        key = _fusion_key(hit, f"r{i}", rank)
                        first.setdefault(key, hit)
                        keys.append(key)
                    legs.append((f"r{i}", keys))
            except ValueError as exc:  # a hit without an identity
                if len(ambiguous) > 1:
                    if self.on_error == "raise":
                        raise
                    errors.append(
                        ErrorRecord.from_exception(
                            exc,
                            code="FUSE_FAILED",
                            category=ErrorCategory.VALIDATION,
                            stage="correct",
                        )
                    )
                # One round needs no identity: its own order, cut and weighed
                # as fusion would; of several, the first is kept, unfused.
                only = ambiguous[0]
                weight = dict(policy.weights).get(f"r{only}", policy.default_weight)
                kept = [] if weight == 0 else ranked[only]
                hits = tuple(kept if policy.top_k is None else kept[: policy.top_k])
            else:
                hits = tuple(first[item.key] for item in policy.fuse(legs))
        verdict = CorrectionVerdict.PARTIAL if hits else CorrectionVerdict.ABSTAINED
        return CorrectionResult(
            query, verdict, tuple(rounds), hits, stop, tuple(errors)
        )
