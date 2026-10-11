"""
Detectors: pure functions from text to spans.

A detector receives the **original** text and returns
:class:`~scikitplot.cleanprompt._types.Span` objects over it. It never receives
rewritten text, never mutates anything, and never decides what a span becomes.

Notes
-----
**User notes.** Two detectors cover the common cases. :class:`RegexDetector`
wraps an entry of the pattern library; :class:`LiteralDetector` hides exact
strings you name yourself, which is the "additional words to hide" feature.
Build a registry with :func:`default_registry` and add to it.

**Developer notes.** The protocol is deliberately narrow — ``name``, ``kind``,
``detect(text, policy)`` — because widening it is how staged pipelines grow. The
upstream project applied three stages in sequence, each rewriting the string
before handing it to the next, so the named-entity stage saw ``[EMAIL-1]`` and
tagged inside it (defect ``CP-006``). Detectors here compose over *spans*, and
composition happens once, in :mod:`scikitplot.cleanprompt._engine`.

A detector that raises is wrapped in
:class:`~scikitplot.cleanprompt._exceptions.DetectorError` naming it, so one bad
custom detector cannot masquerade as a pipeline bug.

See Also
--------
scikitplot.cleanprompt._patterns : The curated pattern library.
scikitplot.cleanprompt._engine : Composes detector output.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, Sequence

from ._exceptions import (
    CleanPromptError,
    DetectorError,
    LimitExceededError,
    PolicyError,
)
from ._patterns import PatternSpec, default_patterns, get_pattern
from ._policy import RedactionPolicy
from ._types import Span

__all__ = [
    "Detector",
    "DetectorRegistry",
    "LiteralDetector",
    "RegexDetector",
    "default_registry",
]


class Detector:
    """
    Base class for detectors.

    Subclasses implement :meth:`detect`. The base class supplies the identity
    attributes and the error wrapping used by
    :meth:`DetectorRegistry.detect_all`.

    Parameters
    ----------
    name : str
        Unique, stable identifier. Used in arbitration tie-breaks, so it must
        not change between runs.
    kind : str
        Category assigned to every span this detector produces.
    priority : int, default=50
        Arbitration weight.
    confidence : float, default=1.0
        Confidence reported for every span.

    Raises
    ------
    PolicyError
        If ``name`` or ``kind`` is empty, or ``confidence`` is out of range.
    """

    __slots__ = ("confidence", "kind", "name", "priority")

    #: Whether :meth:`detect` is a pure function of the text it is given, so
    #: that it may also read the detection view (``CP-098``) and have its spans
    #: mapped back onto the original. ``False`` by default: a detector bound to
    #: one document — offsets computed in advance from the original, such as a
    #: field or region detector — would have its spans mapped twice and cut
    #: through a file's structure (``CP-102``). Opt in only when every offset a
    #: detector yields indexes the string passed to ``detect``.
    reads_view = False

    def __init__(
        self,
        name: str,
        kind: str,
        priority: int = 50,
        confidence: float = 1.0,
    ) -> None:
        if not name:
            raise PolicyError("detector name must be a non-empty string")
        if not kind:
            raise PolicyError("detector kind must be a non-empty string")
        if not 0.0 <= confidence <= 1.0:
            raise PolicyError(
                f"detector confidence must be in [0.0, 1.0], got {confidence!r}"
            )
        self.name = name
        self.kind = kind
        self.priority = priority
        self.confidence = confidence

    def detect(self, text: str, policy: RedactionPolicy) -> Iterable[Span]:
        """
        Yield spans over ``text``.

        Parameters
        ----------
        text : str
            The original text. Never a rewritten one.
        policy : RedactionPolicy
            The active policy.

        Yields
        ------
        Span
            One per match, in ascending start order.

        Raises
        ------
        NotImplementedError
            Always, on the base class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement detect()")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, kind={self.kind!r})"


class RegexDetector(Detector):
    r"""
    Detector backed by one entry of the pattern library.

    Parameters
    ----------
    spec : PatternSpec
        The pattern specification. Its ``kind``, ``priority`` and ``confidence``
        become the detector's.

    Notes
    -----
    **Developer notes.** A zero-width match cannot produce a
    :class:`~scikitplot.cleanprompt._types.Span`, which requires ``end > start``.
    Rather than let a pattern that can match empty silently produce nothing,
    such matches are skipped explicitly and the scan advances, so a pattern
    author sees "no detections" rather than an infinite loop.

    **Validated patterns are not scanned with** :meth:`re.Pattern.finditer`.
    ``finditer`` consumes each match before the caller sees it and resumes from
    its end, so a match the validator later *rejects* has already swallowed the
    text it covered. Every shorter or later candidate beginning inside that
    span is then never tried.

    That is not a cosmetic difference; it loses detections and therefore leaks.
    Scanning ``"123-45-6789 4242 4242 4242 4242"`` for phone numbers, the greedy
    expression first matches ``"123-45-6789 4242 4242"``, which the validator
    rejects for having nineteen digits. ``finditer`` then resumes past it, and
    the trailing ``"4242 4242"`` — a perfectly good match — is never offered, so
    those characters survive into the redacted output in the clear.

    So a validated pattern is scanned with :meth:`re.Pattern.search` from an
    explicit cursor, and at each start position the scanner looks for the
    longest match the validator *accepts*, not merely the longest match. When
    the greedy match is rejected, the window is shortened and the attempt is
    repeated at the same start before the start is advanced.

    That second step matters as much as the first, because a greedy expression
    over ``digits separator digits`` glues independent numbers together across
    a space. Scanning ``"4477 555 010 4477 2024-01-15"``, the greedy match
    spans the whole string and is rejected for digit count; advancing past it
    finds ``"010 4477 2024-01-15"``, which validates, and strands ``"4477 555"``
    in the clear. Retrying shorter windows at the earliest start finds the
    intended number instead.

    **Window boundaries are re-verified.** Limiting a match with ``endpos``
    makes the engine treat that offset as the end of the string, so a trailing
    ``\b`` or ``(?!\w)`` succeeds there even when the real text continues with
    a word character. Accepting such a match would cut a number in half and
    leave its tail in the clear. Each candidate ending before the end of the
    text is therefore re-matched with one further character visible, and is
    kept only if the engine still ends at the same offset. That is exact for
    the single-character trailing assertions this library uses — ``\b``,
    ``(?!\w)`` and ``(?![\w.])``; a pattern needing wider trailing context
    would have to widen this probe with it.

    Unvalidated patterns keep the cheaper ``finditer``, since nothing can
    reject their matches.

    Examples
    --------
    >>> from ._patterns import get_pattern
    >>> from ._policy import DEFAULT_POLICY
    >>> detector = RegexDetector(get_pattern("EMAIL"))
    >>> [span.text for span in detector.detect("write to a@b.co", DEFAULT_POLICY)]
    ['a@b.co']
    """

    #: Offsets index the text given to ``detect``: may read the detection view.
    reads_view = True

    __slots__ = ("spec",)

    def __init__(self, spec: PatternSpec) -> None:
        super().__init__(
            name=f"regex:{spec.kind}",
            kind=spec.kind,
            priority=spec.priority,
            confidence=spec.confidence,
        )
        self.spec = spec

    def detect(self, text: str, policy: RedactionPolicy) -> Iterator[Span]:
        """
        Yield one span per accepted match.

        Parameters
        ----------
        text : str
            The original text.
        policy : RedactionPolicy
            The active policy. Unused; present for protocol conformance.

        Yields
        ------
        Span
            Accepted matches, in ascending start order.
        """
        del policy  # the pattern library is policy-independent
        validate = self.spec.validate
        pattern = self.spec.compiled()

        if validate is None:
            for match in pattern.finditer(text):
                start, end = match.span()
                if end > start:
                    yield self._span(match, start, end)
            return

        # Validated: find the longest ACCEPTED match at each start position.
        cursor = 0
        length = len(text)
        while cursor <= length:
            found = pattern.search(text, cursor)
            if found is None:
                return
            start = found.start()
            accepted = None
            window = found.end()
            while window > start:
                candidate = pattern.match(text, start, window)
                if candidate is None or candidate.end() <= start:
                    break
                end = candidate.end()
                if end < length and not self._boundary_is_real(
                    pattern,
                    text,
                    start,
                    end,
                ):
                    window = end - 1
                    continue
                if validate(candidate):
                    accepted = candidate
                    break
                window = end - 1
            if accepted is not None:
                yield self._span(accepted, start, accepted.end())
                cursor = accepted.end()
            else:
                cursor = start + 1

    @staticmethod
    def _boundary_is_real(
        pattern: re.Pattern,
        text: str,
        start: int,
        end: int,
    ) -> bool:
        """
        Return whether a match ending at ``end`` really ends there.

        Parameters
        ----------
        pattern : re.Pattern
            The pattern being scanned.
        text : str
            The full original text.
        start : int
            Where the candidate match begins.
        end : int
            Where the candidate match ends, strictly inside ``text``.

        Returns
        -------
        bool
            ``True`` when the engine still ends the match at ``end`` with one
            further character of real context visible.

        Notes
        -----
        **Developer notes.** This distinguishes a genuine boundary from one
        fabricated by the ``endpos`` window. If the pattern can continue past
        ``end`` once the next character is visible, then ``end`` was not a
        boundary at all and the candidate would have cut a value in half.
        """
        probe = pattern.match(text, start, end + 1)
        return probe is not None and probe.end() == end

    def _span(self, match: re.Match, start: int, end: int) -> Span:
        """Build the span for an accepted match."""
        return Span(
            start=start,
            end=end,
            kind=self.kind,
            text=match.group(),
            detector=self.name,
            priority=self.priority,
            confidence=self.confidence,
        )


class LiteralDetector(Detector):
    """
    Detector for exact strings the caller names.

    Parameters
    ----------
    terms : iterable of str
        Strings to hide. Empty and whitespace-only terms are rejected.
    kind : str, default='CUSTOM'
        Category for every match.
    name : str, optional
        Detector name. Defaults to ``"literal:<kind>"``.
    priority : int, default=75
        Arbitration weight. Above the structural patterns by default, because a
        caller naming a term explicitly has expressed stronger intent than a
        generic pattern.
    word_boundary : bool, default=False
        When ``True``, a term matches only at word boundaries.
    ignore_case : bool, default=False
        When ``True``, terms match regardless of case. Normally supplied from
        :attr:`~scikitplot.cleanprompt._policy.RedactionPolicy.case_insensitive`
        so that one setting governs both halves of "case is not significant":
        which surfaces *match*, and which surfaces *share a label*. Implementing
        only the second half would silently ignore ``--hide acme`` against the
        text ``Acme``.

    Raises
    ------
    PolicyError
        If a term is empty or whitespace-only.

    Notes
    -----
    **User notes.** This is the "additional words to hide" feature. Terms are
    matched literally, not as regular expressions, so ``a.b`` matches exactly
    ``a.b`` and not ``axb``.

    **Developer notes.** Two properties matter here, and both were absent
    upstream.

    *Longest-first alternation.* Terms are sorted by descending length inside
    one alternation, so scanning ``"Ann met Anna"`` for ``["Ann", "Anna"]``
    matches ``Anna`` at its position rather than ``Ann`` followed by a stranded
    ``a``. Upstream rewrote the string once per term with :meth:`str.replace`
    and produced ``"[ADDITIONAL-1] met [ADDITIONAL-1]a"`` — defect ``CP-001``,
    which both corrupts the text and leaks the final ``a`` of the secret.

    *One pass.* All terms are scanned in a single alternation, so cost is linear
    in the text rather than linear in ``len(text) * len(terms)``.

    Examples
    --------
    >>> from ._policy import DEFAULT_POLICY
    >>> detector = LiteralDetector(["Ann", "Anna"])
    >>> [(s.start, s.text) for s in detector.detect("Ann met Anna", DEFAULT_POLICY)]
    [(0, 'Ann'), (8, 'Anna')]
    """

    #: Offsets index the text given to ``detect``: may read the detection view.
    reads_view = True

    __slots__ = ("_pattern", "ignore_case", "terms", "word_boundary")

    def __init__(  # ruff: ignore[too-many-positional-arguments]
        self,
        terms: Iterable[str],
        kind: str = "CUSTOM",
        name: str | None = None,
        priority: int = 75,
        word_boundary: bool = False,
        ignore_case: bool = False,
    ) -> None:
        super().__init__(
            name=name or f"literal:{kind}",
            kind=kind,
            priority=priority,
            confidence=1.0,
        )
        cleaned: list[str] = []
        seen = set()
        for term in terms:
            if not isinstance(term, str):
                raise PolicyError(
                    f"literal terms must be strings, got {type(term).__name__!r}"
                )
            stripped = term.strip()
            if not stripped:
                raise PolicyError("literal terms must not be empty or whitespace-only")
            if stripped not in seen:
                seen.add(stripped)
                cleaned.append(stripped)
        # Longest first: the alternation is ordered, and Python's regular
        # expression engine takes the first alternative that matches.
        cleaned.sort(key=lambda term: (-len(term), term))
        self.terms: tuple[str, ...] = tuple(cleaned)
        self.word_boundary = word_boundary
        self.ignore_case = ignore_case
        if self.terms:
            body = "|".join(re.escape(term) for term in self.terms)
            source = rf"\b(?:{body})\b" if word_boundary else f"(?:{body})"
            flags = re.IGNORECASE if ignore_case else 0
            self._pattern: re.Pattern | None = re.compile(source, flags)
        else:
            self._pattern = None

    def detect(self, text: str, policy: RedactionPolicy) -> Iterator[Span]:
        """
        Yield one span per literal occurrence.

        Parameters
        ----------
        text : str
            The original text.
        policy : RedactionPolicy
            The active policy. Unused; present for protocol conformance.

        Yields
        ------
        Span
            Matches in ascending start order, longest alternative first at any
            given position.
        """
        del policy
        if self._pattern is None:
            return
        for match in self._pattern.finditer(text):
            start, end = match.span()
            if end <= start:
                continue
            yield Span(
                start=start,
                end=end,
                kind=self.kind,
                text=match.group(),
                detector=self.name,
                priority=self.priority,
                confidence=1.0,
            )


class DetectorRegistry:
    """
    An ordered, immutable-by-convention collection of detectors.

    Parameters
    ----------
    detectors : iterable of Detector, optional
        Initial detectors.

    Raises
    ------
    PolicyError
        If two detectors share a name.

    Notes
    -----
    **Developer notes.** ``add`` returns ``self`` for chaining but mutates, so a
    registry is a builder, not a value. The engine never mutates one; it reads
    the registry once per pass. Names must be unique because arbitration uses
    the detector name as its final, deterministic tie-break — two detectors
    sharing a name would make the outcome depend on insertion order.
    """

    __slots__ = ("_detectors",)

    def __init__(self, detectors: Iterable[Detector] | None = None) -> None:
        self._detectors: list[Detector] = []
        for detector in detectors or ():
            self.add(detector)

    def add(self, detector: Detector) -> DetectorRegistry:
        """
        Append a detector.

        Parameters
        ----------
        detector : Detector
            The detector to add.

        Returns
        -------
        DetectorRegistry
            This registry, for chaining.

        Raises
        ------
        PolicyError
            If a detector of the same name is already registered.
        """
        if any(existing.name == detector.name for existing in self._detectors):
            raise PolicyError(
                f"a detector named {detector.name!r} is already registered"
            )
        self._detectors.append(detector)
        return self

    def kinds(self) -> tuple[str, ...]:
        """
        Return every kind this registry can produce.

        Returns
        -------
        tuple of str
            Sorted, deduplicated kinds.
        """
        return tuple(sorted({detector.kind for detector in self._detectors}))

    def select(self, kinds: Iterable[str]) -> tuple[Detector, ...]:
        """
        Return the detectors whose kind is in ``kinds``.

        Parameters
        ----------
        kinds : iterable of str
            Kinds to keep.

        Returns
        -------
        tuple of Detector
            In registration order.
        """
        wanted = frozenset(kinds)
        return tuple(d for d in self._detectors if d.kind in wanted)

    def detect_all(
        self,
        text: str,
        policy: RedactionPolicy,
        detectors: Sequence[Detector] | None = None,
    ) -> list[Span]:
        """
        Run detectors and collect every span.

        Parameters
        ----------
        text : str
            The original text.
        policy : RedactionPolicy
            The active policy; supplies :attr:`Limits.max_spans` and
            :attr:`RedactionPolicy.min_confidence`.
        detectors : sequence of Detector, optional
            Override the registry's own detectors.

        Returns
        -------
        list of Span
            Unresolved spans, which may overlap.

        Raises
        ------
        DetectorError
            If a detector raises. The detector name is carried on the error.
        LimitExceededError
            If more than :attr:`Limits.max_spans` spans are produced.

        Notes
        -----
        **Developer notes.** The span cap is checked while collecting, not
        afterwards, so a pathological detector cannot exhaust memory before the
        bound is noticed.
        """
        limit = policy.limits.max_spans
        minimum = policy.min_confidence
        collected: list[Span] = []
        for detector in detectors if detectors is not None else self._detectors:
            try:
                for span in detector.detect(text, policy):
                    if span.confidence < minimum:
                        continue
                    collected.append(span)
                    if len(collected) > limit:
                        raise LimitExceededError(
                            f"detection produced more than {limit} spans; raise "
                            "Limits.max_spans or narrow the detector set",
                            limit_name="max_spans",
                            limit=limit,
                            actual=len(collected),
                        )
            except CleanPromptError:
                # Already one of ours, already attributed and already carrying
                # its own actionable context. Wrapping it here would bury a
                # CapabilityError's install command inside a DetectorError
                # message, making the actionable text unreachable in practice —
                # the same failure mode as an actionable message placed after
                # the import that raises first.
                raise
            except Exception as exc:  # noqa: BLE001 - attributed, not suppressed
                raise DetectorError(
                    f"detector {detector.name!r} failed: {type(exc).__name__}: {exc}",
                    detector=detector.name,
                ) from exc
        return collected

    def __len__(self) -> int:
        return len(self._detectors)

    def __iter__(self) -> Iterator[Detector]:
        return iter(self._detectors)

    def __repr__(self) -> str:
        return "DetectorRegistry({} detector(s): {})".format(
            len(self._detectors),
            ", ".join(detector.name for detector in self._detectors),
        )


def default_registry(
    kinds: Iterable[str] | None = None,
    literal_terms: Iterable[str] | None = None,
    literal_kind: str = "CUSTOM",
    word_boundary: bool = False,
    ignore_case: bool = False,
) -> DetectorRegistry:
    """
    Build a registry from the pattern library, plus optional literal terms.

    Parameters
    ----------
    kinds : iterable of str, optional
        Pattern kinds to include. ``None`` selects
        :func:`~scikitplot.cleanprompt._patterns.default_patterns`.
    literal_terms : iterable of str, optional
        Exact strings to hide, added as a :class:`LiteralDetector`.
    literal_kind : str, default='CUSTOM'
        Category for the literal detector.
    word_boundary : bool, default=False
        Passed to the literal detector.
    ignore_case : bool, default=False
        Passed to the literal detector.

    Returns
    -------
    DetectorRegistry
        A registry containing one :class:`RegexDetector` per selected pattern
        and, when terms were given, one :class:`LiteralDetector`.

    Raises
    ------
    PatternError
        If a requested kind is not in the pattern library.

    Examples
    --------
    >>> registry = default_registry(kinds=["EMAIL"], literal_terms=["Acme"])
    >>> len(registry)
    2
    """
    specs = (
        default_patterns()
        if kinds is None
        else tuple(get_pattern(kind) for kind in kinds)
    )
    registry = DetectorRegistry(RegexDetector(spec) for spec in specs)
    terms = tuple(literal_terms or ())
    if terms:
        registry.add(
            LiteralDetector(
                terms,
                kind=literal_kind,
                word_boundary=word_boundary,
                ignore_case=ignore_case,
            )
        )
    return registry
