"""
Immutable value types carried through the redaction pipeline.

Every type here is a frozen dataclass: the pipeline is a sequence of pure
transformations over values, and nothing downstream may mutate what an earlier
stage produced.

Notes
-----
**User notes.** :class:`RedactionResult` is what you get back from
:meth:`~scikitplot.cleanprompt._engine.Redactor.redact`. Its ``text`` is safe to
send; its ``vault`` is not. :class:`RestorationResult` is what you get back from
:func:`~scikitplot.cleanprompt._engine.restore`.

**Developer notes.** A :class:`Span` is a half-open ``[start, end)`` interval of
*character* offsets into the original text. All arithmetic in the engine is done
on these offsets, never on rewritten strings. Working on offsets over one
immutable input is what makes the pipeline's invariants provable: detections
cannot see each other's output, and a shorter secret cannot corrupt a longer one
during rewriting.

Offsets are Python string indices, which count code points. A grapheme cluster
(an emoji with a modifier, a combining accent) may span several code points; a
detector that reports a span must therefore report code-point offsets, which is
what :mod:`re` yields naturally.

See Also
--------
scikitplot.cleanprompt._engine : Consumes and produces these types.
scikitplot.cleanprompt._vault : Holds the secrets referenced by :class:`Entry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "Entry",
    "RedactionResult",
    "RestorationResult",
    "Span",
    "Stats",
    "as_dict",
]


@dataclass(frozen=True, order=True)
class Span:
    """
    A half-open ``[start, end)`` character interval flagged by one detector.

    Parameters
    ----------
    start : int
        Inclusive start offset into the original text. Must be ``>= 0``.
    end : int
        Exclusive end offset. Must be ``> start``.
    kind : str
        Category label used to build the placeholder, for example ``"EMAIL"``.
        Must be non-empty and contain no placeholder delimiter characters.
    text : str
        The matched surface, exactly as it appears in the original.
    detector : str
        Name of the detector that produced this span.
    priority : int, default=0
        Arbitration weight. Higher wins under
        :attr:`~scikitplot.cleanprompt._policy.OverlapStrategy.PRIORITY_WINS`.
    confidence : float, default=1.0
        Detector-reported confidence in ``[0.0, 1.0]``. Carried for reporting
        and filtering; it never silently drops a span.

    Raises
    ------
    ValueError
        If the interval is empty or negative, if ``kind`` is empty, or if
        ``confidence`` is outside ``[0.0, 1.0]``.

    Notes
    -----
    **Developer notes.** The field order gives a meaningful ``order=True``: spans
    sort by start, then by end. Resolution relies on that natural order and then
    applies its own tie-breaks, so sorting is stable and reproducible across
    processes regardless of hash randomisation.

    Examples
    --------
    >>> Span(0, 5, "EMAIL", "a@b.c", "email").length
    5
    """

    start: int
    end: int
    kind: str
    text: str = field(compare=False)
    detector: str = field(compare=False)
    priority: int = field(default=0, compare=False)
    confidence: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"Span.start must be >= 0, got {self.start!r}")
        if self.end <= self.start:
            raise ValueError(
                f"Span must be non-empty: got start={self.start!r}, end={self.end!r}"
            )
        if not self.kind:
            raise ValueError("Span.kind must be a non-empty string")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Span.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )

    @property
    def length(self) -> int:
        """int: Number of characters covered by this span."""
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        """
        Return whether this span shares at least one character with ``other``.

        Parameters
        ----------
        other : Span
            The span to test against.

        Returns
        -------
        bool
            ``True`` when the half-open intervals intersect.
        """
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class Entry:
    """
    One resolved detection and the placeholder label assigned to it.

    Parameters
    ----------
    label : str
        The rendered placeholder, for example ``"[EMAIL-1]"``.
    kind : str
        Category of the detection.
    ordinal : int
        1-based index within ``kind``, in first-appearance order.
    original : str
        The secret surface this label stands for.
    occurrences : tuple of tuple of int
        Every ``(start, end)`` offset in the original text that this label
        replaced, in ascending order.
    detector : str
        Name of the detector that produced the first occurrence.
    confidence : float
        Confidence of the first occurrence.

    Notes
    -----
    **Developer notes.** One :class:`Entry` per *distinct value*, not per
    occurrence: repeating a name ten times yields one entry with ten
    occurrences and one label. That is what makes the redacted text readable to
    a language model, and it is why ``occurrences`` is a tuple rather than a
    single offset pair.
    """

    label: str
    kind: str
    ordinal: int
    original: str = field(repr=False)
    occurrences: tuple[tuple[int, int], ...]
    detector: str
    confidence: float = 1.0

    @property
    def count(self) -> int:
        """int: How many times this value was replaced."""
        return len(self.occurrences)


@dataclass(frozen=True)
class Stats:
    """
    Counters describing one redaction pass.

    Parameters
    ----------
    input_chars : int
        Length of the input text.
    output_chars : int
        Length of the redacted text.
    detected_spans : int
        Spans produced by detectors, before overlap resolution.
    resolved_spans : int
        Spans surviving overlap resolution.
    dropped_spans : int
        Spans discarded by overlap resolution.
    entries : int
        Distinct values redacted.
    by_kind : dict of str to int
        Distinct values per category.
    """

    input_chars: int
    output_chars: int
    detected_spans: int
    resolved_spans: int
    dropped_spans: int
    entries: int
    by_kind: Mapping[str, int]


@dataclass(frozen=True)
class RedactionResult:
    """
    Output of a redaction pass.

    Parameters
    ----------
    text : str
        The redacted text. Safe to transmit.
    entries : tuple of Entry
        Ordered record of every value replaced, sorted by first occurrence.
    vault : Vault
        The label-to-secret mapping needed to restore. **Not** safe to transmit.
    stats : Stats
        Counters for this pass.
    policy_fingerprint : str
        Stable digest of the policy that produced this result. Restoration
        checks it, so a vault cannot be applied under a different placeholder
        grammar than the one that created it.
    truncated : bool, default=False
        Always ``False``. Present so that callers can assert the pipeline never
        truncates; a bound is exceeded by raising, not by dropping content.

    See Also
    --------
    scikitplot.cleanprompt._engine.restore : The inverse operation.
    """

    text: str
    entries: tuple[Entry, ...]
    vault: Any
    stats: Stats
    policy_fingerprint: str
    truncated: bool = False

    @property
    def labels(self) -> tuple[str, ...]:
        """Tuple of str: Every placeholder label issued, in order."""
        return tuple(entry.label for entry in self.entries)

    def summary(self) -> str:
        """
        Return a one-line, secret-free description of this pass.

        Returns
        -------
        str
            A summary naming counts and categories but never the secrets.

        Notes
        -----
        **Developer notes.** Upstream printed
        ``"Removed sensitive information: " + ", ".join(mapping.keys())``, which
        echoes every secret to the terminal and therefore into shell history and
        any captured log. This deliberately names categories and counts only.
        """
        if not self.entries:
            return "no sensitive values detected"
        parts = [
            f"{kind}={count}" for kind, count in sorted(self.stats.by_kind.items())
        ]
        return "redacted {} value(s): {}".format(len(self.entries), ", ".join(parts))


@dataclass(frozen=True)
class RestorationResult:
    """
    Output of a restoration pass.

    Parameters
    ----------
    text : str
        The text with placeholders replaced by their original values.
    restored : tuple of str
        Labels that were found and replaced, in first-appearance order.
    unknown : tuple of str
        Labels that matched the placeholder grammar but were not in the vault.
        Empty unless restoration ran in non-strict mode.
    unused : tuple of str
        Vault labels that did not appear in the text.
    repaired : tuple of tuple of str
        ``(as the reply spelled it, the label it was taken to mean)`` for every
        placeholder that had to be matched leniently, in first-appearance
        order. Empty when the reply came back with its labels intact.

    Notes
    -----
    **User notes.** A non-empty ``unknown`` means the reply referred to a
    placeholder this vault never issued; the model may have invented it. A
    non-empty ``unused`` is normal: a summary need not mention every entity.

    ``repaired`` is worth reading. It lists the places where the model rewrote
    a placeholder and restoration matched it anyway — ``[EMAIL_1]`` taken to
    mean ``[EMAIL-1]``. The substitution happened, and the entry says so rather
    than leaving you to notice.
    """

    text: str
    restored: tuple[str, ...]
    unknown: tuple[str, ...] = ()
    unused: tuple[str, ...] = ()
    repaired: tuple[tuple[str, str], ...] = ()

    @property
    def complete(self) -> bool:
        """bool: ``True`` when every placeholder found was resolved."""
        return not self.unknown


def as_dict(obj: Any) -> dict[str, Any]:
    """
    Return a JSON-safe, secret-free dictionary for a result object.

    Parameters
    ----------
    obj : RedactionResult or RestorationResult or Stats
        The value to describe.

    Returns
    -------
    dict
        A plain dictionary. Secrets are never included: :class:`Entry` is
        rendered without its ``original`` field.

    Raises
    ------
    TypeError
        If ``obj`` is not one of the supported result types.

    Notes
    -----
    **Developer notes.** This is the only sanctioned serializer for results.
    :func:`dataclasses.asdict` would walk into :class:`Entry.original` and emit
    the secrets, which is exactly the mistake that made the upstream web tier
    ship personal data to the browser.
    """
    if isinstance(obj, Stats):
        return {
            "input_chars": obj.input_chars,
            "output_chars": obj.output_chars,
            "detected_spans": obj.detected_spans,
            "resolved_spans": obj.resolved_spans,
            "dropped_spans": obj.dropped_spans,
            "entries": obj.entries,
            "by_kind": dict(obj.by_kind),
        }
    if isinstance(obj, RedactionResult):
        return {
            "text": obj.text,
            "entries": [
                {
                    "label": entry.label,
                    "kind": entry.kind,
                    "ordinal": entry.ordinal,
                    "count": entry.count,
                    "detector": entry.detector,
                    "confidence": entry.confidence,
                }
                for entry in obj.entries
            ],
            "stats": as_dict(obj.stats),
            "policy_fingerprint": obj.policy_fingerprint,
            "truncated": obj.truncated,
        }
    if isinstance(obj, RestorationResult):
        return {
            "text": obj.text,
            "restored": list(obj.restored),
            "unknown": list(obj.unknown),
            "unused": list(obj.unused),
            "complete": obj.complete,
        }
    raise TypeError(f"unsupported result type: {type(obj).__name__!r}")
