"""
The redaction engine: resolve, assign, rewrite — and its exact inverse.

This module owns every algorithmic decision in the submodule. It is deliberately
the only place where a string is rebuilt.

Notes
-----
**User notes.** :class:`Redactor` is the entry point. Construct one, call
:meth:`Redactor.redact`, transmit ``result.text``, and pass ``result.vault`` to
:func:`restore` when the reply comes back.

**Developer notes — the pipeline and why it has this shape.**

::

    text ──► detect ──► reserve ──► resolve ──► assign ──► rewrite
             (pure)     (grammar)   (disjoint)  (labels)   (one pass)

*Detect* runs every detector against the **original** text. No detector sees
another's output. Upstream chained three rewriting stages, so the named-entity
stage received ``"Contact [EMAIL-1] now"`` and produced
``"Contact [[ORG-1]-1] now"``, destroying a placeholder that could then never be
restored. Composing over spans instead of over strings removes that failure
class rather than patching an instance of it.

*Reserve* finds substrings of the original text that already match the active
placeholder grammar. Their ranges are excluded from detection, which makes
redaction idempotent, and the labels they occupy are excluded from allocation,
so a freshly issued label can never collide with one that was already in the
text.

*Resolve* turns a possibly overlapping span set into a disjoint one. Overlaps
are grouped into connected clusters. When one span covers its whole cluster —
the ordinary nested case, such as an email inside a URL — that span wins
outright. When no span covers the cluster, the cluster is *merged* into a single
span over its full extent. Merging rather than discarding is the safety-relevant
choice: discarding a partially overlapping span would leave the characters it
alone covered in the output, which is a disclosure. Merging cannot disclose,
and it round-trips exactly, because the merged surface is what the vault stores.

*Assign* gives one label per distinct value, so a name repeated ten times reads
as one entity to the language model.

*Rewrite* walks the disjoint, ascending spans once and concatenates. It is O(n)
and it cannot let one replacement be re-matched as part of another — the defect
behind ``"Ann met Anna"`` becoming ``"[ADDITIONAL-1] met [ADDITIONAL-1]a"``,
which both corrupted the text and leaked the final character of a secret.

*Restore* is the mirror: one compiled scan for the placeholder grammar, one
dictionary lookup per hit, one pass to rebuild. Its cost does not grow with the
number of entries, and no restored value can be re-matched as a placeholder.

See Also
--------
scikitplot.cleanprompt._policy : The declarative configuration this consumes.
scikitplot.cleanprompt._detectors : Produces the spans this resolves.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ._canonical import canonical, detection_view, normal_form, value_pattern
from ._detectors import DetectorRegistry, LiteralDetector, default_registry
from ._exceptions import LimitExceededError, OverlapError, PolicyError, RestorationError
from ._policy import (
    DEFAULT_POLICY,
    OverlapStrategy,
    RedactionPolicy,
    allowed_surfaces,
)
from ._surrogates import surrogate_for
from ._types import Entry, RedactionResult, RestorationResult, Span, Stats
from ._vault import Vault

if TYPE_CHECKING:  # pragma: no cover - annotations only
    # Named in string annotations on the private restoration helpers. Imported
    # here rather than at runtime so the names resolve for a type checker and
    # for Sphinx without adding an import this module does not otherwise need.
    from ._policy import TagStyle

__all__ = [
    "Redactor",
    "reserved_label_spans",
    "resolve_spans",
    "restore",
]


# ---------------------------------------------------------------------------
# reservation
# ---------------------------------------------------------------------------


def reserved_label_spans(
    text: str, policy: RedactionPolicy
) -> tuple[tuple[tuple[int, int], ...], set[str]]:
    """
    Find substrings of ``text`` that already match the placeholder grammar.

    Parameters
    ----------
    text : str
        The original text.
    policy : RedactionPolicy
        Supplies the placeholder grammar.

    Returns
    -------
    ranges : tuple of tuple of int
        ``(start, end)`` pairs to exclude from detection, ascending.
    labels : set of str
        The label strings occupying those ranges, excluded from allocation.

    Notes
    -----
    **Developer notes.** Both halves matter and they solve different problems.
    Excluding the *ranges* gives idempotence: a detector cannot tag the inside of
    a placeholder. Excluding the *labels* prevents an aliasing bug: for the input
    ``"[EMAIL-1] and bob@example.com"`` the real address would otherwise be
    issued ``[EMAIL-1]`` as well, and restoration would then rewrite both
    occurrences to the same value.
    """
    if not policy.preserve_placeholders:
        return (), set()
    ranges: list[tuple[int, int]] = []
    labels: set[str] = set()
    for match in policy.tag_style.pattern().finditer(text):
        ranges.append(match.span())
        labels.add(match.group())
    return tuple(ranges), labels


def _in_reserved(span: Span, ranges: Sequence[tuple[int, int]]) -> bool:
    """Return whether ``span`` intersects any reserved range."""
    return any(span.start < end and start < span.end for start, end in ranges)


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------


def _sort_key(span: Span) -> tuple[int, int, int, str]:
    """Deterministic ordering key: earliest, then longest, then most specific."""
    return (span.start, -span.end, -span.priority, span.detector)


def _cluster_winner(cluster: Sequence[Span], strategy: OverlapStrategy) -> Span:
    """
    Pick the span that decides a cluster's kind.

    Parameters
    ----------
    cluster : sequence of Span
        Two or more mutually connected spans.
    strategy : OverlapStrategy
        ``LONGEST_WINS`` or ``PRIORITY_WINS``.

    Returns
    -------
    Span
        The winner. Ties break on detector name, which is unique within a
        registry, so the result never depends on insertion order.
    """
    if strategy is OverlapStrategy.PRIORITY_WINS:
        return min(cluster, key=lambda s: (-s.priority, -s.length, s.detector))
    return min(cluster, key=lambda s: (-s.length, -s.priority, s.detector))


def resolve_spans(
    spans: Iterable[Span],
    policy: RedactionPolicy = DEFAULT_POLICY,
    text: str | None = None,
) -> tuple[Span, ...]:
    """
    Reduce possibly overlapping spans to a disjoint, ascending sequence.

    Parameters
    ----------
    spans : iterable of Span
        Detections over one text. May overlap in any pattern.
    policy : RedactionPolicy, default=DEFAULT_POLICY
        Supplies :attr:`~scikitplot.cleanprompt._policy.RedactionPolicy.overlap`.
    text : str, optional
        The source text the spans index into. Required only when a cluster has
        to be merged, because the merged span's
        :attr:`~scikitplot.cleanprompt._types.Span.text` must be the real
        surface rather than a member's.

    Returns
    -------
    tuple of Span
        Pairwise disjoint spans in ascending start order. Every returned span's
        ``text`` equals ``text[start:end]`` when ``text`` was supplied.

    Raises
    ------
    OverlapError
        If two spans overlap and the strategy is
        :attr:`~scikitplot.cleanprompt._policy.OverlapStrategy.STRICT`.
    PolicyError
        If a cluster must be merged but ``text`` was not supplied.

    Notes
    -----
    **Developer notes.** The guarantee is *coverage preservation*: every
    character covered by any input span is covered by exactly one output span.
    That is what makes the redaction safe under partial overlap, and it is
    checked directly by ``test__engine.py::test_resolution_preserves_coverage``.

    Identical duplicate spans from two detectors collapse into one cluster and
    therefore into one output span, so a value matched by both a pattern and a
    literal term is replaced once.

    Examples
    --------
    >>> a = Span(0, 10, "URL", "x" * 10, "regex:URL")
    >>> b = Span(4, 8, "EMAIL", "x" * 4, "regex:EMAIL")
    >>> [(s.start, s.end, s.kind) for s in resolve_spans([a, b])]
    [(0, 10, 'URL')]
    """
    ordered = sorted(spans, key=_sort_key)
    if not ordered:
        return ()

    strategy = policy.overlap
    resolved: list[Span] = []
    cluster: list[Span] = [ordered[0]]
    cluster_end = ordered[0].end

    def flush(group: list[Span], end: int) -> None:
        """Emit one span for a connected cluster."""
        if len(group) == 1:
            resolved.append(group[0])
            return
        if strategy is OverlapStrategy.STRICT:
            first, second = group[0], group[1]
            raise OverlapError(
                "overlapping detections under STRICT policy: "
                f"{first.kind!r} at [{first.start}, {first.end}) and {second.kind!r} at [{second.start}, {second.end})",
                spans=(
                    (first.start, first.end, first.kind),
                    (second.start, second.end, second.kind),
                ),
            )
        winner = _cluster_winner(group, strategy)
        start = group[0].start
        if winner.start == start and winner.end == end:
            resolved.append(winner)
            return
        # No member covers the cluster. Merge, so that no covered character is
        # left in the output. Discarding here would be a disclosure.
        if text is None:
            raise PolicyError(
                f"spans at [{start}, {end}) overlap only partially and must be merged, "
                "but resolve_spans was called without the source text"
            )
        resolved.append(
            Span(
                start=start,
                end=end,
                kind=winner.kind,
                text=text[start:end],
                detector=winner.detector,
                priority=winner.priority,
                confidence=winner.confidence,
            )
        )

    for span in ordered[1:]:
        if span.start < cluster_end:
            cluster.append(span)
            cluster_end = max(cluster_end, span.end)
        else:
            flush(cluster, cluster_end)
            cluster = [span]
            cluster_end = span.end
    flush(cluster, cluster_end)
    return tuple(resolved)


# ---------------------------------------------------------------------------
# the redactor
# ---------------------------------------------------------------------------


class Redactor:
    """
    Reusable, stateless redactor.

    Parameters
    ----------
    policy : RedactionPolicy, optional
        Configuration. Defaults to
        :data:`~scikitplot.cleanprompt._policy.DEFAULT_POLICY`.
    registry : DetectorRegistry, optional
        Detectors to run. Defaults to
        :func:`~scikitplot.cleanprompt._detectors.default_registry` restricted
        to the policy's kinds.

    Raises
    ------
    PolicyError
        If the policy names a kind the registry cannot produce.

    Notes
    -----
    **User notes.** One redactor may be reused for any number of documents, from
    any number of threads. Each call returns its own result and its own vault;
    nothing carries over between documents.

    **Developer notes.** Statelessness is the contract, not an implementation
    detail. Upstream kept ``counters`` and ``found_items`` on the instance and
    never reset them, so a second document continued the first document's
    numbering — and the web tier shared one instance across every HTTP request,
    which made one visitor's data influence another's output. Every counter here
    lives in a local of :meth:`redact`.

    Examples
    --------
    >>> redactor = Redactor()
    >>> first = redactor.redact("mail a@b.co")
    >>> second = redactor.redact("mail c@d.co")
    >>> first.text, second.text
    ('mail [EMAIL-1]', 'mail [EMAIL-1]')
    >>> restore(first.text, first.vault).text
    'mail a@b.co'
    """

    __slots__ = ("_kinds", "policy", "registry")

    def __init__(
        self,
        policy: RedactionPolicy | None = None,
        registry: DetectorRegistry | None = None,
    ) -> None:
        self.policy = policy if policy is not None else DEFAULT_POLICY
        if registry is None:
            registry = default_registry(kinds=self.policy.kinds)
        self.registry = registry
        self._kinds = self.policy.selected_kinds(registry.kinds())

    def redact(
        self,
        text: str,
        extra_terms: Iterable[str] | None = None,
        extra_kind: str = "CUSTOM",
        word_boundary: bool = False,
        seed: Iterable[Entry] | None = None,
    ) -> RedactionResult:
        """
        Replace sensitive values in ``text`` with stable placeholders.

        Parameters
        ----------
        text : str
            The text to redact.
        extra_terms : iterable of str, optional
            Additional exact strings to hide in this call only. They are
            detected in the same pass as everything else.
        extra_kind : str, default='CUSTOM'
            Category for ``extra_terms``.
        word_boundary : bool, default=False
            Whether ``extra_terms`` match only at word boundaries.
        seed : iterable of Entry, optional
            Label assignments from an earlier pass. A value that already has a
            label keeps it, and fresh values are numbered after the highest
            ordinal already used for their kind.

        Returns
        -------
        RedactionResult
            The redacted text, the ordered entries, the vault and the stats.

        Raises
        ------
        TypeError
            If ``text`` is not a string.
        LimitExceededError
            If a policy bound is exceeded.
        PolicyError
            If ``extra_terms`` contains an empty or non-string entry.
        OverlapError
            Under :attr:`OverlapStrategy.STRICT` when detections overlap.
        DetectorError
            If a detector raises.

        Notes
        -----
        **User notes.** ``seed`` is what makes a multi-turn conversation
        coherent: pass the previous result's entries and the same person keeps
        the same placeholder across every prompt, so a model reading the thread
        sees one consistent participant rather than a new stranger each turn.

        **Developer notes.** ``seed`` is an *argument*, not instance state, and
        that distinction is the whole point. Statelessness (invariant I4) means
        this object carries nothing between calls, not that continuity is
        impossible; making the caller pass the history explicitly keeps two
        documents independent unless someone deliberately links them. Holding
        it on the instance instead is precisely the upstream defect ``CP-003``.

        **User notes.** Allowlisted surfaces
        (:attr:`~scikitplot.cleanprompt._policy.RedactionPolicy.allow`) are
        dropped here, before overlap resolution. Filtering at this point rather
        than afterwards means an allowed value cannot be swept into a merged
        span by a neighbour, which would have redacted it anyway and made the
        allowlist look unreliable.

        **User notes.** ``extra_terms`` is the "additional words to hide"
        feature, and it composes with everything else in one pass. Upstream had
        a separate code path for it which re-ran the wrong method and silently
        skipped email, phone and URL detection entirely — so supplying an extra
        term made the result *less* redacted. There is one pipeline here, and
        adding a term can only add detections.

        Examples
        --------
        >>> result = Redactor().redact("Ann met Anna", extra_terms=["Ann", "Anna"])
        >>> result.text
        '[CUSTOM-1] met [CUSTOM-2]'
        >>> restore(result.text, result.vault).text
        'Ann met Anna'
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}")
        policy = self.policy
        limits = policy.limits
        if len(text) > limits.max_input_chars:
            raise LimitExceededError(
                f"input is {len(text)} characters, above the limit of {limits.max_input_chars}; raise "
                "Limits.max_input_chars or split the document",
                limit_name="max_input_chars",
                limit=limits.max_input_chars,
                actual=len(text),
            )

        detectors = list(self.registry.select(self._kinds))
        terms = tuple(extra_terms or ())
        if terms:
            self._check_terms(terms, limits)
            detectors.append(
                LiteralDetector(
                    terms,
                    kind=extra_kind,
                    word_boundary=word_boundary,
                    ignore_case=policy.case_insensitive,
                )
            )

        reserved_ranges, reserved_labels = reserved_label_spans(text, policy)

        detected = self.registry.detect_all(text, policy, detectors=detectors)
        detected.extend(self._view_spans(text, detected, detectors))
        allow = allowed_surfaces(policy)
        fold = policy.case_insensitive
        candidate = [
            span
            for span in detected
            if not _in_reserved(span, reserved_ranges)
            and (
                not allow or (span.text.casefold() if fold else span.text) not in allow
            )
        ]
        resolved = resolve_spans(candidate, policy, text=text)

        return self._assign_and_rewrite(
            text=text,
            resolved=resolved,
            detected_count=len(detected),
            reserved_labels=reserved_labels,
            seed=tuple(seed or ()),
        )

    # -- internals --------------------------------------------------------

    def _view_spans(
        self,
        text: str,
        detected: Sequence[Span],
        detectors: Sequence[Any],
    ) -> list[Span]:
        """
        Return what structural and literal detectors find in the detection view.

        Parameters
        ----------
        text : str
            The original text.
        detected : sequence of Span
            What the original-text pass found; exact repeats are not returned.
        detectors : sequence of Detector
            The detectors of this call.

        Returns
        -------
        list of Span
            New spans, as offsets into ``text`` with the original surface.

        Raises
        ------
        LimitExceededError
            If the two passes together exceed :attr:`Limits.max_spans`.

        Notes
        -----
        **Developer notes — CP-098.** Measured on the untouched tree:
        ``ada\u200b@example.com``, a full-width address, ``+1 555\u00a00100``,
        ``+1\u2011555\u20110100`` and ``192.0.2.\u200b10`` all went to the model
        in the clear, and ``4111\u200b1111 1111 1111`` came out as
        ``4111\u200b[PHONE-1]`` — the card's first group sent, the rest
        mislabelled. Each value reads the same to a person and to a model.

        The fix keeps invariant I6 — detectors never see rewritten text — by
        not rewriting it: :func:`~scikitplot.cleanprompt._canonical.detection_view`
        is a second *reading*, and every span it yields is mapped back onto
        the original before it joins the others. Only detectors that declare
        ``reads_view`` (structural patterns, pack patterns and literal terms)
        read it. Entity engines do not: they are statistical models of natural
        text and already read it as written. Detectors bound to one document —
        field, region and JSON-token detectors, whose offsets were computed
        from the original in advance — do not either: reading the view, their
        spans were mapped a second time and cut through a record file's
        newlines and separators (``CP-102``). The view's spans are additions only, and the resolver
        merges overlaps, so this pass can widen redaction and cannot narrow it.
        """
        view = detection_view(text)
        if view is None:
            return []
        # Only detectors whose offsets index the text they are given (CP-102):
        # a detector bound to the original document would be mapped twice.
        readers = [d for d in detectors if getattr(d, "reads_view", False)]
        if not readers:
            return []
        policy = self.policy
        seen = {(span.start, span.end, span.kind) for span in detected}
        added: list[Span] = []
        for span in self.registry.detect_all(view.text, policy, detectors=readers):
            start, end = view.source_span(span.start, span.end)
            if (start, end, span.kind) in seen:
                continue
            seen.add((start, end, span.kind))
            added.append(
                Span(
                    start,
                    end,
                    span.kind,
                    text[start:end],
                    span.detector,
                    span.priority,
                    span.confidence,
                )
            )
        limit = policy.limits.max_spans
        if len(detected) + len(added) > limit:
            raise LimitExceededError(
                f"detection produced more than {limit} spans; raise "
                "Limits.max_spans or narrow the detector set",
                limit_name="max_spans",
                limit=limit,
                actual=len(detected) + len(added),
            )
        return added

    @staticmethod
    def _check_terms(terms: Sequence[str], limits) -> None:
        """Validate ``extra_terms`` against the policy bounds."""
        if len(terms) > limits.max_literal_terms:
            raise LimitExceededError(
                f"{len(terms)} literal terms supplied, above the limit of {limits.max_literal_terms}",
                limit_name="max_literal_terms",
                limit=limits.max_literal_terms,
                actual=len(terms),
            )
        for term in terms:
            if isinstance(term, str) and len(term) > limits.max_literal_length:
                raise LimitExceededError(
                    f"a literal term is {len(term)} characters, above the limit of "
                    f"{limits.max_literal_length}",
                    limit_name="max_literal_length",
                    limit=limits.max_literal_length,
                    actual=len(term),
                )

    def _assign_and_rewrite(
        self,
        text: str,
        resolved: Sequence[Span],
        detected_count: int,
        reserved_labels: set[str],
        seed: tuple[Entry, ...] = (),
    ) -> RedactionResult:
        """Allocate labels and rebuild the text in one left-to-right pass."""
        policy = self.policy
        style = policy.tag_style
        fingerprint = policy.fingerprint
        fold = policy.case_insensitive

        labels_by_value: dict[tuple[str, str], str] = {}
        issued: set[str] = set()
        ordinals: dict[str, int] = {}
        order: list[tuple[str, str]] = []
        originals: dict[str, str] = {}
        occurrences: dict[str, list[tuple[int, int]]] = {}
        provenance: dict[str, tuple[str, float]] = {}
        ordinal_of: dict[str, int] = {}

        pieces: list[str] = []
        cursor = 0
        vault = Vault(grammar_fingerprint=style.fingerprint)

        # Seeded assignments come first, so an earlier pass's labels are reused
        # rather than reissued, and new values are numbered after them.
        for prior in seed:
            key = (
                prior.kind,
                prior.original.casefold() if fold else prior.original,
            )
            labels_by_value.setdefault(key, prior.label)
            issued.add(prior.label)
            ordinals[prior.kind] = max(ordinals.get(prior.kind, 0), prior.ordinal)
            reserved_labels.add(prior.label)
            originals.setdefault(prior.label, prior.original)
            ordinal_of.setdefault(prior.label, prior.ordinal)
            provenance.setdefault(prior.label, (prior.detector, prior.confidence))

        seeded = len(labels_by_value)
        held_pattern = None
        for span in resolved:
            surface = text[span.start : span.end]
            key = (span.kind, surface.casefold() if fold else surface)
            label = labels_by_value.get(key)
            if label is not None and label not in occurrences:
                # Reused from the seed: register it in this pass's bookkeeping.
                occurrences[label] = []
                order.append(key)
                originals[label] = surface
                provenance[label] = (span.detector, span.confidence)
                vault.add(label, surface)
            if label is None:
                # Only values found in *this* text count against the bound.
                # Counting the seed too meant a conversation or a large file
                # encoded in pieces stopped working once its history reached
                # the limit, however small each new text was (CP-064).
                if len(labels_by_value) - seeded >= policy.limits.max_entries:
                    raise LimitExceededError(
                        f"more than {policy.limits.max_entries} distinct values detected; raise "
                        "Limits.max_entries",
                        limit_name="max_entries",
                        limit=policy.limits.max_entries,
                        actual=len(labels_by_value) - seeded + 1,
                    )
                ordinal = ordinals.get(span.kind, 0) + 1
                label = style.render(span.kind, ordinal)
                while label in reserved_labels:
                    # The input already contained this label. Skip the ordinal
                    # rather than issue a colliding one.
                    ordinal += 1
                    label = style.render(span.kind, ordinal)
                if style.style == "surrogate":
                    # An invented value the model will neither rewrite nor
                    # comment on. It falls back to the label for kinds where a
                    # plausible-looking stand-in would be a hazard rather than
                    # a convenience; see _surrogates for which and why.
                    if held_pattern is None:
                        # Every value this text or the conversation holds;
                        # a stand-in containing one would show it (CP-071).
                        held_pattern = value_pattern(
                            [
                                *originals.values(),
                                *(text[r.start : r.end] for r in resolved),
                            ]
                        )
                    stand_in = surrogate_for(
                        span.kind,
                        ordinal,
                        avoid=issued,
                        source=text,
                        forbidden=lambda one: bool(
                            held_pattern.search(  # ruff: ignore[function-uses-loop-variable]
                                canonical(one),
                            )
                        ),
                    )
                    if stand_in is not None:
                        label = stand_in
                        issued.add(stand_in)
                ordinals[span.kind] = ordinal
                labels_by_value[key] = label
                order.append(key)
                originals[label] = surface
                occurrences[label] = []
                provenance[label] = (span.detector, span.confidence)
                ordinal_of[label] = ordinal
                vault.add(label, surface)

            occurrences[label].append((span.start, span.end))
            pieces.append(text[cursor : span.start])
            pieces.append(label)
            cursor = span.end

        pieces.append(text[cursor:])
        redacted = "".join(pieces)

        entries: list[Entry] = []
        by_kind: dict[str, int] = {}
        for key in order:
            kind = key[0]
            label = labels_by_value[key]
            detector, confidence = provenance[label]
            entries.append(
                Entry(
                    label=label,
                    kind=kind,
                    ordinal=ordinal_of[label],
                    original=originals[label],
                    occurrences=tuple(occurrences[label]),
                    detector=detector,
                    confidence=confidence,
                )
            )
            by_kind[kind] = by_kind.get(kind, 0) + 1

        stats = Stats(
            input_chars=len(text),
            output_chars=len(redacted),
            detected_spans=detected_count,
            resolved_spans=len(resolved),
            dropped_spans=max(0, detected_count - len(resolved)),
            entries=len(entries),
            by_kind=by_kind,
        )
        return RedactionResult(
            text=redacted,
            entries=tuple(entries),
            vault=vault,
            stats=stats,
            policy_fingerprint=fingerprint,
        )

    def __repr__(self) -> str:
        return (
            f"Redactor(kinds={sorted(self._kinds)}, policy={self.policy.fingerprint})"
        )


# ---------------------------------------------------------------------------
# restoration
# ---------------------------------------------------------------------------


#: Longest whitespace run a model may put between the words of a literal
#: stand-in and still have it restored (``CP-072``): a line break and an
#: indentation. Bounded, so a stream decoder knows how far back one can begin.
RESTORE_MAX_GAP = 16


def _variant_keys(keys: Iterable[str]) -> dict[str, str]:
    """
    Map each literal key's normal form to the key, dropping ambiguous forms.

    Notes
    -----
    **Developer notes.** Two keys that differ only in case or spacing cannot
    be told apart once a model has rewritten one, so neither is restored from
    a rewritten form; each still restores when written exactly.
    """
    found: dict[str, str | None] = {}
    for key in keys:
        form = normal_form(key)
        found[form] = key if form not in found else None
    return {form: key for form, key in found.items() if key is not None}


@lru_cache(maxsize=64)
def _restore_keys(style: TagStyle, labels: tuple[str, ...]) -> dict[str, str]:
    """Return :func:`_variant_keys` over a vault's literal keys, once per key set."""
    return _variant_keys(label for label in labels if style.normalize(label) is None)


@lru_cache(maxsize=64)
def _literal_matchers(style: TagStyle, labels: tuple[str, ...]) -> tuple[Any, Any]:
    """
    Return the exact and the rewritten-stand-in matchers for a vault's keys.

    Parameters
    ----------
    style : TagStyle
        Decides which keys are bracket labels (matched by the grammar).
    labels : tuple of str
        Every key in the vault, sorted.

    Returns
    -------
    tuple
        ``(detector, variants)``: a
        :class:`~scikitplot.cleanprompt._detectors.LiteralDetector` over the
        literal keys and the :func:`value_pattern` for their rewritten forms,
        each ``None`` when there are no literal keys.

    Notes
    -----
    **Developer notes.** A stream decoder asks on every chunk and a vault only
    grows, so the same key set recurs; building both once per set keeps a long
    stream's cost per chunk independent of how many keys the vault holds
    (measured: rebuilding them was most of each chunk's time). Ambiguous
    normal forms are left out of the variant matcher (:func:`_variant_keys`).
    The cache holds vault *keys* — labels and invented stand-ins — never a
    removed value, so nothing secret outlives a cleared vault in it.
    """
    from ._detectors import (  # ruff: ignore[import-outside-top-level]
        LiteralDetector,
    )

    literal = [label for label in labels if style.normalize(label) is None]
    if not literal:
        return None, None
    return (
        LiteralDetector(literal, kind="SURROGATE"),
        value_pattern(_variant_keys(literal).values(), max_gap=RESTORE_MAX_GAP),
    )


def _restoration_candidates(  # ruff: ignore[undocumented-param]
    text: str,
    vault: Vault,
    style: TagStyle,
    scanner: re.Pattern,
    lenient: bool = True,
) -> list[tuple[int, int, str]]:
    """
    Return every place in ``text`` that may name a vault entry.

    Parameters
    ----------
    text : str
        The reply being restored.
    vault : Vault
        Supplies the labels to look for.
    style : TagStyle
        The grammar the labels were issued under.
    scanner : re.Pattern
        The label recogniser, exact or lenient.

    Returns
    -------
    list of tuple
        ``(start, end, matched text)`` in ascending, non-overlapping order.

    Notes
    -----
    **Developer notes — why this is not two passes.**

    Under ``style="surrogate"`` a vault key is an ordinary phrase, not a
    bracket label, so the grammar scanner cannot see it. The obvious fix is a
    second pass over the result of the first, and it is wrong: a restored value
    may itself contain a phrase that looks like another key, and the second
    pass would rewrite text the first pass had just put there. That is
    ``CP-006`` reappearing on the restoration side.

    So both sources are collected over the *original* reply, merged, and
    rewritten once. Literal keys are found with
    :class:`~scikitplot.cleanprompt._detectors.LiteralDetector`, which already
    solves the substring problem that ``CP-001`` is named for: it matches the
    longest key first, so a stand-in like ``Marion`` cannot claim part of
    ``Marion Holt``.

    Where a label and a literal overlap, the longer wins, and a tie goes to
    whichever starts first. A grammar label and a surrogate cannot plausibly
    overlap in practice; the rule is stated so that the outcome is defined
    rather than incidental.
    """
    candidates: list[tuple[int, int, str]] = [
        (match.start(), match.end(), match.group()) for match in scanner.finditer(text)
    ]

    detector, variants = _literal_matchers(style, tuple(sorted(vault.labels())))
    if detector is not None:
        candidates.extend(
            (span.start, span.end, span.text)
            for span in detector.detect(text, DEFAULT_POLICY)
        )
        if lenient:
            # A stand-in the model re-cased or re-spaced (CP-072). Added to
            # the exact matches, never instead of them, so nothing that
            # restored before stops restoring; the longer candidate still wins.
            candidates.extend(
                (match.start(), match.end(), text[match.start() : match.end()])
                for match in variants.finditer(canonical(text))
                if vault.get(text[match.start() : match.end()]) is None
            )

    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    chosen: list[tuple[int, int, str]] = []
    cursor = 0
    for start, end, found in candidates:
        if start < cursor:
            continue  # already covered by a longer, earlier candidate
        chosen.append((start, end, found))
        cursor = end
    return chosen


def restore(  # ruff: ignore[too-many-branches]
    text: str,
    vault: Vault,
    policy: RedactionPolicy | None = None,
    strict: bool = False,
    lenient: bool = True,
) -> RestorationResult:
    """
    Replace placeholders in ``text`` with the values held in ``vault``.

    Parameters
    ----------
    text : str
        Text containing placeholders, typically a language model's reply.
    vault : Vault
        The vault produced by the matching redaction.
    policy : RedactionPolicy, optional
        Supplies the placeholder grammar. Defaults to
        :data:`~scikitplot.cleanprompt._policy.DEFAULT_POLICY`. Its fingerprint
        is checked against the vault's.
    strict : bool, default=False
        When ``True``, a placeholder that is not in the vault raises.
    lenient : bool, default=True
        Also match placeholders the model rewrote — a different case, a
        different separator, escaped brackets, a line break inside the label.
        Each one acted on is listed in
        :attr:`~scikitplot.cleanprompt._types.RestorationResult.repaired`. Set
        ``False`` to match only labels spelled exactly as they were issued.

    Returns
    -------
    RestorationResult
        The restored text plus which labels were restored, unknown and unused.

    Raises
    ------
    TypeError
        If ``text`` is not a string.
    PolicyError
        If the vault is closed, or was issued under a different placeholder
        grammar.
    RestorationError
        Under ``strict`` when a placeholder is not in the vault.

    Notes
    -----
    **User notes.** ``strict=False`` is the default because a reply may
    legitimately contain placeholder-shaped text the vault never issued — for
    instance if your original text already contained one. Unknown labels are
    left exactly as they are and reported in
    :attr:`~scikitplot.cleanprompt._types.RestorationResult.unknown`; check that
    field rather than assuming completeness.

    **Developer notes.** One compiled scan, one dictionary lookup per hit, one
    join. Replacing each known label in turn would be O(entries × length) and
    would let a restored value containing a placeholder be re-matched on a later
    iteration.

    The fingerprint checked here is the *grammar* digest
    (:attr:`~scikitplot.cleanprompt._policy.TagStyle.fingerprint`), not the full
    policy digest. Only the grammar can change how a label is spelled, so
    narrowing ``kinds`` or raising a limit between redaction and restoration is
    legitimate and must not be rejected.

    Examples
    --------
    >>> result = Redactor().redact("call +1 555 010 4477")
    >>> restore(result.text, result.vault).text
    'call +1 555 010 4477'
    >>> restore("[EMAIL-9] unknown", result.vault).unknown
    ('[EMAIL-9]',)
    """  # ruff: ignore[ambiguous-unicode-character-docstring]
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__!r}")
    if vault.closed:
        raise PolicyError("cannot restore from a cleared Vault")
    active = policy if policy is not None else DEFAULT_POLICY
    expected = vault.grammar_fingerprint
    if expected is not None and expected != active.tag_style.fingerprint:
        raise PolicyError(
            f"this vault was issued under placeholder grammar {expected} but "
            f"restoration was asked for grammar {active.tag_style.fingerprint}; the labels in the text "
            "cannot be read with this TagStyle"
        )

    style = active.tag_style
    exact = style.pattern()
    scanner = style.lenient_pattern() if lenient else exact

    pieces: list[str] = []
    restored: list[str] = []
    unknown: list[str] = []
    repaired: list[tuple[str, str]] = []
    seen_restored: set[str] = set()
    seen_unknown: set[str] = set()
    seen_repaired: set[str] = set()
    cursor = 0

    variant_keys = (
        _restore_keys(style, tuple(sorted(vault.labels()))) if lenient else {}
    )
    for start, end, found in _restoration_candidates(
        text, vault, style, scanner, lenient
    ):
        # "Exact" means spelled the way this grammar renders it, which is not
        # the same as matching the exact pattern: that pattern's category class
        # accepts any case, so a lower-cased [email-1] matches it and then
        # misses the vault. Normalising and comparing is the test that means
        # what it says.
        label = style.normalize(found) or found if lenient else found
        is_exact = label == found
        if vault.get(found) is not None:
            # A surrogate is its own key: the phrase in the reply *is* what the
            # vault is keyed on, so no normalisation applies to it.
            label = found
            is_exact = True
        elif lenient and normal_form(found) in variant_keys:
            # The model re-cased or re-spaced a stand-in (CP-072): restore it
            # and report the repair, as for a rewritten bracket label.
            label = variant_keys[normal_form(found)]
            is_exact = False
        secret = vault.get(label)

        if secret is None and not is_exact:
            # A rewritten-looking token that resolves to nothing is far more
            # likely to be ordinary prose than a mangled placeholder, so it is
            # left exactly as it is and not reported. Only labels this grammar
            # really issued are ever called unknown.
            continue

        pieces.append(text[cursor:start])
        if secret is None:
            if strict:
                raise RestorationError(
                    f"placeholder {label!r} is not present in the vault",
                    labels=(label,),
                )
            pieces.append(found)
            if label not in seen_unknown:
                seen_unknown.add(label)
                unknown.append(label)
        else:
            pieces.append(secret)
            if label not in seen_restored:
                seen_restored.add(label)
                restored.append(label)
            if not is_exact and found not in seen_repaired:
                seen_repaired.add(found)
                repaired.append((found, label))
        cursor = end

    pieces.append(text[cursor:])
    unused = tuple(label for label in vault.labels() if label not in seen_restored)
    return RestorationResult(
        text="".join(pieces),
        restored=tuple(restored),
        unknown=tuple(unknown),
        unused=unused,
        repaired=tuple(repaired),
    )
