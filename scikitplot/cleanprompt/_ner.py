"""
Optional named-entity detector backed by spaCy (tier: ``ner``).

Detects people, organisations, places and other linguistic entities that no
regular expression can find.

Notes
-----
**User notes.** This tier needs ``spacy`` *and* a downloaded model::

    pip install "spacy>=3.4,<5"
    python -m spacy download en_core_web_lg

Then::

    from scikitplot.cleanprompt import Redactor, spacy_detector, default_registry

    registry = default_registry()
    registry.add(spacy_detector())
    result = Redactor(registry=registry).redact(text)

**Developer notes.** Nothing in this module runs at import time. ``spacy`` is
imported inside :meth:`NerDetector._ensure_pipeline`, on the first call to
:meth:`NerDetector.detect`, and the capability is checked *before* that import
so the actionable message is reachable. Importing ``scikitplot.cleanprompt``
therefore costs nothing even on a machine where spaCy is installed.

The detector reports spans over the original text and never rewrites anything,
so it composes with the structural patterns instead of running after them. That
is what stops a model from tagging the inside of a placeholder.

Model loading is expensive — hundreds of megabytes and seconds of wall clock —
so a loaded pipeline is cached per ``(model, tuple(sorted(disable)))`` at module
level. The cache is keyed on the arguments rather than held on the instance so
that constructing several detectors over the same model does not reload it.

See Also
--------
scikitplot.cleanprompt._detectors : The detector protocol this implements.
scikitplot.cleanprompt._capabilities : The tier probe used here.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from ._capabilities import require
from ._detectors import Detector
from ._engines import DEFAULT_ENTITY_LABELS as _CANONICAL_DEFAULT
from ._engines import ENGINES, canonical_label, trim_entity_span
from ._exceptions import CapabilityError, DetectorError
from ._languages import resolve_model
from ._policy import RedactionPolicy
from ._types import Span

__all__ = [
    "DEFAULT_ENTITY_LABELS",
    "DEFAULT_MODEL",
    "NerDetector",
    "spacy_detector",
]

#: Default spaCy model when no language and no explicit model is given.
#:
#: ``sm`` rather than the upstream project's ``lg``: it is a tenth of the size,
#: it is what ``python -m spacy download`` fetches fastest, and a user who
#: wants the larger model can ask for ``--model-size lg``. Defaulting to a
#: 560 MB download is a poor first experience for a tool whose base tier needs
#: no download at all.
DEFAULT_MODEL = "en_core_web_sm"

#: Remedy offered when spaCy is present in metadata but will not import.
#: ``--force-reinstall`` is deliberate: an ordinary install is a no-op against
#: a distribution pip already believes is satisfied, which is precisely the
#: state this message is reporting.
_REINSTALL_SPACY = 'pip install --force-reinstall "spacy>=3.4,<4"'

#: Entity labels redacted by default, in the canonical vocabulary.
#:
#: Re-exported from :mod:`scikitplot.cleanprompt._engines` rather than defined
#: here, so that spaCy and NLTK share one default. A second copy is how the two
#: engines would drift apart, and a placeholder that depends on which engine
#: happened to be installed is exactly what the canonical vocabulary exists to
#: prevent.
DEFAULT_ENTITY_LABELS = _CANONICAL_DEFAULT

#: Process-wide pipeline cache, keyed on ``(model, disabled components)``.
_PIPELINES: dict[tuple[str, tuple[str, ...]], Any] = {}


class NerDetector(Detector):
    """
    Detect named entities with a spaCy pipeline.

    Parameters
    ----------
    model : str, default='en_core_web_lg'
        Name of an installed spaCy model.
    labels : iterable of str, optional
        Entity labels to redact. ``None`` redacts every label the model
        reports. Defaults to :data:`DEFAULT_ENTITY_LABELS`.
    kind_prefix : str, default=''
        Prepended to the spaCy label to form the placeholder category, so
        ``kind_prefix="NE_"`` yields ``[NE_PERSON-1]``.
    priority : int, default=30
        Arbitration weight. Below the structural patterns by default: an email
        address recognised by both should be labelled ``EMAIL``, not ``ORG``.
    confidence : float, default=0.7
        Reported confidence. spaCy's entity recogniser does not expose a
        calibrated per-entity score, so this is a fixed, documented value used
        for filtering and reporting, never presented as a probability.
    disable : iterable of str, optional
        Pipeline components to disable at load time. Defaults to disabling
        ``lemmatizer`` and ``textcat``, which the entity recogniser does not
        need.

    Raises
    ------
    CapabilityError
        On first detection, if the ``ner`` tier is unavailable or the model is
        not installed.

    Notes
    -----
    **Developer notes.** ``sentencizer``-style components are left alone; only
    components irrelevant to entity recognition are disabled, because disabling
    ``tok2vec`` or ``ner`` itself would silently produce zero entities, which in
    a privacy tool reads as "nothing sensitive found".

    Examples
    --------
    >>> detector = NerDetector()  # doctest: +SKIP
    >>> detector.kind  # doctest: +SKIP
    'NE'
    """

    __slots__ = ("_disable", "kind_prefix", "labels", "language", "model", "resolution")

    def __init__(  # ruff: ignore[too-many-positional-arguments]
        self,
        model: str | None = None,
        labels: Iterable[str] | None = DEFAULT_ENTITY_LABELS,
        kind_prefix: str = "",
        priority: int | None = None,
        confidence: float | None = None,
        disable: Iterable[str] | None = None,
        language: str = "en",
        size: str = "sm",
    ) -> None:
        resolved, note = resolve_model(language, size, explicit=model)
        spec = ENGINES["spacy"]
        super().__init__(
            name=f"ner:{resolved}",
            kind="NE",
            priority=spec.priority if priority is None else priority,
            confidence=spec.confidence if confidence is None else confidence,
        )
        self.model = resolved
        self.language = language
        self.resolution = note
        self.labels: frozenset[str] | None = (
            None if labels is None else frozenset(labels)
        )
        self.kind_prefix = kind_prefix
        self._disable: tuple[str, ...] = tuple(
            sorted(disable if disable is not None else ("lemmatizer", "textcat"))
        )

    def kinds(self) -> tuple[str, ...]:
        """
        Return the placeholder categories this detector can produce.

        Returns
        -------
        tuple of str
            Sorted category names, or an empty tuple when every label is
            accepted and the set is therefore open.

        Notes
        -----
        **Developer notes.** A registry needs this because
        :attr:`Detector.kind` is a single value, whereas this detector produces
        one category per entity label. :meth:`detect` sets each span's ``kind``
        from the entity label, and the registry selects this detector by its
        declared ``kind`` of ``"NE"``.
        """
        if self.labels is None:
            return ()
        return tuple(sorted(self.kind_prefix + label for label in self.labels))

    def _ensure_pipeline(self) -> Any:
        """
        Load the spaCy pipeline, checking the tier first.

        Returns
        -------
        spacy.language.Language
            The loaded pipeline.

        Raises
        ------
        CapabilityError
            If ``spacy`` is unusable, or the model is not installed.

        Notes
        -----
        **Developer notes.** The capability check precedes the import, and the
        import precedes the model load. Each failure therefore produces the
        message that matches its own cause: "install spacy", then "download the
        model". Putting the actionable message after the import would make it
        unreachable, because the import raises first.
        """
        key = (self.model, self._disable)
        cached = _PIPELINES.get(key)
        if cached is not None:
            return cached

        require("ner")  # raises CapabilityError before spacy is imported

        try:
            import spacy  # noqa: PLC0415 - deliberately deferred to call time
        except ImportError as exc:
            # Installed according to its metadata, yet it will not import. This
            # is the BROKEN state the capability vocabulary exists to name: a
            # half-finished upgrade, a wheel built for another interpreter, a
            # shadowing file. Letting the raw ImportError escape turns it into
            # a generic "detector failed", which sends the reader looking at
            # this submodule for a fault that is in their environment.
            raise CapabilityError(
                "spaCy reports itself installed but cannot be imported "
                f"({type(exc).__name__}: {exc}). The installation is broken; reinstall it with: "
                f"{_REINSTALL_SPACY}",
                tier="ner",
                status="BROKEN",
                install_hint=_REINSTALL_SPACY,
            ) from exc

        try:
            pipeline = spacy.load(self.model, disable=list(self._disable))
        except OSError as exc:
            raise CapabilityError(
                f"spaCy model {self.model!r} is not installed; download it with: "
                f"python -m spacy download {self.model}",
                tier="ner",
                status="MISCONFIGURED",
                install_hint=f"python -m spacy download {self.model}",
            ) from exc
        _PIPELINES[key] = pipeline
        return pipeline

    def detect(self, text: str, policy: RedactionPolicy) -> Iterator[Span]:
        """
        Yield one span per accepted named entity.

        Parameters
        ----------
        text : str
            The original text.
        policy : RedactionPolicy
            Supplies :attr:`Limits.max_input_chars`, which is propagated to the
            pipeline's own document ceiling.

        Yields
        ------
        Span
            Entities whose label is selected, in ascending start order.

        Raises
        ------
        CapabilityError
            If the tier or the model is unavailable.
        DetectorError
            If the pipeline raises while processing the text.

        Notes
        -----
        **Developer notes.** spaCy's ``max_length`` guard exists to stop a very
        long document exhausting memory during parsing. It is aligned with this
        submodule's own bound so that one limit governs, and a document that the
        base tier accepts is not rejected by the NER tier at a different size.

        Entities reported by spaCy never overlap, so the spans yielded here are
        already disjoint among themselves; they may still overlap structural
        detections, which is what
        :func:`~scikitplot.cleanprompt._engine.resolve_spans` settles.
        """
        pipeline = self._ensure_pipeline()
        ceiling = policy.limits.max_input_chars
        if getattr(pipeline, "max_length", 0) < ceiling:
            pipeline.max_length = ceiling
        try:
            document = pipeline(text)
        except Exception as exc:  # noqa: BLE001 - attributed, not suppressed
            raise DetectorError(
                f"spaCy pipeline {self.model!r} failed: {type(exc).__name__}: {exc}",
                detector=self.name,
            ) from exc

        selected = self.labels
        for entity in document.ents:
            label = canonical_label(entity.label_)
            if selected is not None and label not in selected:
                continue
            start, end = trim_entity_span(text, entity.start_char, entity.end_char)
            if end <= start:
                continue  # the span held only brackets
            yield Span(
                start=start,
                end=end,
                kind=self.kind_prefix + label,
                text=text[start:end],
                detector=self.name,
                priority=self.priority,
                confidence=self.confidence,
            )


def spacy_detector(  # ruff: ignore[undocumented-param]
    model: str | None = None,
    labels: Iterable[str] | None = DEFAULT_ENTITY_LABELS,
    language: str = "en",
    size: str = "sm",
    **kwargs: Any,
) -> NerDetector:
    """
    Build a :class:`NerDetector`.

    Parameters
    ----------
    model : str, default='en_core_web_lg'
        Name of an installed spaCy model.
    labels : iterable of str, optional
        Entity labels to redact.
    **kwargs
        Forwarded to :class:`NerDetector`.

    Returns
    -------
    NerDetector
        The detector. Construction does not import spaCy or load the model;
        both happen on the first :meth:`NerDetector.detect` call.

    Examples
    --------
    >>> detector = spacy_detector()
    >>> detector.name
    'ner:en_core_web_lg'
    """
    return NerDetector(
        model=model, labels=labels, language=language, size=size, **kwargs
    )
