"""
Named-entity detection with NLTK (tier: ``nltk``).

Notes
-----
**User notes.** Lighter than spaCy and English-only. Install the tier and the
corpora it needs::

    pip install "nltk>=3.6,<4"
    python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','maxent_ne_chunker','maxent_ne_chunker_tab','words')]"

Then::

    python -m scikitplot.cleanprompt redact --in p.txt --vault v.json --ner --ner-engine nltk

**Developer notes — the hard part is offsets, not entities.**

:func:`nltk.ne_chunk` returns a tree of *tokens*. It does not say where in the
source text those tokens were, and this pipeline is built entirely on character
spans over the original string. Recovering the offsets by searching for the
entity text would be wrong in the ordinary case, never mind the adversarial
one: a name appearing twice would always resolve to the first occurrence, and
NLTK's tokenizer rewrites some characters, so the token text is not always a
substring of the source at all.

So the offsets are *carried*, never recovered:

1. :meth:`PunktSentenceTokenizer.span_tokenize` gives sentence spans;
2. :meth:`TreebankWordTokenizer.span_tokenize` gives token spans within each
   sentence — the span-aware variant, which does not perform the destructive
   quote and bracket rewriting that :func:`nltk.word_tokenize` does;
3. the tagger and chunker run over those tokens, and the chunk tree is walked
   while counting token positions, so every entity subtree maps back to the
   first token's start and the last token's end.

Every span therefore satisfies ``text[span.start:span.end]`` being the entity as
it appears in the source. ``test__nltk.py`` asserts exactly that over text
containing tabs, newlines, quotation marks, accented letters and emoji.

**Label normalisation.** NLTK reports ``ORGANIZATION``, ``LOCATION``,
``FACILITY`` and ``GSP`` where spaCy reports ``ORG``, ``LOC``, ``FAC`` and
``GPE``. Both are mapped to one canonical vocabulary by
:func:`~scikitplot.cleanprompt._engines.canonical_label`, so a vault written
under one engine restores under the other.

**Quality, honestly.** On one measured sentence NLTK split ``Mustafa Kemal
Atatürk`` into two entities and labelled the verb ``Mail`` as a person. It is a
1990s-vintage chunker and it shows. It earns its place because it is small, has
no compiled dependencies, and is the difference between *some* name detection
and none on a machine that cannot take spaCy.

See Also
--------
scikitplot.cleanprompt._ner : The spaCy detector.
scikitplot.cleanprompt._engines : The canonical vocabulary both map onto.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from ._capabilities import require
from ._detectors import Detector
from ._engines import DEFAULT_ENTITY_LABELS, ENGINES, canonical_label, trim_entity_span
from ._exceptions import CapabilityError, DetectorError
from ._logging import get_logger
from ._policy import RedactionPolicy
from ._types import Span

__all__ = ["REQUIRED_CORPORA", "NltkDetector", "nltk_detector"]

logger = get_logger(__name__)

#: NLTK data packages the chunker needs, with the resource path that proves
#: each one is present.
#:
#: Both the legacy and the ``_tab`` names are listed because NLTK 3.8.2 split
#: several packages, and which one a given installation has depends on its
#: version. Either satisfying the lookup is enough, so they are grouped.
REQUIRED_CORPORA: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("punkt", ("tokenizers/punkt_tab", "tokenizers/punkt")),
    (
        "averaged_perceptron_tagger",
        (
            "taggers/averaged_perceptron_tagger_eng",
            "taggers/averaged_perceptron_tagger",
        ),
    ),
    (
        "maxent_ne_chunker",
        ("chunkers/maxent_ne_chunker_tab", "chunkers/maxent_ne_chunker"),
    ),
    ("words", ("corpora/words",)),
)

#: Remedy offered when NLTK is present in metadata but will not import.
_REINSTALL_NLTK = 'pip install --force-reinstall "nltk>=3.6,<4"'

#: Cached NLTK machinery: the two tokenizers and the chunker. All three are
#: stateless with respect to the text and expensive to build, so they are built
#: once per process.
#:
#: The chunker is the one that matters. :func:`nltk.ne_chunk` constructs a fresh
#: ``Maxent_NE_Chunker`` on **every call**, and this detector calls it once per
#: sentence. Measured on NLTK 3.10.3: constructing the chunker takes 460 ms and
#: parsing a sentence with it takes 1 ms, so a ten-sentence prompt spent 4.6
#: seconds rebuilding a model that never changes — long enough to look like a
#: hang in the web interface, and enough to make the lighter engine the slower
#: one by two orders of magnitude. Holding one chunker turns that into 10 ms.
_RESOURCES: dict[str, Any] = {}


class NltkDetector(Detector):
    """
    Detect named entities with NLTK's maximum-entropy chunker.

    Parameters
    ----------
    labels : iterable of str, optional
        Canonical labels to keep. ``None`` keeps every label reported.
        Defaults to
        :data:`~scikitplot.cleanprompt._engines.DEFAULT_ENTITY_LABELS`.
    language : str, default='en'
        Language code. Anything but ``"en"`` raises, because the chunker is
        trained on English only.
    kind_prefix : str, default=''
        Prepended to the canonical label to form the placeholder category.
    priority : int, optional
        Arbitration weight. Defaults to the engine's declared priority, which
        is below the structural patterns so an address recognised by both is
        labelled ``EMAIL`` rather than ``ORG``.
    confidence : float, optional
        Reported confidence. Defaults to the engine's declared value, which is
        lower than spaCy's — a documented judgement about relative quality, not
        a calibrated probability.

    Raises
    ------
    CapabilityError
        On first detection, if the tier or its corpora are unavailable, or if
        a language other than English was requested.

    Examples
    --------
    >>> detector = NltkDetector()
    >>> detector.name
    'nltk'
    """

    __slots__ = ("kind_prefix", "labels", "language")

    def __init__(
        self,
        labels: Iterable[str] | None = DEFAULT_ENTITY_LABELS,
        language: str = "en",
        kind_prefix: str = "",
        priority: int | None = None,
        confidence: float | None = None,
    ) -> None:
        spec = ENGINES["nltk"]
        super().__init__(
            name="nltk",
            kind="NE",
            priority=spec.priority if priority is None else priority,
            confidence=spec.confidence if confidence is None else confidence,
        )
        self.labels: frozenset[str] | None = (
            None if labels is None else frozenset(labels)
        )
        self.language = language
        self.kind_prefix = kind_prefix

    def kinds(self) -> tuple[str, ...]:
        """Return the placeholder categories this detector can produce."""
        if self.labels is None:
            return ()
        return tuple(sorted(self.kind_prefix + label for label in self.labels))

    def _ensure_ready(self) -> tuple[Any, Any, Any, Any]:
        """
        Check the tier, the language and the corpora, then load NLTK.

        Returns
        -------
        tuple
            ``(nltk, sentence_tokenizer, word_tokenizer, chunker)``. ``chunker``
            is ``None`` on an NLTK whose chunker cannot be built directly, in
            which case the caller falls back to :func:`nltk.ne_chunk`.

        Raises
        ------
        CapabilityError
            If the tier is unavailable, the language unsupported, or a corpus
            missing. Each case names its own remedy.

        Notes
        -----
        **Developer notes.** The three checks are ordered so that each failure
        produces the message that matches its cause: "install nltk", then
        "NLTK cannot do this language", then "download this corpus". Collapsing
        them would send a user with a missing corpus to reinstall a package
        they already have.
        """
        if self.language != "en":
            raise CapabilityError(
                f"NLTK's chunker is trained on English only; {self.language!r} was "
                f"requested. Use --ner-engine spacy with --lang {self.language}, or "
                "--ner-engine none.",
                tier="nltk",
                status="MISCONFIGURED",
                install_hint=f"--ner-engine spacy --lang {self.language}",
            )

        require("nltk")  # raises CapabilityError before nltk is imported

        try:
            import nltk  # noqa: PLC0415 - deliberately deferred to call time
        except ImportError as exc:
            # See the matching note in _ner.py: metadata says installed, the
            # import says otherwise. That is BROKEN, not ABSENT, and the two
            # need different remedies.
            raise CapabilityError(
                "NLTK reports itself installed but cannot be imported "
                f"({type(exc).__name__}: {exc}). The installation is broken; reinstall it with: "
                f"{_REINSTALL_NLTK}",
                tier="nltk",
                status="BROKEN",
                install_hint=_REINSTALL_NLTK,
            ) from exc

        missing = _missing_corpora(nltk)
        if missing:
            msg = (
                "NLTK is installed but {} data package(s) are missing: {}. "
                'Download them with: python -c "import nltk; {}"'.format(
                    len(missing),
                    ", ".join(missing),
                    "; ".join(f"nltk.download('{m}')" for m in missing),
                )
            )
            raise CapabilityError(
                msg,
                tier="nltk",
                status="MISCONFIGURED",
                install_hint=(
                    'python -c "import nltk; '
                    + "; ".join(f"nltk.download('{m}')" for m in missing)
                    + '"'
                ),
            )

        if "sentence" not in _RESOURCES:
            from nltk.tokenize.punkt import PunktSentenceTokenizer  # noqa: PLC0415
            from nltk.tokenize.treebank import TreebankWordTokenizer  # noqa: PLC0415

            _RESOURCES["sentence"] = PunktSentenceTokenizer()
            _RESOURCES["word"] = TreebankWordTokenizer()
            _RESOURCES["chunker"] = _build_chunker()
            logger.debug(
                "nltk resources built (chunker cached: %s)",
                _RESOURCES["chunker"] is not None,
            )

        return (
            nltk,
            _RESOURCES["sentence"],
            _RESOURCES["word"],
            _RESOURCES["chunker"],
        )

    def detect(self, text: str, policy: RedactionPolicy) -> Iterator[Span]:
        """
        Yield one span per accepted named entity.

        Parameters
        ----------
        text : str
            The original text.
        policy : RedactionPolicy
            The active policy. Unused; present for protocol conformance.

        Yields
        ------
        Span
            Entities whose canonical label is selected, in ascending start
            order. Every span's ``text`` equals ``text[start:end]``.

        Raises
        ------
        CapabilityError
            If the tier, language or corpora are unavailable.
        DetectorError
            If NLTK raises while processing the text.
        """
        del policy
        nltk, sentence_tokenizer, word_tokenizer, chunker = self._ensure_ready()
        selected = self.labels
        found = 0

        try:
            sentences = list(sentence_tokenizer.span_tokenize(text))
        except Exception as exc:  # noqa: BLE001 - attributed, not suppressed
            raise DetectorError(
                f"NLTK sentence tokenization failed: {type(exc).__name__}: {exc}",
                detector=self.name,
            ) from exc

        for sentence_start, sentence_end in sentences:
            sentence = text[sentence_start:sentence_end]
            try:
                spans = list(word_tokenizer.span_tokenize(sentence))
                tokens = [sentence[a:b] for a, b in spans]
                if not tokens:
                    continue
                tagged = nltk.pos_tag(tokens)
                tree = (
                    chunker.parse(tagged)
                    if chunker is not None
                    else nltk.ne_chunk(tagged)
                )
            except Exception as exc:  # noqa: BLE001 - attributed, not suppressed
                raise DetectorError(
                    f"NLTK chunking failed: {type(exc).__name__}: {exc}",
                    detector=self.name,
                ) from exc

            index = 0
            for node in tree:
                if not hasattr(node, "label"):
                    index += 1
                    continue
                width = len(node.leaves())
                label = canonical_label(node.label())
                if selected is None or label in selected:
                    start, end = trim_entity_span(
                        text,
                        sentence_start + spans[index][0],
                        sentence_start + spans[index + width - 1][1],
                    )
                    if end > start:
                        found += 1
                        yield Span(
                            start=start,
                            end=end,
                            kind=self.kind_prefix + label,
                            text=text[start:end],
                            detector=self.name,
                            priority=self.priority,
                            confidence=self.confidence,
                        )
                index += width

        logger.debug(
            "nltk scanned %d characters in %d sentence(s), kept %d entity span(s)",
            len(text),
            len(sentences),
            found,
        )


def _build_chunker() -> Any:
    """
    Return a reusable named-entity chunker, or ``None`` if none can be built.

    Returns
    -------
    object or None
        Something with a ``parse(tagged_tokens)`` method, or ``None`` to say
        that the caller should use :func:`nltk.ne_chunk` per sentence.

    Notes
    -----
    **Developer notes.** The declared range is ``nltk>=3.6,<4``, and the
    constructor moved across it: ``nltk.chunk.ne_chunker`` is recent, while
    older releases exposed the chunker only through a pickle path that has also
    changed name. Rather than branch on a version number — which is a guess
    about what a release contains — this tries the documented constructor and
    falls back to the slow path when it is absent.

    Returning ``None`` rather than raising is deliberate. A chunker that cannot
    be pre-built is a *performance* problem, not a correctness one:
    :func:`nltk.ne_chunk` produces identical output. Refusing to run would turn
    a slow engine into no engine, which is the worse outcome.
    """
    try:
        from nltk.chunk import ne_chunker  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return ne_chunker()
    except Exception as exc:  # noqa: BLE001 - fall back, never fail the run
        logger.debug("nltk chunker could not be pre-built (%s); using ne_chunk", exc)
        return None


def _missing_corpora(nltk: Any) -> list[str]:
    """
    Return the NLTK data packages that cannot be found.

    Parameters
    ----------
    nltk : module
        The imported :mod:`nltk` module.

    Returns
    -------
    list of str
        Package names to download, in declaration order.

    Notes
    -----
    **Developer notes.** A package counts as present when *any* of its
    candidate resource paths resolves, because NLTK 3.8.2 split several
    packages and which name an installation carries depends on its version.
    Demanding both would report a working installation as broken.
    """
    missing: list[str] = []
    for package, paths in REQUIRED_CORPORA:
        for path in paths:
            try:
                nltk.data.find(path)
                break
            except LookupError:
                continue
            # treat any failure as absent
            except Exception:  # noqa: BLE001  # ruff: ignore[try-except-continue]
                continue
        else:
            missing.append(package)
    return missing


def corpora_status() -> dict[str, Any]:
    """
    Report whether NLTK's data packages are present.

    Returns
    -------
    dict
        ``{"available": bool, "missing": [...], "install_hint": str}``. When
        NLTK itself is absent, ``available`` is ``False`` and ``missing`` lists
        every package.

    Notes
    -----
    **Developer notes.** Used by ``doctor``. Importing NLTK here is acceptable
    because the caller has already decided to ask about NLTK specifically; it
    is never called during ``import scikitplot.cleanprompt``.
    """
    try:
        require("nltk")
        import nltk  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - absent tier is a reported state
        return {
            "available": False,
            "missing": [package for package, _ in REQUIRED_CORPORA],
            "install_hint": 'pip install "nltk>=3.6,<4"',
        }

    missing = _missing_corpora(nltk)
    return {
        "available": not missing,
        "missing": missing,
        "install_hint": (
            (
                'python -c "import nltk; '
                + "; ".join(f"nltk.download('{m}')" for m in missing)
                + '"'
            )
            if missing
            else ""
        ),
    }


def nltk_detector(
    labels: Iterable[str] | None = DEFAULT_ENTITY_LABELS,
    language: str = "en",
    **kwargs: Any,
) -> NltkDetector:
    """
    Build an :class:`NltkDetector`.

    Parameters
    ----------
    labels : iterable of str, optional
        Canonical labels to keep.
    language : str, default='en'
        Language code.
    **kwargs
        Forwarded to :class:`NltkDetector`.

    Returns
    -------
    NltkDetector
        The detector. Construction imports nothing; NLTK loads on first use.

    Examples
    --------
    >>> nltk_detector().kind
    'NE'
    """
    return NltkDetector(labels=labels, language=language, **kwargs)
