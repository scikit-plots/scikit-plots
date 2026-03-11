"""
scikitplot.corpus._normalizers._normalizer
==========================================
Concrete normaliser implementations.

Design invariants
-----------------
* Every normaliser is **side-effect free**: it returns a new
  :class:`~scikitplot.corpus._schema.CorpusDocument` via ``replace()``
  and never mutates the input.
* Normalised text is always written to ``normalized_text``, never to
  ``text``. The original ``text`` field is immutable once created.
* Normalisers must handle empty ``normalized_text`` (use ``text`` as the
  base when ``normalized_text`` is ``None``).
* All optional dependencies are guarded by ``ImportError`` at call time
  with an actionable install instruction.

Available normalisers
---------------------
:class:`UnicodeNormalizer`
    NFC/NFD/NFKC/NFKD Unicode normalisation. Zero dependencies.

:class:`WhitespaceNormalizer`
    Collapse runs of whitespace, strip leading/trailing whitespace.

:class:`HTMLStripNormalizer`
    Remove HTML tags. Zero dependencies (regex) or richer via ``bs4``.

:class:`LowercaseNormalizer`
    Lowercase the text.

:class:`DedupLinesNormalizer`
    Remove exact duplicate lines while preserving order.

:class:`LanguageDetectionNormalizer`
    Detect language and set ``CorpusDocument.language``.
    Requires ``langdetect`` (``pip install langdetect``).

:class:`NormalizationPipeline`
    Chain multiple normalisers; applies them in order.

Python compatibility
--------------------
Python 3.8-3.15. No ``match``, no ``StrEnum``, ``from __future__ import
annotations`` throughout.
"""  # noqa: D205, D400

from __future__ import annotations

import abc
import logging
import re
import unicodedata
from typing import Any, List, Optional, Sequence  # noqa: F401

from scikitplot.corpus._schema import CorpusDocument

logger = logging.getLogger(__name__)

# Matches one or more whitespace characters (including non-breaking space).
_WS_RE: re.Pattern[str] = re.compile(r"\s+")

# Matches HTML/XML tags.
_HTML_TAG_RE: re.Pattern[str] = re.compile(r"<[^>]+>", re.DOTALL)

# Common HTML entities (subset — full decode via html module).
_HTML_ENTITY_RE: re.Pattern[str] = re.compile(r"&(?:#\d+|#x[\da-fA-F]+|\w+);")


# ===========================================================================
# Abstract base
# ===========================================================================


class NormalizerBase(abc.ABC):
    """
    Abstract base class for all text normalisers.

    A normaliser receives a :class:`~scikitplot.corpus._schema.CorpusDocument`
    and returns a new instance with ``normalized_text`` updated. If the
    normaliser has nothing to do (empty text, already clean, etc.) it
    returns the document unchanged.

    Normalisers are composable via :class:`NormalizationPipeline`.

    Notes
    -----
    Subclasses must implement :meth:`normalize_doc`. They must never
    call :meth:`CorpusDocument.replace` with ``text=`` — only
    ``normalized_text=`` may be modified.
    """

    @abc.abstractmethod
    def normalize_doc(self, doc: CorpusDocument) -> CorpusDocument:
        """
        Apply normalisation to ``doc``.

        Parameters
        ----------
        doc : CorpusDocument
            Input document. Must be a valid, validated instance.

        Returns
        -------
        CorpusDocument
            New instance with ``normalized_text`` updated. ``text`` is
            never modified.
        """

    def _get_source_text(self, doc: CorpusDocument) -> str:  # noqa: D417
        """
        Return the text to normalise.

        Uses ``normalized_text`` if already set; falls back to ``text``.

        Parameters
        ----------
        doc : CorpusDocument

        Returns
        -------
        str
        """
        # if doc.normalized_text is None:
        #     raise RuntimeError(
        #         "NormalizationPipeline invariant violated: "
        #         "normalized_text must be set before this stage."
        #     )
        return doc.normalized_text if doc.normalized_text is not None else doc.text

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}()"


# ===========================================================================
# Unicode normaliser
# ===========================================================================


class UnicodeNormalizer(NormalizerBase):
    r"""
    Apply Unicode normalisation (NFC, NFD, NFKC, or NFKD).

    Parameters
    ----------
    form : {\"NFC\", \"NFD\", \"NFKC\", \"NFKD\"}, optional
        Unicode normalisation form. ``NFKC`` is recommended for NLP
        (decomposes ligatures, expands compatibility characters).
        Default: ``\"NFC\"``.

    Examples
    --------
    >>> from scikitplot.corpus._normalizers import UnicodeNormalizer
    >>> norm = UnicodeNormalizer(form="NFKC")
    >>> doc = CorpusDocument.create("f.txt", 0, "ﬁle")  # fi ligature
    >>> norm.normalize_doc(doc).normalized_text
    'file'
    """

    _VALID_FORMS: frozenset[str] = frozenset({"NFC", "NFD", "NFKC", "NFKD"})

    def __init__(self, form: str = "NFC") -> None:
        if form not in self._VALID_FORMS:
            raise ValueError(
                f"UnicodeNormalizer: form must be one of {sorted(self._VALID_FORMS)}; "
                f"got {form!r}."
            )
        self.form: str = form

    def normalize_doc(self, doc: CorpusDocument) -> CorpusDocument:  # noqa: D417
        """
        Apply Unicode normalisation to the document text.

        Parameters
        ----------
        doc : CorpusDocument

        Returns
        -------
        CorpusDocument
        """
        source = self._get_source_text(doc)
        normalised = unicodedata.normalize(self.form, source)
        return doc.replace(normalized_text=normalised)

    def __repr__(self) -> str:  # noqa: D105
        return f"UnicodeNormalizer(form={self.form!r})"


# ===========================================================================
# Whitespace normaliser
# ===========================================================================


class WhitespaceNormalizer(NormalizerBase):
    """
    Collapse runs of whitespace and optionally strip leading/trailing space.

    Parameters
    ----------
    collapse_newlines : bool, optional
        When ``True``, newline characters are treated as whitespace and
        collapsed with other spaces. When ``False``, newlines are
        preserved (only intra-line spaces are collapsed). Default: ``False``.
    strip : bool, optional
        Strip leading and trailing whitespace from the result.
        Default: ``True``.

    Examples
    --------
    >>> norm = WhitespaceNormalizer()
    >>> doc = CorpusDocument.create("f.txt", 0, "Hello   world.")
    >>> norm.normalize_doc(doc).normalized_text
    'Hello world.'
    """

    def __init__(
        self,
        collapse_newlines: bool = False,
        strip: bool = True,
    ) -> None:
        self.collapse_newlines: bool = collapse_newlines
        self.strip: bool = strip

    def normalize_doc(self, doc: CorpusDocument) -> CorpusDocument:  # noqa: D417
        """
        Collapse whitespace in the document text.

        Parameters
        ----------
        doc : CorpusDocument

        Returns
        -------
        CorpusDocument
        """
        source = self._get_source_text(doc)

        if self.collapse_newlines:
            normalised = _WS_RE.sub(" ", source)
        else:
            # Process line by line, collapsing only within each line.
            lines = source.splitlines()
            normalised = "\n".join(re.sub(r"[ \t]+", " ", line) for line in lines)

        if self.strip:
            normalised = normalised.strip()
        return doc.replace(normalized_text=normalised)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"WhitespaceNormalizer("
            f"collapse_newlines={self.collapse_newlines}, "
            f"strip={self.strip})"
        )


# ===========================================================================
# HTML strip normaliser
# ===========================================================================


class HTMLStripNormalizer(NormalizerBase):
    r"""
    Remove HTML and XML tags from the document text.

    Two modes are available:
    * ``use_beautifulsoup=False`` (default): regex-based stripping.
      Zero additional dependencies; handles well-formed HTML.
    * ``use_beautifulsoup=True``: uses ``bs4.BeautifulSoup`` for robust
      parsing of malformed or deeply nested HTML.
      Requires ``pip install beautifulsoup4``.

    Parameters
    ----------
    use_beautifulsoup : bool, optional
        Use BeautifulSoup for parsing. Default: ``False``.
    parser : str, optional
        BeautifulSoup parser (``\"html.parser\"``, ``\"lxml\"``,
        ``\"html5lib\"``). Ignored when ``use_beautifulsoup=False``.
        Default: ``\"html.parser\"`` (stdlib, no extra deps).
    decode_entities : bool, optional
        Decode HTML entities (e.g. ``&amp;`` → ``&``). Default: ``True``.

    Examples
    --------
    >>> norm = HTMLStripNormalizer()
    >>> doc = CorpusDocument.create("f.txt", 0, "<p>Hello <b>world</b>.</p>")
    >>> norm.normalize_doc(doc).normalized_text
    'Hello world.'
    """

    def __init__(
        self,
        use_beautifulsoup: bool = False,
        parser: str = "html.parser",
        decode_entities: bool = True,
    ) -> None:
        self.use_beautifulsoup: bool = use_beautifulsoup
        self.parser: str = parser
        self.decode_entities: bool = decode_entities

    def normalize_doc(self, doc: CorpusDocument) -> CorpusDocument:  # noqa: D417
        """
        Strip HTML tags from the document text.

        Parameters
        ----------
        doc : CorpusDocument

        Returns
        -------
        CorpusDocument

        Raises
        ------
        ImportError
            If ``use_beautifulsoup=True`` and ``beautifulsoup4`` is not installed.
        """
        source = self._get_source_text(doc)

        if self.use_beautifulsoup:
            normalised = self._strip_with_bs4(source)
        else:
            normalised = self._strip_with_regex(source)
        return doc.replace(normalized_text=normalised)

    def _strip_with_regex(self, text: str) -> str:
        """Remove tags with regex and optionally decode entities."""
        result = _HTML_TAG_RE.sub("", text)
        if self.decode_entities:
            import html  # noqa: PLC0415

            result = html.unescape(result)
        return result

    def _strip_with_bs4(self, text: str) -> str:
        """Remove tags using BeautifulSoup."""
        try:
            from bs4 import (  # type: ignore[import-untyped]  # noqa: PLC0415
                BeautifulSoup,
            )
        except ImportError as exc:
            raise ImportError(
                "HTMLStripNormalizer(use_beautifulsoup=True) requires "
                "'beautifulsoup4'. Install with: pip install beautifulsoup4"
            ) from exc
        soup = BeautifulSoup(text, self.parser)
        return soup.get_text(separator=" ")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"HTMLStripNormalizer("
            f"use_beautifulsoup={self.use_beautifulsoup}, "
            f"parser={self.parser!r})"
        )


# ===========================================================================
# Lowercase normaliser
# ===========================================================================


class LowercaseNormalizer(NormalizerBase):
    """
    Convert the document text to lowercase.

    Parameters
    ----------
    locale_aware : bool, optional
        When ``True``, use Python's ``str.casefold()`` (more aggressive,
        handles German ß → ss etc.) instead of ``str.lower()``.
        Default: ``False``.

    Examples
    --------
    >>> norm = LowercaseNormalizer()
    >>> doc = CorpusDocument.create("f.txt", 0, "Hello World.")
    >>> norm.normalize_doc(doc).normalized_text
    'hello world.'
    """

    def __init__(self, locale_aware: bool = False) -> None:
        self.locale_aware: bool = locale_aware

    def normalize_doc(self, doc: CorpusDocument) -> CorpusDocument:  # noqa: D417
        """
        Lowercase the document text.

        Parameters
        ----------
        doc : CorpusDocument

        Returns
        -------
        CorpusDocument
        """
        source = self._get_source_text(doc)
        normalised = source.casefold() if self.locale_aware else source.lower()
        return doc.replace(normalized_text=normalised)

    def __repr__(self) -> str:  # noqa: D105
        return f"LowercaseNormalizer(locale_aware={self.locale_aware})"


# ===========================================================================
# Dedup lines normaliser
# ===========================================================================


class DedupLinesNormalizer(NormalizerBase):
    r"""
    Remove exact duplicate lines while preserving first-occurrence order.

    Useful for de-noising OCR output and web-scraped text which often
    contains repeated navigation bars, headers, or footers.

    Parameters
    ----------
    ignore_whitespace : bool, optional
        When ``True``, lines are compared after stripping; the original
        (un-stripped) line is preserved in the output. Default: ``True``.
    min_line_length : int, optional
        Lines shorter than this (after stripping) are always kept even
        if they are duplicates. Prevents discarding single-character
        structural lines. Default: ``0``.

    Examples
    --------
    >>> norm = DedupLinesNormalizer()
    >>> doc = CorpusDocument.create("f.txt", 0, "Hello.\\nHello.\\nWorld.")
    >>> norm.normalize_doc(doc).normalized_text
    'Hello.\\nWorld.'
    """

    def __init__(
        self,
        ignore_whitespace: bool = True,
        min_line_length: int = 0,
    ) -> None:
        if min_line_length < 0:
            raise ValueError(
                f"DedupLinesNormalizer: min_line_length must be >= 0; "
                f"got {min_line_length!r}."
            )
        self.ignore_whitespace: bool = ignore_whitespace
        self.min_line_length: int = min_line_length

    def normalize_doc(self, doc: CorpusDocument) -> CorpusDocument:  # noqa: D417
        """
        Remove duplicate lines from the document text.

        Parameters
        ----------
        doc : CorpusDocument

        Returns
        -------
        CorpusDocument
        """
        source = self._get_source_text(doc)
        lines = source.splitlines()
        seen: set[str] = set()
        output: list[str] = []

        for line in lines:
            key = line.strip() if self.ignore_whitespace else line
            should_dedup = len(key) >= self.min_line_length
            # Always keep very short lines.
            if should_dedup:
                if key in seen:
                    logger.debug(
                        "DedupLinesNormalizer: removed duplicate line: %r.",
                        key[:60],
                    )
                    continue
                seen.add(key)
            output.append(line)
        normalized = "\n".join(output)
        return doc.replace(normalized_text=normalized)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DedupLinesNormalizer("
            f"ignore_whitespace={self.ignore_whitespace}, "
            f"min_line_length={self.min_line_length})"
        )


# ===========================================================================
# Language detection normaliser
# ===========================================================================


class LanguageDetectionNormalizer(NormalizerBase):
    """
    Detect document language and set ``CorpusDocument.language``.

    Uses ``langdetect`` (``pip install langdetect``) which is a port of
    Google's language-detection library. Falls back to the provided
    ``fallback_language`` if detection fails or the detected language
    has confidence below ``min_confidence``.

    Parameters
    ----------
    fallback_language : str or None, optional
        ISO 639-1 language code to use when detection fails.
        ``None`` leaves ``language`` unchanged on failure.
        Default: ``None``.
    min_confidence : float, optional
        Minimum probability threshold for accepting a detected language.
        Must be in ``[0.0, 1.0]``. Default: ``0.7``.
    overwrite : bool, optional
        When ``False``, skip detection if the document already has a
        non-``None`` ``language`` field. Default: ``False``.

    Examples
    --------
    >>> norm = LanguageDetectionNormalizer(fallback_language="en")
    >>> doc = CorpusDocument.create("f.txt", 0, "The quick brown fox.")
    >>> result = norm.normalize_doc(doc)
    >>> result.language
    'en'
    """

    def __init__(
        self,
        fallback_language: str | None = None,
        min_confidence: float = 0.7,
        overwrite: bool = False,
    ) -> None:
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(
                f"LanguageDetectionNormalizer: min_confidence must be in "
                f"[0.0, 1.0]; got {min_confidence!r}."
            )
        self.fallback_language: str | None = fallback_language
        self.min_confidence: float = min_confidence
        self.overwrite: bool = overwrite

    def normalize_doc(self, doc: CorpusDocument) -> CorpusDocument:  # noqa: D417
        """
        Detect language and update ``doc.language``.

        Parameters
        ----------
        doc : CorpusDocument

        Returns
        -------
        CorpusDocument
            New instance with ``language`` set (or unchanged on failure).

        Raises
        ------
        ImportError
            If ``langdetect`` is not installed.
        """
        if not self.overwrite and doc.language is not None:
            return doc

        try:
            from langdetect import (  # type: ignore[import-untyped]  # noqa: PLC0415
                detect_langs,
            )
        except ImportError as exc:
            raise ImportError(
                "LanguageDetectionNormalizer requires 'langdetect'. "
                "Install with: pip install langdetect"
            ) from exc

        source = self._get_source_text(doc)
        if (
            len(source.split()) < 5  # noqa: PLR2004
            and self.fallback_language is not None
        ):
            # Too short for reliable detection.
            detected_lang = self.fallback_language
        else:
            try:
                candidates = detect_langs(source)
                best = max(candidates, key=lambda c: c.prob)
                if best.prob >= self.min_confidence:
                    detected_lang = best.lang
                    logger.debug(
                        "LanguageDetectionNormalizer: detected %r "
                        "(prob=%.2f) for doc %r.",
                        detected_lang,
                        best.prob,
                        doc.doc_id,
                    )
                else:
                    logger.debug(
                        "LanguageDetectionNormalizer: low confidence "
                        "%.2f for %r; using fallback %r.",
                        best.prob,
                        doc.doc_id,
                        self.fallback_language,
                    )
                    detected_lang = self.fallback_language
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "LanguageDetectionNormalizer: detection failed for "
                    "doc %r: %s. Using fallback %r.",
                    doc.doc_id,
                    exc,
                    self.fallback_language,
                )
                detected_lang = self.fallback_language

        if detected_lang == doc.language:
            return doc
        return doc.replace(language=detected_lang)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"LanguageDetectionNormalizer("
            f"fallback={self.fallback_language!r}, "
            f"min_confidence={self.min_confidence})"
        )


# ===========================================================================
# NormalizationPipeline — ordered chain of normalisers
# ===========================================================================


class NormalizationPipeline(NormalizerBase):
    """
    Apply a sequence of normalisers in order.

    Each normaliser in the pipeline receives the output of the previous
    one. Normalisers that have no effect return the document unchanged,
    so only modified documents incur a ``replace()`` call.

    Parameters
    ----------
    normalizers : sequence of NormalizerBase
        Ordered list of normalisers to apply.

    Raises
    ------
    ValueError
        If ``normalizers`` is empty.

    Examples
    --------
    >>> pipeline = NormalizationPipeline(
    ...     [
    ...         UnicodeNormalizer(form="NFKC"),
    ...         HTMLStripNormalizer(),
    ...         WhitespaceNormalizer(),
    ...     ]
    ... )
    >>> result = pipeline.normalize_doc(doc)
    """

    def __init__(self, normalizers: Sequence[NormalizerBase]) -> None:
        if not normalizers:
            raise ValueError("NormalizationPipeline: normalizers must not be empty.")
        bad = [n for n in normalizers if not isinstance(n, NormalizerBase)]
        if bad:
            raise TypeError(
                f"NormalizationPipeline: all items must be NormalizerBase "
                f"instances; got {[type(b).__name__ for b in bad]!r}."
            )
        self.normalizers: list[NormalizerBase] = list(normalizers)

    def normalize_doc(self, doc: CorpusDocument) -> CorpusDocument:  # noqa: D417
        """
        Apply all normalisers in order.

        Parameters
        ----------
        doc : CorpusDocument

        Returns
        -------
        CorpusDocument
            Document after all normalisation stages.
        """
        current = doc
        for norm in self.normalizers:
            current = norm.normalize_doc(current)
        return current

    def normalize_batch(  # noqa: D417
        self, docs: list[CorpusDocument]
    ) -> list[CorpusDocument]:
        """
        Apply the pipeline to a list of documents.

        Parameters
        ----------
        docs : list[CorpusDocument]

        Returns
        -------
        list[CorpusDocument]
        """
        return [self.normalize_doc(doc) for doc in docs]

    def __repr__(self) -> str:  # noqa: D105
        names = ", ".join(repr(n) for n in self.normalizers)
        return f"NormalizationPipeline([{names}])"


__all__ = [
    "DedupLinesNormalizer",
    "HTMLStripNormalizer",
    "LanguageDetectionNormalizer",
    "LowercaseNormalizer",
    "NormalizationPipeline",
    "NormalizerBase",
    "UnicodeNormalizer",
    "WhitespaceNormalizer",
]
