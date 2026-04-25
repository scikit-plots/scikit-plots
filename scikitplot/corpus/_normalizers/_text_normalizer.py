# scikitplot/corpus/_normalizers/_text_normalizer.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Text normalisation for clean embedding and retrieval input.

This module provides :class:`TextNormalizer`, a pipeline component
that populates ``CorpusDocument.normalized_text`` from ``doc.text``.
Downstream components (``EmbeddingEngine``, BM25 retrieval, MCP
servers) should prefer ``normalized_text`` when present.

Design Rationale:

Raw ``text`` from PDF, OCR, and subtitle readers carries artefacts
(hyphenation splits, ligatures, header/footer noise, double
whitespace, invisible Unicode control chars) that degrade both dense
vector quality and sparse keyword matching.  A ``normalized_text``
field lets the pipeline store **both** the original (provenance) and
the cleaned version (for embedding), without conflating them.

The normaliser is **optional** — it is inserted between
filter and embedding stages in the pipeline.  If skipped,
``EmbeddingEngine`` falls back to ``doc.text`` as today.

Supports Python 3.8 through 3.15.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "NormalizerConfig",
    "TextNormalizer",
    "normalize_text",
]


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class NormalizerConfig:
    r"""Configuration for :class:`TextNormalizer`.

    Parameters
    ----------
    unicode_form : str
        Unicode normalisation form: ``"NFKC"`` (default, canonical
        decomposition then compatibility composition) or ``"NFC"``,
        ``"NFD"``, ``"NFKD"``.  Set to ``""`` to disable.
    expand_ligatures : bool
        Replace common ligatures (ﬁ → fi, ﬂ → fl, ﬀ → ff, etc.).
    fix_hyphenation : bool
        Re-join words split across line breaks by a trailing hyphen
        (e.g., ``"compu-\\nter"`` → ``"computer"``).
    collapse_whitespace : bool
        Replace runs of whitespace (including ``\\t``, ``\\r``,
        ``\\n``) with a single space, then strip leading/trailing.
    strip_control_chars : bool
        Remove Unicode category Cc/Cf chars except ``\\n`` and
        ``\\t``.  Removes zero-width joiners, BOM, soft hyphens, etc.
    lowercase : bool
        Convert text to lowercase.  **Default False** — casing often
        matters for named-entity recognition in RAG contexts.
    min_length : int
        If the normalised text is shorter than this (in chars),
        set ``normalized_text = None`` so the embedding engine
        falls back to raw ``text``.
    custom_pipeline : tuple of Callable[[str], str]
        Additional user-supplied ``str → str`` transforms applied
        **after** all built-in steps.  Order is preserved.
    steps : list of str or None, optional
        Ordered list of step names to apply.  ``None`` (default)
        derives the list automatically from the boolean flags above.
        Pass an explicit list to run only a named subset, e.g.
        ``steps=["unicode", "whitespace"]``.  Valid names:
        ``"unicode"``, ``"ligatures"``, ``"control_chars"``,
        ``"hyphenation"``, ``"whitespace"``, ``"lowercase"``,
        ``"custom"``.  Used by :meth:`TextNormalizer.normalize`.

    Notes
    -----
    **User note:** The default configuration is designed for
    English-language PDF and OCR sources.  For CJK text, set
    ``fix_hyphenation=False`` (CJK does not hyphenate) and
    ``unicode_form="NFKC"`` (normalises full-width characters).

    **Developer note:** ``steps`` is excluded from ``__hash__`` and
    ``__eq__`` so that two configs with identical boolean flags but
    different ``steps`` lists are treated as equal for caching.  If you
    need strict equality on ``steps``, compare the lists explicitly.
    """

    unicode_form: str = "NFKC"
    expand_ligatures: bool = True
    fix_hyphenation: bool = True
    collapse_whitespace: bool = True
    strip_control_chars: bool = True
    lowercase: bool = False
    min_length: int = 1
    custom_pipeline: tuple[Callable[[str], str], ...] = field(default_factory=tuple)
    steps: list[str] | None = field(default=None, hash=False, compare=False)
    """Ordered list of normalisation step names to apply.

    When ``None`` (default), the list is derived automatically from the
    boolean flags above in pipeline order:
    ``"unicode"``, ``"ligatures"``, ``"control_chars"``, ``"hyphenation"``,
    ``"whitespace"``, ``"lowercase"``, ``"custom"``.
    Pass an explicit list to run only a named subset, e.g.
    ``steps=["unicode", "whitespace"]``.

    Notes
    -----
    Valid step names:

    ``"unicode"`` :
        Apply ``unicode_form`` normalisation.
    ``"ligatures"`` :
        Expand typographic ligatures (requires ``expand_ligatures=True``).
    ``"control_chars"`` :
        Strip Unicode control characters (requires ``strip_control_chars=True``).
    ``"hyphenation"`` :
        Re-join hyphenated line-breaks (requires ``fix_hyphenation=True``).
    ``"whitespace"`` :
        Collapse runs of whitespace (requires ``collapse_whitespace=True``).
    ``"lowercase"`` :
        Convert to lowercase (requires ``lowercase=True``).
    ``"custom"`` :
        Apply ``custom_pipeline`` callables.
    """

    def __post_init__(self) -> None:
        valid_forms = ("NFC", "NFD", "NFKC", "NFKD", "")
        if self.unicode_form not in valid_forms:
            raise ValueError(
                f"unicode_form must be one of {valid_forms}, got {self.unicode_form!r}"
            )
        if self.min_length < 0:
            raise ValueError(f"min_length must be >= 0, got {self.min_length}")
        # Derive the steps list from boolean flags when not explicitly supplied.
        if self.steps is None:
            derived: list[str] = []
            if self.unicode_form:
                derived.append("unicode")
            if self.expand_ligatures:
                derived.append("ligatures")
            if self.strip_control_chars:
                derived.append("control_chars")
            if self.fix_hyphenation:
                derived.append("hyphenation")
            if self.collapse_whitespace:
                derived.append("whitespace")
            if self.lowercase:
                derived.append("lowercase")
            if self.custom_pipeline:
                derived.append("custom")
            # frozen=True blocks normal assignment; object.__setattr__ is the
            # canonical workaround used throughout the Python dataclass ecosystem.
            object.__setattr__(self, "steps", derived)


# =====================================================================
# Compiled regex patterns (module-level for reuse)
# =====================================================================

# Hyphenation: word-char, hyphen, newline, word-char
_HYPHEN_RE = re.compile(r"(\w)-\s*\n\s*(\w)")

# Ligature map (the most common typographic ligatures from PDF engines)
_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",  # ſt ligature  # noqa: RUF003
    "\ufb06": "st",
    "\u0132": "IJ",
    "\u0133": "ij",
    "\u01c7": "LJ",
    "\u01c8": "Lj",
    "\u01c9": "lj",
    "\u01ca": "NJ",
    "\u01cb": "Nj",
    "\u01cc": "nj",
    "\u0152": "OE",
    "\u0153": "oe",
    "\u00c6": "AE",
    "\u00e6": "ae",
}

_LIGATURE_RE = re.compile("|".join(re.escape(k) for k in _LIGATURE_MAP))

# Control characters (Cc and Cf categories) minus \n and \t
_CONTROL_RE = re.compile(
    r"[^\S \n\t]"  # non-space whitespace except \n and \t
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # C0 controls minus \t \n \r
    r"|[\xad]"  # soft hyphen
    r"|[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\ufffe\uffff]"
)

# Multiple whitespace → single space
_MULTI_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


# =====================================================================
# Pure function: normalize a single string
# =====================================================================


def normalize_text(
    text: str,
    *,
    config: NormalizerConfig | None = None,
) -> str | None:
    r"""Normalise *text* according to *config*.

    Parameters
    ----------
    text : str
        Raw text to normalise.
    config : NormalizerConfig or None, optional
        Configuration.  ``None`` uses defaults.

    Returns
    -------
    str or None
        Normalised text, or ``None`` if the result is shorter than
        ``config.min_length``.

    Examples
    --------
    >>> normalize_text("The  ﬁrst  compu-\\nter  was huge.")
    'The first computer was huge.'
    """
    if config is None:
        config = NormalizerConfig()

    s = text

    # 1. Unicode normalisation
    if config.unicode_form:
        s = unicodedata.normalize(config.unicode_form, s)

    # 2. Ligature expansion (before hyphen fix — ligatures may span)
    if config.expand_ligatures:
        s = _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group()], s)

    # 3. Strip control characters (before whitespace collapse)
    if config.strip_control_chars:
        s = _CONTROL_RE.sub("", s)

    # 4. Fix hyphenation (before whitespace collapse)
    if config.fix_hyphenation:
        s = _HYPHEN_RE.sub(r"\1\2", s)

    # 5. Collapse whitespace
    if config.collapse_whitespace:
        s = _MULTI_NL_RE.sub("\n\n", s)
        s = _MULTI_WS_RE.sub(" ", s)
        s = "\n".join(line.strip() for line in s.split("\n"))
        s = s.strip()

    # 6. Lowercase
    if config.lowercase:
        s = s.lower()

    # 7. Custom pipeline
    for fn in config.custom_pipeline:
        s = fn(s)

    # 8. Min-length guard
    if len(s) < config.min_length:
        return None

    return s


# =====================================================================
# Pipeline component: operates on CorpusDocument sequences
# =====================================================================


class TextNormalizer:
    r"""Pipeline component that populates ``normalized_text`` on
    :class:`~scikitplot.corpus._schema.CorpusDocument` instances.

    Parameters
    ----------
    config : NormalizerConfig or None, optional
        Normalisation settings.  ``None`` uses defaults.

    Notes
    -----
    **User note:** Insert this component between the filter and
    embedding stages::

        source → reader → chunker → filter → **normalizer** → embedder

    If ``normalized_text`` is already set on a document (e.g., by a
    reader that does its own cleaning), this component skips it
    unless ``overwrite=True`` is passed to :meth:`normalize_documents`.

    **Developer note:** This class is stateless and thread-safe.
    All mutable state lives in the documents being processed.

    See Also
    --------
    scikitplot.corpus._schema.CorpusDocument : The normalised
        ``normalized_text`` field.
    scikitplot.corpus._enrichers._nlp_enricher.NLPEnricher :
        Downstream component that tokenises ``normalized_text``.

    Examples
    --------
    >>> from scikitplot.corpus._normalizers._text_normalizer import (
    ...     TextNormalizer,
    ... )
    >>> normalizer = TextNormalizer()
    >>> # doc = CorpusDocument(text="The  ﬁrst  compu-\\nter.", ...)
    >>> # docs = normalizer.normalize_documents([doc])
    >>> # docs[0].normalized_text == "The first computer."
    """  # noqa: D205

    def __init__(
        self,
        config: NormalizerConfig | None = None,
    ) -> None:
        self.config = config or NormalizerConfig()

    def normalize(self, text: str) -> str:  # noqa: PLR0912
        r"""Normalise a single string using only the steps in ``config.steps``.

        Unlike :func:`normalize_text`, this method:

        * Applies steps *selectively* — only those listed in
          ``self.config.steps`` are executed, in that order.
        * Returns ``""`` for empty input rather than ``None``.
        * Never returns ``None`` — callers that need the min-length guard
          should use :func:`normalize_text` directly.

        Parameters
        ----------
        text : str
            Raw text to normalise.

        Returns
        -------
        str
            Normalised text, or ``""`` if *text* is empty or becomes empty
            after normalisation.

        Examples
        --------
        >>> n = TextNormalizer(NormalizerConfig(steps=["unicode"]))
        >>> "\\ufb01" not in n.normalize("fi\\ufb01rst")
        True
        >>> n2 = TextNormalizer(NormalizerConfig(steps=["whitespace"]))
        >>> "   " not in n2.normalize("Hello   world")
        True
        >>> TextNormalizer(NormalizerConfig()).normalize("")
        ''
        """
        if not text:
            return ""
        s = text
        for step in self.config.steps or []:
            if step == "unicode":
                if self.config.unicode_form:
                    s = unicodedata.normalize(self.config.unicode_form, s)
            elif step == "ligatures":
                if self.config.expand_ligatures:
                    s = _LIGATURE_RE.sub(lambda m: _LIGATURE_MAP[m.group()], s)
            elif step == "control_chars":
                if self.config.strip_control_chars:
                    s = _CONTROL_RE.sub("", s)
            elif step == "hyphenation":
                if self.config.fix_hyphenation:
                    s = _HYPHEN_RE.sub(r"\1\2", s)
            elif step == "whitespace":
                if self.config.collapse_whitespace:
                    s = _MULTI_NL_RE.sub("\n\n", s)
                    s = _MULTI_WS_RE.sub(" ", s)
                    s = "\n".join(line.strip() for line in s.split("\n"))
                    s = s.strip()
            elif step == "lowercase":
                if self.config.lowercase:
                    s = s.lower()
            elif step == "custom":
                for fn in self.config.custom_pipeline:
                    s = fn(s)
        return s

    def normalize_documents(
        self,
        documents: Sequence[Any],
        *,
        overwrite: bool = False,
    ) -> list[Any]:
        """Normalise text for a batch of ``CorpusDocument`` instances.

        Parameters
        ----------
        documents : Sequence[CorpusDocument]
            Documents to normalise.  Not mutated — new instances are
            returned via ``doc.replace()``.
        overwrite : bool, optional
            If ``True``, re-normalise even if ``normalized_text`` is
            already set.  Default ``False``.

        Returns
        -------
        list[CorpusDocument]
            New document instances with ``normalized_text`` populated.
        """
        out: list[Any] = []
        n_normalised = 0
        n_skipped = 0

        for doc in documents:
            if not overwrite and getattr(doc, "normalized_text", None):
                out.append(doc)
                n_skipped += 1
                continue

            raw = getattr(doc, "text", "") or ""
            cleaned = normalize_text(raw, config=self.config)
            out.append(doc.replace(normalized_text=cleaned))
            n_normalised += 1

        logger.info(
            "TextNormalizer: normalised=%d, skipped=%d",
            n_normalised,
            n_skipped,
        )
        return out

    def __repr__(self) -> str:
        return f"TextNormalizer(config={self.config!r})"
