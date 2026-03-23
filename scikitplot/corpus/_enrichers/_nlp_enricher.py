# scikitplot/corpus/_enrichers/_nlp_enricher.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
NLP enrichment component for ``CorpusDocument``.

Populates the NLP enrichment fields on
:class:`~scikitplot.corpus._schema.CorpusDocument`:

- ``tokens``   — whitespace/NLP-tokenised word list
- ``lemmas``   — lemmatised tokens (spaCy or NLTK WordNet)
- ``stems``    — stemmed tokens (Porter, Snowball, Lancaster)
- ``keywords`` — extracted keywords (TF-IDF, YAKE, KeyBERT, or
  simple frequency)

These fields enable KEYWORD match mode, BM25 sparse retrieval,
and word-corpus research (lemmatisation / stemming studies).

.. admonition:: Backends are lazy-loaded

   No NLP library is imported at module level.  Each backend
   (spaCy, NLTK, YAKE, KeyBERT) is imported on first use and
   cached per instance.  The enricher works with zero optional
   deps installed (falls back to regex tokenisation and no
   lemmatisation / stemming / keywords).

Supports Python 3.8 through 3.15.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field  # noqa: F401
from typing import Any, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "BUILTIN_STOPWORDS",
    "EnricherConfig",
    "NLPEnricher",
]


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class EnricherConfig:
    r"""
    Configuration for :class:`NLPEnricher`.

    Parameters
    ----------
    tokenizer : str
        Tokenisation backend: ``"simple"`` (regex ``\\w+``),
        ``"nltk"`` (``nltk.tokenize.word_tokenize``), or
        ``"spacy"`` (spaCy tokenizer).
    spacy_model : str
        spaCy model name, used when ``tokenizer="spacy"`` or
        ``lemmatizer="spacy"``.
    lemmatizer : str or None
        Lemmatisation backend: ``"spacy"``, ``"nltk"``
        (``WordNetLemmatizer``), or ``None`` (skip).
    stemmer : str or None
        Stemming backend: ``"porter"``, ``"snowball"``,
        ``"lancaster"``, or ``None`` (skip).
    stemmer_language : str
        Language for Snowball stemmer.
    keyword_extractor : str or None
        Keyword extraction backend: ``"frequency"`` (top-N by
        term frequency), ``"yake"``, ``"keybert"``, or ``None``
        (skip).
    max_keywords : int
        Maximum keywords to extract per document.
    lowercase_tokens : bool
        Lowercase all tokens before further processing.
    remove_stopwords : bool
        Remove stopwords.  Uses NLTK's English stopword list when
        available, otherwise a small built-in set.
    min_token_length : int
        Discard tokens shorter than this (after lowercasing).
    remove_punctuation : bool
        Remove tokens that are entirely punctuation.

    Notes
    -----
    **User note:** For RAG pipelines, ``tokenizer="simple"`` with
    ``keyword_extractor="frequency"`` is usually sufficient.
    For linguistic research, use ``"spacy"`` with
    ``lemmatizer="spacy"`` for best accuracy.
    """

    tokenizer: str = "simple"
    spacy_model: str = "en_core_web_sm"
    lemmatizer: str | None = None
    stemmer: str | None = None
    stemmer_language: str = "english"
    keyword_extractor: str | None = "frequency"
    max_keywords: int = 20
    lowercase_tokens: bool = True
    remove_stopwords: bool = True
    min_token_length: int = 2
    remove_punctuation: bool = True

    def __post_init__(self) -> None:
        valid_tokenizers = ("simple", "nltk", "spacy")
        if self.tokenizer not in valid_tokenizers:
            raise ValueError(
                f"tokenizer must be one of {valid_tokenizers}, got {self.tokenizer!r}"
            )
        valid_lemmers = (None, "spacy", "nltk")
        if self.lemmatizer not in valid_lemmers:
            raise ValueError(
                f"lemmatizer must be one of {valid_lemmers}, got {self.lemmatizer!r}"
            )
        valid_stemmers = (None, "porter", "snowball", "lancaster")
        if self.stemmer not in valid_stemmers:
            raise ValueError(
                f"stemmer must be one of {valid_stemmers}, got {self.stemmer!r}"
            )
        valid_kw = (None, "frequency", "yake", "keybert")
        if self.keyword_extractor not in valid_kw:
            raise ValueError(
                f"keyword_extractor must be one of {valid_kw}, "
                f"got {self.keyword_extractor!r}"
            )
        if self.max_keywords < 1:
            raise ValueError(f"max_keywords must be >= 1, got {self.max_keywords}")
        if self.min_token_length < 0:
            raise ValueError(
                f"min_token_length must be >= 0, got {self.min_token_length}"
            )


# =====================================================================
# Built-in stopwords (small English set for zero-dep fallback)
# =====================================================================

BUILTIN_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "not",
        "no",
        "nor",
        "so",
        "as",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "we",
        "you",
        "i",
        "me",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "also",
        "am",
        "any",
        "because",
        "before",
        "below",
        "between",
        "during",
        "here",
        "into",
        "out",
        "over",
        "then",
        "there",
        "through",
        "too",
        "under",
        "until",
        "up",
    }
)

# Regex tokeniser: word chars (Unicode aware)
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Punctuation-only token check
_PUNCT_RE = re.compile(r"^[^\w]+$", re.UNICODE)


# =====================================================================
# NLPEnricher
# =====================================================================


class NLPEnricher:
    """
    Pipeline component that populates NLP enrichment fields on
    :class:`~scikitplot.corpus._schema.CorpusDocument`.

    Parameters
    ----------
    config : EnricherConfig or None, optional
        Enrichment settings.  ``None`` uses defaults.

    Notes
    -----
    **User note:** Insert after ``TextNormalizer`` and before
    ``EmbeddingEngine`` in the pipeline::

        source → reader → chunker → filter → normalizer
          → **enricher** → embedder

    The enricher reads ``doc.normalized_text`` when available,
    falling back to ``doc.text``.

    **Developer note:** All NLP backends are lazy-loaded and
    cached on ``self._*`` attributes.  The class is NOT thread-safe
    (shared mutable cache).  Use separate instances per thread.

    See Also
    --------
    scikitplot.corpus._normalizers._text_normalizer.TextNormalizer :
        Upstream component that prepares ``normalized_text``.
    scikitplot.corpus._schema.CorpusDocument : The ``tokens``,
        ``lemmas``, ``stems``, ``keywords`` fields.

    Examples
    --------
    >>> enricher = NLPEnricher()
    >>> # doc = CorpusDocument(text="The quick brown fox.", ...)
    >>> # docs = enricher.enrich_documents([doc])
    >>> # docs[0].tokens == ["quick", "brown", "fox"]
    """  # noqa: D205

    def __init__(
        self,
        config: EnricherConfig | None = None,
    ) -> None:
        self.config = config or EnricherConfig()
        # Lazy-loaded backends
        self._spacy_nlp: Any = None
        self._nltk_lemmatizer: Any = None
        self._stemmer_obj: Any = None
        self._stopwords: frozenset[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich_documents(
        self,
        documents: Sequence[Any],
        *,
        overwrite: bool = False,
    ) -> list[Any]:
        """Enrich a batch of ``CorpusDocument`` instances.

        Parameters
        ----------
        documents : Sequence[CorpusDocument]
            Documents to enrich.  Not mutated.
        overwrite : bool, optional
            Re-enrich even if NLP fields are already populated.

        Returns
        -------
        list[CorpusDocument]
            New instances with NLP fields populated.
        """
        out: list[Any] = []
        n_enriched = 0

        for doc in documents:
            has_tokens = getattr(doc, "tokens", None) is not None
            if not overwrite and has_tokens:
                out.append(doc)
                continue

            text = getattr(doc, "normalized_text", None) or getattr(doc, "text", "")

            tokens = self._tokenize(text)
            tokens = self._filter_tokens(tokens)

            lemmas = self._lemmatize(tokens) if self.config.lemmatizer else None
            stems = self._stem(tokens) if self.config.stemmer else None
            keywords = self._extract_keywords(text, tokens)

            out.append(
                doc.replace(
                    tokens=tokens or None,
                    lemmas=lemmas,
                    stems=stems,
                    keywords=keywords,
                )
            )
            n_enriched += 1

        logger.info(
            "NLPEnricher: enriched=%d, total=%d",
            n_enriched,
            len(documents),
        )
        return out

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Tokenise *text* using the configured backend."""
        cfg = self.config

        if cfg.tokenizer == "spacy":
            nlp = self._get_spacy()
            doc = nlp(text)
            return [tok.text for tok in doc if not tok.is_space]

        if cfg.tokenizer == "nltk":
            try:
                from nltk.tokenize import (  # type: ignore[import]  # noqa: PLC0415
                    word_tokenize,
                )

                return word_tokenize(text)
            except ImportError:
                logger.warning("NLTK not installed; falling back to simple tokenizer.")

        # "simple" or fallback
        return _WORD_RE.findall(text)

    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        """
        Apply lowercase, stopword removal, punctuation removal,
        and min-length filtering.
        """  # noqa: D205
        cfg = self.config
        result: list[str] = []
        stopwords = self._get_stopwords()

        for tok in tokens:
            if cfg.lowercase_tokens:
                tok = tok.lower()  # noqa: PLW2901

            if cfg.remove_punctuation and _PUNCT_RE.match(tok):
                continue

            if cfg.remove_stopwords and tok.lower() in stopwords:
                continue

            if len(tok) < cfg.min_token_length:
                continue

            result.append(tok)

        return result

    # ------------------------------------------------------------------
    # Lemmatisation
    # ------------------------------------------------------------------

    def _lemmatize(self, tokens: list[str]) -> list[str] | None:
        """Lemmatise *tokens* using the configured backend."""
        cfg = self.config
        if cfg.lemmatizer is None:
            return None

        if cfg.lemmatizer == "spacy":
            nlp = self._get_spacy()
            # Process tokens as a single space-joined string
            doc = nlp(" ".join(tokens))
            return [tok.lemma_ for tok in doc if not tok.is_space]

        if cfg.lemmatizer == "nltk":
            lem = self._get_nltk_lemmatizer()
            return [lem.lemmatize(tok) for tok in tokens]

        return None

    # ------------------------------------------------------------------
    # Stemming
    # ------------------------------------------------------------------

    def _stem(self, tokens: list[str]) -> list[str] | None:
        """Stem *tokens* using the configured backend."""
        cfg = self.config
        if cfg.stemmer is None:
            return None

        stemmer = self._get_stemmer()
        return [stemmer.stem(tok) for tok in tokens]

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    def _extract_keywords(
        self,
        text: str,
        tokens: list[str],
    ) -> list[str] | None:
        """Extract keywords from *text* or *tokens*."""
        cfg = self.config
        if cfg.keyword_extractor is None:
            return None

        if cfg.keyword_extractor == "yake":
            return self._keywords_yake(text)

        if cfg.keyword_extractor == "keybert":
            return self._keywords_keybert(text)

        # "frequency" — simple term-frequency top-N
        return self._keywords_frequency(tokens)

    def _keywords_frequency(self, tokens: list[str]) -> list[str]:
        """Top-N keywords by raw term frequency."""
        freq: dict[str, int] = {}
        for tok in tokens:
            key = tok.lower()
            freq[key] = freq.get(key, 0) + 1

        sorted_terms = sorted(freq, key=freq.__getitem__, reverse=True)
        return sorted_terms[: self.config.max_keywords]

    def _keywords_yake(self, text: str) -> list[str] | None:
        """Extract keywords using YAKE."""
        try:
            import yake  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            logger.warning("YAKE not installed; falling back to frequency keywords.")
            return None

        extractor = yake.KeywordExtractor(
            top=self.config.max_keywords,
            dedupLim=0.9,
        )
        kws = extractor.extract_keywords(text)
        return [kw for kw, _score in kws]

    def _keywords_keybert(self, text: str) -> list[str] | None:
        """Extract keywords using KeyBERT."""
        try:
            from keybert import KeyBERT  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            logger.warning("KeyBERT not installed; falling back to frequency keywords.")
            return None

        model = KeyBERT()
        kws = model.extract_keywords(
            text,
            top_n=self.config.max_keywords,
        )
        return [kw for kw, _score in kws]

    # ------------------------------------------------------------------
    # Lazy backend loaders
    # ------------------------------------------------------------------

    def _get_spacy(self) -> Any:
        """Load and cache spaCy NLP pipeline."""
        if self._spacy_nlp is not None:
            return self._spacy_nlp

        try:
            import spacy  # type: ignore[import]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for tokenizer='spacy' or "
                "lemmatizer='spacy'.  Install: pip install spacy"
            ) from exc

        model = self.config.spacy_model
        try:
            self._spacy_nlp = spacy.load(model)
        except OSError:
            logger.info("Downloading spaCy model %r...", model)
            from spacy.cli import download  # type: ignore[import]  # noqa: PLC0415

            download(model)
            self._spacy_nlp = spacy.load(model)

        return self._spacy_nlp

    def _get_nltk_lemmatizer(self) -> Any:
        """Load and cache NLTK WordNetLemmatizer."""
        if self._nltk_lemmatizer is not None:
            return self._nltk_lemmatizer

        try:
            import nltk  # type: ignore[import]  # noqa: PLC0415
            from nltk.stem import (  # type: ignore[import]  # noqa: PLC0415
                WordNetLemmatizer,
            )
        except ImportError as exc:
            raise ImportError(
                "NLTK is required for lemmatizer='nltk'.  Install: pip install nltk"
            ) from exc

        # Ensure wordnet data is available
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)

        self._nltk_lemmatizer = WordNetLemmatizer()
        return self._nltk_lemmatizer

    def _get_stemmer(self) -> Any:
        """Load and cache the configured stemmer."""
        if self._stemmer_obj is not None:
            return self._stemmer_obj

        cfg = self.config
        try:
            from nltk.stem import (  # type: ignore[import]  # noqa: PLC0415
                LancasterStemmer,
                PorterStemmer,
                SnowballStemmer,
            )
        except ImportError as exc:
            raise ImportError(
                "NLTK is required for stemming.  Install: pip install nltk"
            ) from exc

        if cfg.stemmer == "porter":
            self._stemmer_obj = PorterStemmer()
        elif cfg.stemmer == "snowball":
            self._stemmer_obj = SnowballStemmer(cfg.stemmer_language)
        elif cfg.stemmer == "lancaster":
            self._stemmer_obj = LancasterStemmer()
        else:
            raise ValueError(f"Unknown stemmer: {cfg.stemmer!r}")

        return self._stemmer_obj

    def _get_stopwords(self) -> frozenset[str]:
        """Load and cache stopwords."""
        if self._stopwords is not None:
            return self._stopwords

        try:
            import nltk  # type: ignore[import]  # noqa: PLC0415
            from nltk.corpus import (  # type: ignore[import]  # noqa: PLC0415
                stopwords as sw,
            )

            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)
            self._stopwords = frozenset(sw.words("english"))
        except (ImportError, LookupError):
            self._stopwords = BUILTIN_STOPWORDS

        return self._stopwords

    def __repr__(self) -> str:
        return f"NLPEnricher(config={self.config!r})"
