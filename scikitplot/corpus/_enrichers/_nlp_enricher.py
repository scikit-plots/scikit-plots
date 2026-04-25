# scikitplot/corpus/_enrichers/_nlp_enricher.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
NLP enrichment component for ``CorpusDocument``.

Populates NLP enrichment fields on
:class:`~scikitplot.corpus._schema.CorpusDocument`:

- ``tokens``            — tokenised word list (after stopword/punct filtering)
- ``lemmas``            — lemmatised tokens (spaCy or NLTK WordNet)
- ``stems``             — stemmed tokens (Porter, Snowball, Lancaster, CUSTOM)
- ``keywords``          — extracted keywords with optional TF-IDF scores
- ``language``          — detected/assigned ISO 639-1 language code (new)

Extended fields written to ``CorpusDocument.metadata`` when enabled:

- ``pos_tags``          — part-of-speech tag list (``["NN","VBZ",…]``)
- ``ner_entities``      — named entities ``[{"text":"..","label":"ORG"}]``
- ``sentence_count``    — number of sentences in ``text``
- ``char_count``        — character count of raw text
- ``token_scores``      — ``{token: tfidf_score}`` mapping
- ``type_token_ratio``  — lexical diversity (unique tokens / total tokens)

Multi-language support:

``language: str | list[str] | None`` accepts:

* ``None``          — auto-detect per document using :func:`detect_script`
* ``"en"``          — single language ISO code or NLTK name
* ``["en", "ar"]``  — union stopwords for mixed-language documents

Supported languages:

200+ world languages via :mod:`~._language_data`.  Stopwords are
unioned across all specified languages.  Tokenisation uses:

- ``"simple"`` (regex ``\\w+``) — works for all Latin/spaced scripts
- ``"nltk"``    — requires NLTK + punkt_tab data
- ``"spacy"``   — requires a loaded spaCy model
- ``"custom"``  — any callable or :class:`~._custom_tokenizer.TokenizerProtocol`

Backends are lazy-loaded:

No NLP library is imported at module level.  Each backend is imported on
first use and cached per instance.  The enricher functions with zero optional
deps (falls back to regex tokenisation when NLTK/spaCy are absent).

Supports Python 3.8 through 3.15.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import (  # noqa: F401
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Union,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BUILTIN_STOPWORDS",
    "EnricherConfig",
    "NLPEnricher",
]

# ---------------------------------------------------------------------------
# Built-in English stopwords (zero-dep fallback)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Regex helpers (module-level, compiled once)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_PUNCT_RE = re.compile(r"^[^\w]+$", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?。！？؟।۔།…])\s*(?=\S)",  # noqa: RUF001
    re.UNICODE,
)

# Snowball-supported NLTK language names (guard against silent crashes)
_SNOWBALL_LANGUAGES: frozenset[str] = frozenset(
    {
        "arabic",
        "danish",
        "dutch",
        "english",
        "finnish",
        "french",
        "german",
        "hungarian",
        "italian",
        "norwegian",
        "porter",
        "portuguese",
        "romanian",
        "russian",
        "spanish",
        "swedish",
    }
)


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class EnricherConfig:
    r"""Configuration for :class:`NLPEnricher`.

    Parameters
    ----------
    language : str or list[str] or None, optional
        Language(s) to use for stopword loading and tokenisation.
        Accepts:

        * ``None``         — auto-detect per document from the text content
        * ``"en"``         — ISO 639-1 two-letter code, resolved to NLTK name
        * ``"english"``    — NLTK-style full language name
        * ``["en", "ar"]`` — multi-language: union stopwords for both

        Supports 200+ world languages via :mod:`~._language_data`.
        See :func:`~._language_data.coerce_language` for the full resolution
        chain, including regional aliases (``"chilean_spanish"`` → ``"spanish"``,
        ``"new_zealand_english"`` → ``"english"``, etc.).

    tokenizer : str
        Tokenisation backend:

        * ``"simple"`` (default) — regex ``\\w+`` (Unicode-aware, no deps)
        * ``"nltk"``             — ``nltk.tokenize.word_tokenize``
        * ``"spacy"``            — spaCy tokenizer (requires ``spacy_model``)
        * ``"custom"``           — use ``custom_tokenizer``

    custom_tokenizer : callable or TokenizerProtocol or None
        User tokenizer for ``tokenizer="custom"``.  Accepts any object with a
        ``tokenize(text: str) -> list[str]`` method, or a plain callable.
        Useful for MeCab (Japanese), jieba (Chinese), camel-tools (Arabic),
        Stanza (100+ languages), HuggingFace tokenizers, etc.

    spacy_model : str
        spaCy model name, used for ``tokenizer="spacy"`` or
        ``lemmatizer="spacy"`` or ``pos_tags=True`` or ``ner_entities=True``.
        Example: ``"en_core_web_sm"``.

    lemmatizer : str or None
        Lemmatisation backend: ``"spacy"``, ``"nltk"``, ``"custom"``, or
        ``None`` (skip).

    custom_lemmatizer : callable or LemmatizerProtocol or None
        User lemmatizer for ``lemmatizer="custom"``.  Must have a
        ``lemmatize(word: str, pos: str | None = None) -> str`` method,
        or be a plain callable.

    stemmer : str or None
        Stemming backend: ``"porter"``, ``"snowball"``, ``"lancaster"``,
        ``"custom"``, or ``None`` (skip).

    custom_stemmer : callable or StemmerProtocol or None
        User stemmer for ``stemmer="custom"``.  Must have a
        ``stem(word: str) -> str`` method, or be a plain callable.

    stemmer_language : str or list[str] or None
        Language(s) for the Snowball stemmer.  Accepts the same forms as
        *language*.  Defaults to ``"english"``.

    keyword_extractor : str or None
        Keyword extraction backend: ``"frequency"``, ``"tfidf"``,
        ``"yake"``, ``"keybert"``, or ``None`` (skip).

        * ``"frequency"`` — top-N by raw term count (no deps)
        * ``"tfidf"``     — top-N by within-document TF-IDF score (no deps)
        * ``"yake"``      — unsupervised statistical (requires ``yake``)
        * ``"keybert"``   — embedding-based (requires ``keybert``)

    keyword_extractor_kwargs : dict or None
        Extra kwargs forwarded to the keyword extractor (e.g. YAKE
        language setting, KeyBERT model name).

    max_keywords : int
        Maximum number of keywords to extract per document.

    save_token_scores : bool
        When ``True`` and ``keyword_extractor="tfidf"``, store per-token
        TF-IDF scores as a ``token_scores: dict`` in document metadata.

    lowercase_tokens : bool
        Lowercase all tokens before further processing.

    remove_stopwords : bool
        Remove stopwords.  Stopword language(s) follow *language*.

    extra_stopwords : frozenset[str] or None
        Additional custom stopwords merged with the detected/specified list.

    min_token_length : int
        Discard tokens shorter than this (after lowercasing).

    remove_punctuation : bool
        Remove tokens that are entirely ASCII punctuation.

    strip_unicode_punctuation : bool
        Remove Unicode punctuation characters from token text (superset of
        *remove_punctuation*; handles CJK ``。！？``, Arabic ``،؟``, etc.).

    pos_tags : bool
        When ``True``, populate a ``pos_tags`` list in document metadata
        (requires ``tokenizer="spacy"`` or ``lemmatizer="spacy"``).

    ner_entities : bool
        When ``True``, populate a ``ner_entities`` list in document metadata
        (requires ``tokenizer="spacy"`` or ``lemmatizer="spacy"``).

    sentence_count : bool
        When ``True``, compute and store the sentence count in document
        metadata (uses multi-script regex, no external deps).

    char_count : bool
        When ``True``, store raw character count in document metadata.

    type_token_ratio : bool
        When ``True``, store lexical diversity (unique/total tokens) in
        document metadata.  Useful for LLM context quality assessment.

    Notes
    -----
    **User note:** For RAG pipelines:

    * ``tokenizer="simple"`` + ``keyword_extractor="tfidf"`` + no
      stemmer/lemmatizer is fast and works for all Latin-script languages.
    * For multilingual RAG: set ``language=["en", "ar"]`` and the enricher
      will union stopwords for both languages automatically.
    * For linguistic research: ``tokenizer="spacy"`` + ``lemmatizer="spacy"``
      + ``pos_tags=True`` + ``ner_entities=True`` gives the richest output.
    * For LLM fine-tuning data: enable ``sentence_count``, ``char_count``,
      ``type_token_ratio``, and ``save_token_scores`` to add quality signals
      to each document.

    **Developer note:** All NLP backends are lazy-loaded and cached on
    ``NLPEnricher._*`` attributes.  The class is NOT thread-safe.  Use
    separate instances per thread.
    """  # noqa: D205, RUF002

    # --- Language ---
    language: Any = field(default=None, hash=False, compare=False)

    # --- Tokenization ---
    tokenizer: str = "simple"
    custom_tokenizer: Any = field(default=None, hash=False, compare=False)
    spacy_model: str = "en_core_web_sm"

    # --- Lemmatization ---
    lemmatizer: str | None = None
    custom_lemmatizer: Any = field(default=None, hash=False, compare=False)

    # --- Stemming ---
    stemmer: str | None = None
    custom_stemmer: Any = field(default=None, hash=False, compare=False)
    stemmer_language: Any = field(default="english", hash=False, compare=False)

    # --- Keywords ---
    keyword_extractor: str | None = "frequency"
    keyword_extractor_kwargs: Any = field(default=None, hash=False, compare=False)
    max_keywords: int = 20
    save_token_scores: bool = False

    # --- Filtering ---
    lowercase_tokens: bool = True
    remove_stopwords: bool = True
    extra_stopwords: Any = field(default=None, hash=False, compare=False)
    min_token_length: int = 2
    remove_punctuation: bool = True
    strip_unicode_punctuation: bool = False

    # --- Extended fields (written to metadata) ---
    pos_tags: bool = False
    ner_entities: bool = False
    sentence_count: bool = False
    char_count: bool = False
    type_token_ratio: bool = False

    def __post_init__(self) -> None:  # noqa: PLR0912
        valid_tokenizers = ("simple", "nltk", "spacy", "custom")
        if self.tokenizer not in valid_tokenizers:
            raise ValueError(
                f"tokenizer must be one of {valid_tokenizers}, got {self.tokenizer!r}"
            )
        if self.tokenizer == "custom" and self.custom_tokenizer is None:
            raise ValueError("custom_tokenizer must be set when tokenizer='custom'.")
        valid_lemmers = (None, "spacy", "nltk", "custom")
        if self.lemmatizer not in valid_lemmers:
            raise ValueError(
                f"lemmatizer must be one of {valid_lemmers}, got {self.lemmatizer!r}"
            )
        if self.lemmatizer == "custom" and self.custom_lemmatizer is None:
            raise ValueError("custom_lemmatizer must be set when lemmatizer='custom'.")
        valid_stemmers = (None, "porter", "snowball", "lancaster", "custom")
        if self.stemmer not in valid_stemmers:
            raise ValueError(
                f"stemmer must be one of {valid_stemmers}, got {self.stemmer!r}"
            )
        if self.stemmer == "custom" and self.custom_stemmer is None:
            raise ValueError("custom_stemmer must be set when stemmer='custom'.")
        valid_kw = (None, "frequency", "tfidf", "yake", "keybert")
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
        # Validate Snowball language when snowball is selected
        if self.stemmer == "snowball":
            from .._chunkers._language_data import coerce_language  # noqa: PLC0415

            langs = coerce_language(self.stemmer_language, default="english")
            for lang in langs:
                if lang not in _SNOWBALL_LANGUAGES:
                    raise ValueError(
                        f"stemmer='snowball' does not support language "
                        f"{lang!r}. Supported: {sorted(_SNOWBALL_LANGUAGES)}. "
                        f"Use stemmer='custom' to supply your own stemmer."
                    )
        # Validate extra_stopwords type
        if self.extra_stopwords is not None:  # noqa: SIM102
            if not isinstance(self.extra_stopwords, (frozenset, set)):
                raise TypeError(
                    f"extra_stopwords must be frozenset or set, "
                    f"got {type(self.extra_stopwords).__name__!r}."
                )
        # pos_tags / ner_entities require spaCy
        if (self.pos_tags or self.ner_entities) and self.tokenizer not in (
            "spacy",
            "custom",
        ):
            logger.warning(
                "pos_tags=True or ner_entities=True without tokenizer='spacy'. "
                "POS/NER requires spaCy; set tokenizer='spacy' and a "
                "spacy_model, or provide a custom_tokenizer with POS/NER support."
            )


# =====================================================================
# NLPEnricher
# =====================================================================


class NLPEnricher:
    """Pipeline component that populates NLP enrichment fields on
    :class:`~scikitplot.corpus._schema.CorpusDocument`.

    Parameters
    ----------
    config : EnricherConfig or None, optional
        Enrichment settings.  ``None`` uses all defaults.

    Notes
    -----
    **User note:** Insert after ``TextNormalizer`` and before
    ``EmbeddingEngine`` in the pipeline::

        source → reader → chunker → filter → normalizer
          → **enricher** → embedder

    The enricher reads ``doc.normalized_text`` when available,
    falling back to ``doc.text``.  When ``language=None``, the
    dominant script of each document is detected independently via
    :func:`~._custom_tokenizer.detect_script`.

    **Developer note:** All NLP backends are lazy-loaded and cached on
    ``self._*`` attributes.  The class is NOT thread-safe.  Use separate
    instances per thread.

    Examples
    --------
    >>> cfg = EnricherConfig(
    ...     language=["en", "ar"],
    ...     keyword_extractor="tfidf",
    ...     sentence_count=True,
    ...     char_count=True,
    ...     save_token_scores=True,
    ... )
    >>> enricher = NLPEnricher(cfg)
    >>> # docs = enricher.enrich_documents([doc1, doc2])
    """  # noqa: D205

    def __init__(self, config: EnricherConfig | None = None) -> None:
        self.config = config or EnricherConfig()
        # Lazy-loaded backends (one per instance, not shared across threads)
        self._spacy_nlp: Any = None
        self._nltk_lemmatizer: Any = None
        self._stemmer_obj: Any = None
        self._stopwords_cache: dict[str, frozenset[str]] = {}

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
            Documents to enrich.  Original objects are not mutated;
            new instances are returned via ``doc.replace()``.
        overwrite : bool, optional
            When ``True``, re-enrich even if NLP fields are already set.
            Default ``False`` (skip already-enriched documents).

        Returns
        -------
        list[CorpusDocument]
            New document instances with NLP and metadata fields populated.

        Notes
        -----
        **Developer note:** Documents are processed sequentially.
        For large corpora, call in batches to control memory.
        """
        out: list[Any] = []
        n_enriched = 0

        for doc in documents:
            has_tokens = getattr(doc, "tokens", None) is not None
            if not overwrite and has_tokens:
                out.append(doc)
                continue

            text: str = (
                getattr(doc, "normalized_text", None) or getattr(doc, "text", "") or ""
            )
            if not text.strip():
                out.append(doc)
                continue

            # Resolve language for this document
            langs = self._resolve_languages(text)
            stopwords = self._get_stopwords_for(langs)

            # Tokenize
            raw_tokens, spacy_doc_obj = self._tokenize_with_spacy(text)
            tokens = self._filter_tokens(raw_tokens, stopwords)

            # Stemming / lemmatization (mutually exclusive; stemmer takes priority)
            lemmas: list[str] | None = None
            stems: list[str] | None = None
            if self.config.stemmer:
                stems = self._stem(tokens)
            elif self.config.lemmatizer:
                lemmas = self._lemmatize(tokens, spacy_doc_obj)

            keywords = self._extract_keywords(text, tokens)
            token_scores: dict[str, float] | None = (
                self._compute_tfidf_scores(tokens)
                if self.config.save_token_scores
                else None
            )

            # Build extra metadata dict
            extra_meta: dict[str, Any] = {}
            if self.config.pos_tags and spacy_doc_obj is not None:
                extra_meta["pos_tags"] = [
                    tok.tag_ for tok in spacy_doc_obj if not tok.is_space
                ]
            if self.config.ner_entities and spacy_doc_obj is not None:
                extra_meta["ner_entities"] = [
                    {"text": ent.text, "label": ent.label_}
                    for ent in spacy_doc_obj.ents
                ]
            if self.config.sentence_count:
                extra_meta["sentence_count"] = self._count_sentences(text)
            if self.config.char_count:
                extra_meta["char_count"] = len(text)
            if self.config.type_token_ratio and tokens:
                extra_meta["type_token_ratio"] = round(
                    len(set(tokens)) / len(tokens), 4
                )
            if token_scores:
                extra_meta["token_scores"] = token_scores

            # Build the replace() kwargs
            replace_kwargs: dict[str, Any] = {
                "tokens": tokens or None,
                "lemmas": lemmas,
                "stems": stems,
                "keywords": keywords,
            }
            # Merge extra metadata into existing doc metadata
            if extra_meta:
                existing_meta = dict(getattr(doc, "metadata", {}) or {})
                existing_meta.update(extra_meta)
                replace_kwargs["metadata"] = existing_meta

            out.append(doc.replace(**replace_kwargs))
            n_enriched += 1

        logger.info(
            "NLPEnricher: enriched=%d, skipped=%d, total=%d",
            n_enriched,
            len(documents) - n_enriched,
            len(documents),
        )
        return out

    # ------------------------------------------------------------------
    # Language resolution
    # ------------------------------------------------------------------

    def _resolve_languages(self, text: str) -> list[str]:
        """Return canonical NLTK language name(s) for *text*.

        When ``language=None``, auto-detects from the first 300 chars.
        Otherwise normalises the configured value.

        Parameters
        ----------
        text : str
            Document text used for auto-detection.

        Returns
        -------
        list[str]
            One or more canonical NLTK language names.
        """
        from .._chunkers._language_data import coerce_language  # noqa: PLC0415

        lang_cfg = self.config.language
        if lang_cfg is None:
            # Auto-detect
            from .._chunkers._custom_tokenizer import (  # noqa: PLC0415
                ScriptType,
                detect_script,
            )

            script = detect_script(text[:300])
            _SCRIPT_LANG: dict[str, str] = {  # noqa: N806
                ScriptType.LATIN.value: "english",
                ScriptType.ARABIC.value: "arabic",
                ScriptType.DEVANAGARI.value: "hindi",
                ScriptType.CJK.value: "chinese",
                ScriptType.CYRILLIC.value: "russian",
                ScriptType.GREEK.value: "greek",
                ScriptType.HEBREW.value: "hebrew",
                ScriptType.ETHIOPIC.value: "amharic",
                ScriptType.GEORGIAN.value: "georgian",
                ScriptType.UNKNOWN.value: "english",
                ScriptType.MIXED.value: "english",
            }
            detected = _SCRIPT_LANG.get(script.value, "english")
            return [detected]
        return coerce_language(lang_cfg, default="english")

    # ------------------------------------------------------------------
    # Stopword loading (cached per language combination)
    # ------------------------------------------------------------------

    def _get_stopwords_for(self, langs: list[str]) -> frozenset[str]:
        """Return union stopwords for *langs*, with NLTK fallback and extras.

        Parameters
        ----------
        langs : list[str]
            Canonical NLTK language names.

        Returns
        -------
        frozenset[str]
            Combined stopword set.
        """
        cache_key = "|".join(sorted(langs))
        if cache_key in self._stopwords_cache:
            sw = self._stopwords_cache[cache_key]
        else:
            from .._chunkers._language_data import (  # noqa: PLC0415
                BUILTIN_LANG_STOPWORDS,
                NLTK_STOPWORD_LANGUAGES,
            )

            result: set = set()
            for lang in langs:
                # Try NLTK first (richer lists)
                if lang in NLTK_STOPWORD_LANGUAGES:
                    try:
                        import nltk  # type: ignore[import]  # noqa: PLC0415
                        from nltk.corpus import (  # type: ignore[import]  # noqa: PLC0415
                            stopwords as sw_corpus,
                        )

                        try:
                            nltk.data.find("corpora/stopwords")
                        except LookupError:
                            nltk.download("stopwords", quiet=True)
                        result |= set(sw_corpus.words(lang))
                        continue
                    except (ImportError, OSError, LookupError):
                        pass  # fall through to built-in
                # Built-in fallback
                builtin = BUILTIN_LANG_STOPWORDS.get(lang)
                if builtin:
                    result |= builtin
                else:
                    result |= BUILTIN_STOPWORDS  # universal English fallback

            sw = frozenset(result)
            self._stopwords_cache[cache_key] = sw

        # Merge extra_stopwords
        extra = self.config.extra_stopwords
        if extra:
            sw = sw | frozenset(extra)
        return sw

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize_with_spacy(self, text: str) -> tuple:
        """Tokenise *text* and optionally return a spaCy Doc.

        Returns
        -------
        tuple[list[str], spacy.Doc or None]
            Raw token strings and the spaCy Doc object (or ``None`` when
            a non-spaCy backend is used).
        """
        cfg = self.config

        if cfg.tokenizer == "custom" and cfg.custom_tokenizer is not None:
            tok_fn = cfg.custom_tokenizer
            if hasattr(tok_fn, "tokenize"):
                tokens = tok_fn.tokenize(text)
            elif callable(tok_fn):
                tokens = tok_fn(text)
            else:
                raise TypeError(
                    f"custom_tokenizer must be callable or have .tokenize(), "
                    f"got {type(tok_fn).__name__!r}."
                )
            return list(tokens), None

        if cfg.tokenizer == "spacy":
            nlp = self._get_spacy()
            doc = nlp(text)
            return [tok.text for tok in doc if not tok.is_space], doc

        # Need spaCy for POS/NER even with other tokenizers
        spacy_doc = None
        if cfg.pos_tags or cfg.ner_entities:
            try:
                nlp = self._get_spacy()
                spacy_doc = nlp(text)
            except Exception:  # noqa: BLE001
                logger.debug("spaCy unavailable for POS/NER; skipping.")

        if cfg.tokenizer == "nltk":
            try:
                from nltk.tokenize import (  # type: ignore[import]  # noqa: PLC0415
                    word_tokenize,
                )

                return word_tokenize(text), spacy_doc
            except ImportError:
                logger.warning("NLTK not installed; falling back to simple tokenizer.")

        # "simple" or fallback
        return _WORD_RE.findall(text), spacy_doc

    def _filter_tokens(
        self,
        tokens: list[str],
        stopwords: frozenset | None = None,
    ) -> list[str]:
        """Apply lowercase, punctuation, stopword, and length filters.

        Parameters
        ----------
        tokens : list[str]
            Raw token list from the tokenizer.
        stopwords : frozenset[str] or None, optional
            Active stopword set.  When ``None``, the stopword set is
            resolved lazily from the configured language(s).  This is
            the backward-compatible form used by
            :class:`~scikitplot.corpus._custom_hooks.CustomNLPEnricher`
            and any external wrapper that does not pre-resolve stopwords.

        Returns
        -------
        list[str]
            Filtered token list.

        Notes
        -----
        **Developer note:** Passing ``None`` triggers a full
        :meth:`_get_stopwords_for` round-trip.  Pre-resolve and pass
        explicitly in hot paths (e.g. ``enrich_documents`` loops).
        """
        if stopwords is None:
            langs = self._resolve_languages("")
            stopwords = self._get_stopwords_for(langs)
        cfg = self.config
        result: list[str] = []

        for tok in tokens:
            if cfg.lowercase_tokens:
                tok = tok.lower()  # noqa: PLW2901

            # Unicode punctuation stripping (superset of ASCII punct)
            if cfg.strip_unicode_punctuation:
                tok = "".join(  # noqa: PLW2901
                    c for c in tok if not unicodedata.category(c).startswith("P")
                )
                if not tok:
                    continue
            elif cfg.remove_punctuation and _PUNCT_RE.match(tok):
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

    def _lemmatize(  # noqa: PLR0911
        self,
        tokens: list[str],
        spacy_doc: Any,
    ) -> list[str] | None:
        """Lemmatise *tokens* using the configured backend.

        Parameters
        ----------
        tokens : list[str]
            Filtered token list.
        spacy_doc : spacy.Doc or None
            Pre-parsed spaCy Doc (reused to avoid double parsing).

        Returns
        -------
        list[str] or None
            Lemma list, or ``None`` when no lemmatizer is configured.
        """
        cfg = self.config
        if cfg.lemmatizer is None:
            return None

        if cfg.lemmatizer == "custom" and cfg.custom_lemmatizer is not None:
            lm = cfg.custom_lemmatizer
            if hasattr(lm, "lemmatize"):
                return [lm.lemmatize(tok) for tok in tokens]
            if callable(lm):
                return [lm(tok) for tok in tokens]

        if cfg.lemmatizer == "spacy":
            if spacy_doc is not None:
                lemma_map: dict[str, str] = {
                    t.text.lower(): t.lemma_.lower()
                    for t in spacy_doc
                    if not t.is_space
                }
                return [lemma_map.get(t, t) for t in tokens]
            # spaCy doc unavailable — try loading it
            try:
                nlp = self._get_spacy()
                doc = nlp(" ".join(tokens))
                return [tok.lemma_ for tok in doc if not tok.is_space]
            except Exception:  # noqa: BLE001
                logger.warning("spaCy lemmatization failed; returning tokens.")
                return tokens

        if cfg.lemmatizer == "nltk":
            lem = self._get_nltk_lemmatizer()
            return [lem.lemmatize(tok) for tok in tokens]

        return None

    # ------------------------------------------------------------------
    # Stemming
    # ------------------------------------------------------------------

    def _stem(self, tokens: list[str]) -> list[str] | None:
        """Stem *tokens* using the configured backend.

        Parameters
        ----------
        tokens : list[str]
            Filtered token list.

        Returns
        -------
        list[str] or None
            Stemmed token list, or ``None`` when no stemmer is configured.
        """
        cfg = self.config
        if cfg.stemmer is None:
            return None

        if cfg.stemmer == "custom" and cfg.custom_stemmer is not None:
            st = cfg.custom_stemmer
            if hasattr(st, "stem"):
                return [st.stem(tok) for tok in tokens]
            if callable(st):
                return [st(tok) for tok in tokens]

        stemmer = self._get_stemmer()
        return [stemmer.stem(tok) for tok in tokens]

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    def _extract_keywords(self, text: str, tokens: list[str]) -> list[str] | None:
        """Extract keywords from *text* / *tokens*.

        Parameters
        ----------
        text : str
            Raw document text (used by YAKE and KeyBERT).
        tokens : list[str]
            Filtered token list (used by frequency and TF-IDF).

        Returns
        -------
        list[str] or None
            Keyword list, or ``None`` when no extractor is configured.
        """
        cfg = self.config
        if cfg.keyword_extractor is None:
            return None
        if cfg.keyword_extractor == "yake":
            return self._keywords_yake(text)
        if cfg.keyword_extractor == "keybert":
            return self._keywords_keybert(text)
        if cfg.keyword_extractor == "tfidf":
            return self._keywords_tfidf(tokens)
        # "frequency" (default)
        return self._keywords_frequency(tokens)

    def _keywords_frequency(self, tokens: list[str]) -> list[str]:
        """Top-N keywords by raw term frequency.

        Parameters
        ----------
        tokens : list[str]
            Filtered token list.

        Returns
        -------
        list[str]
            Top-N keyword strings.
        """
        freq: dict[str, int] = {}
        for tok in tokens:
            key = tok.lower()
            freq[key] = freq.get(key, 0) + 1
        sorted_terms = sorted(freq, key=freq.__getitem__, reverse=True)
        return sorted_terms[: self.config.max_keywords]

    def _keywords_tfidf(self, tokens: list[str]) -> list[str]:
        """Top-N keywords by within-document TF-IDF score.

        Uses a simplified IDF that treats rare terms (appearing once)
        as more informative than high-frequency terms, without requiring
        a background corpus.  This approximation works well for single-
        document keyword extraction.

        Parameters
        ----------
        tokens : list[str]
            Filtered token list.

        Returns
        -------
        list[str]
            Top-N keyword strings by TF-IDF score.
        """
        if not tokens:
            return []
        n = len(tokens)
        freq: dict[str, int] = {}
        for tok in tokens:
            k = tok.lower()
            freq[k] = freq.get(k, 0) + 1

        # TF = count / total; IDF = log(n / count) — higher for rare terms
        scores: dict[str, float] = {
            term: (count / n) * math.log(1.0 + n / count)
            for term, count in freq.items()
        }
        sorted_terms = sorted(scores, key=scores.__getitem__, reverse=True)
        return sorted_terms[: self.config.max_keywords]

    def _compute_tfidf_scores(self, tokens: list[str]) -> dict[str, float] | None:
        """Compute TF-IDF-like scores for all tokens.

        Parameters
        ----------
        tokens : list[str]
            Filtered token list.

        Returns
        -------
        dict[str, float] or None
            Mapping of term → score, rounded to 4 decimal places.
            ``None`` when *tokens* is empty.
        """
        if not tokens:
            return None
        n = len(tokens)
        freq: dict[str, int] = {}
        for tok in tokens:
            k = tok.lower()
            freq[k] = freq.get(k, 0) + 1
        return {
            term: round((count / n) * math.log(1.0 + n / count), 4)
            for term, count in freq.items()
        }

    def _keywords_yake(self, text: str) -> list[str] | None:
        """Extract keywords using YAKE.

        Parameters
        ----------
        text : str
            Raw document text.

        Returns
        -------
        list[str] or None
        """
        try:
            import yake  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            logger.warning("YAKE not installed; falling back to frequency keywords.")
            return None

        kwargs: dict[str, Any] = {
            "top": self.config.max_keywords,
            "dedupLim": 0.9,
        }
        if self.config.keyword_extractor_kwargs:
            kwargs.update(self.config.keyword_extractor_kwargs)
        extractor = yake.KeywordExtractor(**kwargs)
        kws = extractor.extract_keywords(text)
        return [kw for kw, _score in kws]

    def _keywords_keybert(self, text: str) -> list[str] | None:
        """Extract keywords using KeyBERT.

        Parameters
        ----------
        text : str
            Raw document text.

        Returns
        -------
        list[str] or None
        """
        try:
            from keybert import KeyBERT  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            logger.warning("KeyBERT not installed; falling back to frequency keywords.")
            return None

        kwargs: dict[str, Any] = {"top_n": self.config.max_keywords}
        if self.config.keyword_extractor_kwargs:
            kwargs.update(self.config.keyword_extractor_kwargs)
        model = KeyBERT()
        kws = model.extract_keywords(text, **kwargs)
        return [kw for kw, _score in kws]

    # ------------------------------------------------------------------
    # Extended-field helpers
    # ------------------------------------------------------------------

    def _count_sentences(self, text: str) -> int:
        """Count sentences in *text* using the multi-script regex.

        Parameters
        ----------
        text : str
            Document text.

        Returns
        -------
        int
            Estimated sentence count (always >= 1 for non-empty text).
        """
        if not text.strip():
            return 0
        parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
        return max(1, len(parts))

    # ------------------------------------------------------------------
    # Lazy backend loaders
    # ------------------------------------------------------------------

    def _get_spacy(self) -> Any:
        """Load and cache the spaCy NLP pipeline.

        Returns
        -------
        spacy.Language
            Loaded spaCy model.

        Raises
        ------
        ImportError
            If spaCy is not installed.
        OSError
            If the model is not installed.
        """
        if self._spacy_nlp is not None:
            return self._spacy_nlp

        try:
            import spacy  # type: ignore[import]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "spaCy is required for tokenizer='spacy', lemmatizer='spacy', "
                "pos_tags=True, or ner_entities=True.  "
                "Install: pip install spacy"
            ) from exc

        model = self.config.spacy_model
        try:
            self._spacy_nlp = spacy.load(model)
        except OSError:
            logger.info("spaCy model %r not found; attempting download.", model)
            try:
                from spacy.cli import download  # type: ignore[import]  # noqa: PLC0415

                download(model)
                self._spacy_nlp = spacy.load(model)
            except Exception as exc:
                raise OSError(
                    f"spaCy model {model!r} is not installed. "
                    f"Install with: python -m spacy download {model}"
                ) from exc

        return self._spacy_nlp

    def _get_nltk_lemmatizer(self) -> Any:
        """Load and cache the NLTK WordNetLemmatizer.

        Returns
        -------
        WordNetLemmatizer
            Cached lemmatizer instance.

        Raises
        ------
        ImportError
            If NLTK is not installed.
        """
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

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

        self._nltk_lemmatizer = WordNetLemmatizer()
        return self._nltk_lemmatizer

    def _get_stemmer(self) -> Any:
        """Load and cache the configured NLTK stemmer.

        Returns
        -------
        object
            NLTK stemmer with a ``.stem(word)`` method.

        Raises
        ------
        ImportError
            If NLTK is not installed.
        ValueError
            If the stemmer name is not recognised.
        """
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
            from .._chunkers._language_data import coerce_language  # noqa: PLC0415

            langs = coerce_language(cfg.stemmer_language, default="english")
            lang = langs[0]
            self._stemmer_obj = SnowballStemmer(lang)
        elif cfg.stemmer == "lancaster":
            self._stemmer_obj = LancasterStemmer()
        else:
            raise ValueError(
                f"Unknown stemmer: {cfg.stemmer!r}. "
                f"Use one of: porter, snowball, lancaster, custom."
            )

        return self._stemmer_obj

    # ------------------------------------------------------------------
    # Backward-compatibility shims
    # These provide a stable public surface for CustomNLPEnricher and
    # any user code that wraps NLPEnricher, regardless of internal
    # method refactors across library versions.
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize *text* and return a token list (no spaCy Doc).

        Compatibility shim over :meth:`_tokenize_with_spacy`.
        Callers that only need token strings (not the spaCy Doc object)
        should use this method.  :class:`CustomNLPEnricher` uses this
        as the fallback when a custom tokenizer raises.

        Parameters
        ----------
        text : str
            Input text to tokenize.

        Returns
        -------
        list[str]
            Token strings.

        Notes
        -----
        **Developer note:** Delegates to :meth:`_tokenize_with_spacy`
        and discards the second element of the returned tuple.  When the
        spaCy Doc is needed (POS tags, NER), call
        :meth:`_tokenize_with_spacy` directly.
        """
        tokens, _ = self._tokenize_with_spacy(text)
        return tokens

    @property
    def _stopwords(self) -> frozenset:
        """Return the active stopword set for the configured language(s).

        Compatibility property replacing the old ``self._stopwords``
        instance attribute.  Resolves and returns the union stopword set
        for :attr:`EnricherConfig.language`.

        :class:`CustomNLPEnricher` reads this to temporarily override
        stopwords without needing to know about ``_stopwords_cache``.

        Returns
        -------
        frozenset[str]
            Resolved stopword set for the current language config.
        """
        langs = self._resolve_languages("")
        return self._get_stopwords_for(langs)

    @_stopwords.setter
    def _stopwords(self, value: frozenset) -> None:
        """Store a one-shot stopword override in the cache.

        :class:`CustomNLPEnricher` uses this to temporarily inject a
        custom stopword set.  The override key ``__override__`` is
        consumed automatically by the next :meth:`_get_stopwords_for`
        call that encounters it.

        Parameters
        ----------
        value : frozenset[str]
            Replacement stopword set.
        """
        self._stopwords_cache["__override__"] = value

    def _lemmatize_tokens(self, tokens: list[str]) -> list[str] | None:
        """Lemmatize *tokens* without requiring an external spaCy Doc.

        Compatibility shim for :class:`CustomNLPEnricher` and any
        wrapper that calls lemmatization without managing the spaCy Doc
        object directly.  When the spaCy lemmatizer is configured, a
        fresh spaCy Doc is created internally from ``tokens``.

        Parameters
        ----------
        tokens : list[str]
            Filtered token list.

        Returns
        -------
        list[str] or None
            Lemma list, or ``None`` when no lemmatizer is configured.

        Notes
        -----
        **Developer note:** Calls :meth:`_lemmatize` with
        ``spacy_doc=None``.  For the spaCy backend this triggers an
        internal ``nlp(" ".join(tokens))`` call, which is slightly less
        accurate than passing a Doc parsed from the full original text.
        Use :meth:`_lemmatize` with a pre-parsed Doc for best results.
        """
        return self._lemmatize(tokens, spacy_doc=None)

    def __repr__(self) -> str:
        cfg = self.config
        parts = [
            f"tokenizer={cfg.tokenizer!r}",
            f"lemmatizer={cfg.lemmatizer!r}",
            f"stemmer={cfg.stemmer!r}",
            f"keyword_extractor={cfg.keyword_extractor!r}",
        ]
        # Show enabled extended-field flags so the repr is fully informative
        flags = [
            name
            for name in (
                "pos_tags",
                "ner_entities",
                "sentence_count",
                "char_count",
                "type_token_ratio",
                "save_token_scores",
            )
            if getattr(cfg, name, False)
        ]
        if flags:
            parts.append("enabled_fields=[" + ", ".join(repr(f) for f in flags) + "]")
        return "NLPEnricher(" + ", ".join(parts) + ")"
