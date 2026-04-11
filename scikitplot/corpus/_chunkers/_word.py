# scikitplot/corpus/_chunkers/_word.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Word-level text processor for corpus construction.

Provides tokenisation, stemming, lemmatisation, stopword removal,
n-gram extraction, and word-level corpus building with pluggable
backends (pure Python, NLTK, spaCy, Gensim).

Designed for:
- Vocabulary analysis and corpus statistics
- Pre-processing for Bag-of-Words / TF-IDF pipelines
- Gensim Dictionary / Corpus construction (RAG dense retrieval)
- LLM fine-tuning data normalisation
- Linguistic feature extraction for ML feature engineering
"""

from __future__ import annotations

import logging
import re
import string
import unicodedata
from dataclasses import dataclass, field  # noqa: F401
from enum import Enum
from typing import Any, Callable, Final, List, Optional, Union  # noqa: F401

from .._types import Chunk, ChunkerConfig, ChunkResult
from ._custom_tokenizer import (
    FunctionLemmatizer,
    FunctionStemmer,
    FunctionTokenizer,
    LemmatizerProtocol,
    StemmerProtocol,
    TokenizerProtocol,
)
from ._language_data import (  # noqa: F401
    NLTK_STOPWORD_LANGUAGES,
    coerce_language,
    resolve_stopwords,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LemmatizationBackend",
    "StemmingBackend",
    "StopwordSource",
    "TokenizerBackend",
    "WordChunker",
    "WordChunkerConfig",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_PUNCT_TABLE: Final[dict] = str.maketrans("", "", string.punctuation)

# Snowball supports a fixed set of languages; guard against silent crashes.
# Source: nltk.stem.snowball.SnowballStemmer.languages (NLTK 3.x)
_SNOWBALL_SUPPORTED_LANGUAGES: Final[frozenset] = frozenset(
    [
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
    ]
)

# Minimal built-in English stopword set (no external deps required).
_BUILTIN_STOPWORDS: Final[frozenset[str]] = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
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
        "are",
        "was",
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
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "as",
        "if",
        "then",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "up",
        "out",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
    ]
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TokenizerBackend(str, Enum):
    """Word tokenisation backend.

    Values
    ------
    SIMPLE
        Regex whitespace split + ASCII punctuation strip.  No external deps.
    NLTK
        ``nltk.word_tokenize``.  Requires ``nltk`` and ``punkt_tab`` data.
    SPACY
        spaCy pipeline tokenizer.  Requires ``spacy`` and a loaded model.
    CUSTOM
        User-supplied :class:`~._custom_tokenizer.TokenizerProtocol` or
        ``Callable[[str], list[str]]`` stored in
        :attr:`WordChunkerConfig.custom_tokenizer`.
        Use this to plug in MeCab (Japanese), jieba (Chinese),
        camel-tools (Arabic/Ottoman), stanza (100+ languages), or any
        HuggingFace / third-party tokenizer.
    """

    SIMPLE = "simple"
    NLTK = "nltk"
    SPACY = "spacy"
    CUSTOM = "custom"


class StemmingBackend(str, Enum):
    """Stemming algorithm.

    Values
    ------
    NONE
        No stemming applied.
    PORTER
        NLTK PorterStemmer — English only.
    SNOWBALL
        NLTK SnowballStemmer — English, German, French, Spanish, Dutch,
        Portuguese, Italian, Swedish, Norwegian, Danish, Finnish, Russian,
        Hungarian, Romanian.  Unsupported languages raise ``ValueError``
        at construction time.
    LANCASTER
        NLTK LancasterStemmer — English only, aggressive.
    CUSTOM
        User-supplied :class:`~._custom_tokenizer.StemmerProtocol` or
        ``Callable[[str], str]`` stored in
        :attr:`WordChunkerConfig.custom_stemmer`.
    """

    NONE = "none"
    PORTER = "porter"
    SNOWBALL = "snowball"
    LANCASTER = "lancaster"
    CUSTOM = "custom"


class LemmatizationBackend(str, Enum):
    """Lemmatization backend.

    Values
    ------
    NONE
        No lemmatization applied.
    NLTK_WORDNET
        NLTK WordNetLemmatizer — English only.
    SPACY
        spaCy ``.lemma_`` — language depends on loaded model.
    CUSTOM
        User-supplied :class:`~._custom_tokenizer.LemmatizerProtocol` or
        ``Callable[[str, Optional[str]], str]`` stored in
        :attr:`WordChunkerConfig.custom_lemmatizer`.
    """

    NONE = "none"
    NLTK_WORDNET = "nltk_wordnet"
    SPACY = "spacy"
    CUSTOM = "custom"


class StopwordSource(str, Enum):
    """Stopword list source."""

    NONE = "none"
    BUILTIN = "builtin"  # _BUILTIN_STOPWORDS — no deps
    NLTK = "nltk"  # nltk.corpus.stopwords
    SPACY = "spacy"  # spacy model's stop_words


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WordChunkerConfig(ChunkerConfig):
    """Configuration for :class:`WordChunker`.

    Parameters
    ----------
    tokenizer : TokenizerBackend
        Word tokenisation strategy.
    custom_tokenizer : TokenizerProtocol or Callable[[str], list[str]] or None
        User-supplied tokenizer used when ``tokenizer=TokenizerBackend.CUSTOM``.
        Accepts any object satisfying :class:`~._custom_tokenizer.TokenizerProtocol`
        *or* a plain callable.  Callables are auto-wrapped in
        :class:`~._custom_tokenizer.FunctionTokenizer`.
        Example libraries: MeCab, jieba, camel-tools, Stanza, HuggingFace.
    stemmer : StemmingBackend
        Stemming algorithm.  Applied after lowercasing, before stopword
        removal.  Mutually exclusive with *lemmatizer* (stemmer takes
        precedence when both are not ``NONE``).
    custom_stemmer : StemmerProtocol or Callable[[str], str] or None
        User-supplied stemmer used when ``stemmer=StemmingBackend.CUSTOM``.
    lemmatizer : LemmatizationBackend
        Lemmatization backend.  Applied when *stemmer* is ``NONE``.
    custom_lemmatizer : LemmatizerProtocol or Callable or None
        User-supplied lemmatizer used when
        ``lemmatizer=LemmatizationBackend.CUSTOM``.
    stopwords : StopwordSource
        Source of stopword list used for filtering.
    custom_stopwords : frozenset[str] or None
        Additional stopwords merged with the source list.  Lowercasing
        is applied before membership testing, so case does not matter.
    spacy_model : str or None
        spaCy model name. Required for ``SPACY`` tokenizer/lemmatizer.
    nltk_language : str
        Language for NLTK stemmers and stopwords (e.g. ``"english"``).
    lowercase : bool
        Convert all tokens to lowercase before processing.
    remove_punctuation : bool
        Strip ASCII punctuation-only tokens.
    strip_unicode_punctuation : bool
        Strip *all* Unicode punctuation from tokens (superset of
        *remove_punctuation*).  Handles CJK punctuation (``。！？``),
        Arabic punctuation (``،؟``), and all other ``unicodedata``
        ``P*`` category characters.  When ``True``, *remove_punctuation*
        is implicitly satisfied and need not be set separately.
    remove_numbers : bool
        Drop tokens that are purely numeric.
    min_token_length : int
        Drop tokens shorter than this (after normalisation).
    max_token_length : int or None
        Drop tokens longer than this.  ``None`` disables the limit.
    ngram_range : tuple[int, int]
        Inclusive ``(min_n, max_n)`` n-gram range to extract alongside
        unigrams.  ``(1, 1)`` disables n-gram extraction.
    chunk_by : str
        Granularity of output :class:`~.._types.Chunk` objects.
        ``"document"`` returns one chunk per input text.
        ``"sentence"`` splits on sentence boundaries first, then
        processes each sentence as a separate chunk.
    include_offsets : bool
        Store character offsets in each chunk.
    build_gensim_corpus : bool
        If ``True``, attach a ``gensim``-compatible ``(token_id, count)``
        BoW representation to each chunk's metadata (requires Gensim).

    Notes
    -----
    **User note (multi-language):** For CJK text, set
    ``tokenizer=TokenizerBackend.CUSTOM`` with a character-level or
    morpheme-level tokenizer (jieba, MeCab, kss).  Set
    ``remove_punctuation=False, strip_unicode_punctuation=True`` to
    strip CJK punctuation without removing ideographs.

    For Arabic / Ottoman / Persian, use ``tokenizer=TokenizerBackend.CUSTOM``
    with camel-tools or Stanza.  Set ``nltk_language="arabic"`` when using
    NLTK stopwords.

    **Developer note:** Callable fields (``custom_tokenizer``,
    ``custom_stemmer``, ``custom_lemmatizer``) are excluded from ``__hash__``
    and ``__eq__`` (``hash=False, compare=False``) so that two configs with
    identical settings but different callable objects are treated as equal
    for caching purposes.  Compare callables explicitly when identity matters.
    """  # noqa: D205, RUF002

    tokenizer: TokenizerBackend = TokenizerBackend.SIMPLE
    custom_tokenizer: Any = field(default=None, hash=False, compare=False)
    stemmer: StemmingBackend = StemmingBackend.NONE
    custom_stemmer: Any = field(default=None, hash=False, compare=False)
    lemmatizer: LemmatizationBackend = LemmatizationBackend.NONE
    custom_lemmatizer: Any = field(default=None, hash=False, compare=False)
    stopwords: StopwordSource = StopwordSource.BUILTIN
    custom_stopwords: frozenset | None = None
    spacy_model: str | None = None
    nltk_language: str | list[str] | None = "english"
    """Language(s) for NLTK stopwords, Snowball stemmer, and NLTK tokenizer.

    Accepts:

    * ``"en"`` or ``"english"``  — single language (backward-compatible)
    * ``["en", "ar"]``           — multi-language: union stopwords for both
    * ``None``                   — auto-detect from text using detect_script

    All ISO 639-1 codes (``"en"``, ``"ar"``, ``"hi"``, …) and NLTK names
    (``"english"``, ``"arabic"``, …) are accepted.  Regional aliases such
    as ``"chilean_spanish"``, ``"new_zealand_english"``, and ``"ottoman_turkish"``
    are resolved automatically.  200+ languages via :mod:`._language_data`.
    """
    lowercase: bool = True
    remove_punctuation: bool = True
    strip_unicode_punctuation: bool = False
    remove_numbers: bool = False
    min_token_length: int = 2
    max_token_length: int | None = None
    ngram_range: tuple = (1, 1)
    chunk_by: str = "document"  # "document" | "sentence"
    include_offsets: bool = False
    build_gensim_corpus: bool = False


# ---------------------------------------------------------------------------
# Tokeniser implementations — pure functions
# ---------------------------------------------------------------------------


def _tokenize_simple(text: str) -> list[str]:
    """Tokenize by whitespace and strip punctuation.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        Token list.
    """
    tokens = _WHITESPACE_RE.split(text.strip())
    cleaned: list[str] = []
    for tok in tokens:
        stripped = tok.translate(_PUNCT_TABLE)
        if stripped:
            cleaned.append(stripped)
    return cleaned


def _tokenize_nltk(text: str) -> list[str]:
    """Tokenize using ``nltk.word_tokenize``.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        Token list.

    Raises
    ------
    ImportError
        If NLTK is not installed.
    """
    try:
        import nltk  # type: ignore[import-untyped]  # noqa: PLC0415
        from nltk.tokenize import (  # type: ignore[import-untyped]  # noqa: PLC0415
            word_tokenize,
        )
    except ImportError as exc:
        raise ImportError(
            "TokenizerBackend.NLTK requires 'nltk'. Install with: pip install nltk"
        ) from exc

    try:
        return word_tokenize(text)
    except LookupError:
        logger.info("Downloading NLTK 'punkt_tab' model.")
        nltk.download("punkt_tab", quiet=True)
        return word_tokenize(text)


def _tokenize_spacy(text: str, model_name: str) -> list[str]:
    """Tokenize using a spaCy pipeline.

    Parameters
    ----------
    text : str
        Input text.
    model_name : str
        Loaded spaCy model name.

    Returns
    -------
    list[str]
        Token strings.

    Raises
    ------
    ImportError
        If spaCy is not installed.
    OSError
        If *model_name* is not installed.
    """
    try:
        import spacy  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "TokenizerBackend.SPACY requires 'spacy'. Install with: pip install spacy"
        ) from exc

    nlp = spacy.load(model_name, disable=["parser", "ner"])
    doc = nlp(text)
    return [token.text for token in doc]


# ---------------------------------------------------------------------------
# Stemmer cache  (one instance per (backend, language))
# ---------------------------------------------------------------------------

_STEMMER_CACHE: dict[tuple[StemmingBackend, str], Any] = {}


def _get_stemmer(
    backend: StemmingBackend,
    language: str | list[str] | None,
) -> Any:
    """Return a cached NLTK stemmer instance.

    Parameters
    ----------
    backend : StemmingBackend
        Stemmer variant.
    language : str or list[str] or None
        Language specifier for Snowball.  Accepts the same forms as
        :func:`~._language_data.coerce_language`.  When a list is given
        the **first** Snowball-supported language is used.

    Returns
    -------
    object
        NLTK stemmer with a ``.stem(word)`` method.

    Raises
    ------
    ImportError
        If NLTK is not installed.
    ValueError
        If no Snowball-supported language can be resolved.
    """
    # Resolve language to a single canonical string before cache key lookup.
    # coerce_language handles str | list[str] | None uniformly.
    if backend == StemmingBackend.SNOWBALL:
        langs = coerce_language(language, default="english")
        # Select the first language in the list that Snowball supports.
        resolved_lang: str = "english"
        for lang in langs:
            if lang in _SNOWBALL_SUPPORTED_LANGUAGES:
                resolved_lang = lang
                break
    else:
        resolved_lang = "english"  # unused for Porter/Lancaster

    key = (backend, resolved_lang)
    if key in _STEMMER_CACHE:
        return _STEMMER_CACHE[key]

    try:
        from nltk.stem import (  # type: ignore[import-untyped]  # noqa: PLC0415
            LancasterStemmer,
            PorterStemmer,
            SnowballStemmer,
        )
    except ImportError as exc:
        raise ImportError(
            f"StemmingBackend.{backend.name} requires 'nltk'. "
            "Install with: pip install nltk"
        ) from exc

    if backend == StemmingBackend.PORTER:
        stemmer = PorterStemmer()
    elif backend == StemmingBackend.SNOWBALL:
        stemmer = SnowballStemmer(resolved_lang)
    elif backend == StemmingBackend.LANCASTER:
        stemmer = LancasterStemmer()
    else:
        raise ValueError(f"Unknown stemming backend: {backend!r}.")  # pragma: no cover

    _STEMMER_CACHE[key] = stemmer
    return stemmer


# ---------------------------------------------------------------------------
# Lemmatizer cache
# ---------------------------------------------------------------------------

_LEMMATIZER_CACHE: dict[LemmatizationBackend, Any] = {}


def _get_nltk_lemmatizer() -> Any:
    """Return a cached NLTK WordNetLemmatizer.

    Returns
    -------
    WordNetLemmatizer
        NLTK lemmatizer instance.

    Raises
    ------
    ImportError
        If NLTK is not installed.
    LookupError
        If WordNet data is absent and download fails.
    """
    if LemmatizationBackend.NLTK_WORDNET in _LEMMATIZER_CACHE:
        return _LEMMATIZER_CACHE[LemmatizationBackend.NLTK_WORDNET]

    try:
        import nltk  # type: ignore[import-untyped]  # noqa: PLC0415
        from nltk.stem import (  # type: ignore[import-untyped]  # noqa: PLC0415
            WordNetLemmatizer,
        )
    except ImportError as exc:
        raise ImportError(
            "LemmatizationBackend.NLTK_WORDNET requires 'nltk'. "
            "Install with: pip install nltk"
        ) from exc

    try:
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")  # trigger corpus load to catch LookupError
    except LookupError:
        logger.info("Downloading NLTK 'wordnet' corpus.")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        lemmatizer = WordNetLemmatizer()

    _LEMMATIZER_CACHE[LemmatizationBackend.NLTK_WORDNET] = lemmatizer
    return lemmatizer


# ---------------------------------------------------------------------------
# Stopword loaders
# ---------------------------------------------------------------------------


def _load_stopwords(  # noqa: PLR0912
    source: StopwordSource,
    language: str | list[str] | None,
) -> frozenset:
    """Load stopwords from the specified source.

    Parameters
    ----------
    source : StopwordSource
        Where to load stopwords from.
    language : str or list[str] or None
        Language identifier(s) for NLTK/spaCy sources.  Accepts ISO codes,
        NLTK names, lists thereof, or ``None`` (falls back to English).
        When a list is provided the stopword sets are unioned.

    Returns
    -------
    frozenset[str]
        Lowercase stopword set.

    Raises
    ------
    ImportError
        If a required library is not installed.
    """
    if source == StopwordSource.NONE:
        return frozenset()
    if source == StopwordSource.BUILTIN:
        return _BUILTIN_STOPWORDS
    if source == StopwordSource.NLTK:
        try:
            import nltk  # type: ignore[import-untyped]  # noqa: PLC0415
            from nltk.corpus import (  # type: ignore[import-untyped]  # noqa: PLC0415
                stopwords,
            )
        except ImportError as exc:
            raise ImportError(
                "StopwordSource.NLTK requires 'nltk'. Install with: pip install nltk"
            ) from exc
        langs = coerce_language(language, default="english")
        result: set = set()
        for lang in langs:
            if lang not in NLTK_STOPWORD_LANGUAGES:
                # Fall through to built-in for this language
                from ._language_data import BUILTIN_LANG_STOPWORDS  # noqa: PLC0415

                builtin = BUILTIN_LANG_STOPWORDS.get(lang)
                if builtin:
                    result |= builtin
                continue
            try:
                result |= set(stopwords.words(lang))
            except LookupError:
                logger.info("Downloading NLTK 'stopwords' corpus.")
                nltk.download("stopwords", quiet=True)
                try:
                    result |= set(stopwords.words(lang))
                except OSError:
                    logger.warning("NLTK stopwords for %r unavailable.", lang)
        return frozenset(result)

    if source == StopwordSource.SPACY:
        try:
            import spacy  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("StopwordSource.SPACY requires 'spacy'.") from exc
        # Resolve language to a single ISO 639-1 code for spacy.blank().
        # coerce_language returns NLTK names ("english", "arabic", …).
        # spacy.blank() needs 2-letter ISO codes ("en", "ar", …).
        from ._language_data import (  # noqa: PLC0415
            ISO_TO_NLTK,
        )
        from ._language_data import (  # noqa: PLC0415
            coerce_language as _cl,
        )

        langs = _cl(language, default="english")
        first_nltk = langs[0] if langs else "english"
        # Reverse-map NLTK name → ISO code for spacy.blank()
        _NLTK_TO_ISO: dict[str, str] = {  # noqa: N806
            v: k
            for k, v in ISO_TO_NLTK.items()  # noqa: N806
            if len(k) == 2  # noqa: PLR2004
        }  # noqa: PLR2004
        iso_code = _NLTK_TO_ISO.get(first_nltk, "en")

        result_sw: set = set()
        for lng in langs:
            iso = _NLTK_TO_ISO.get(lng, "en")
            try:
                nlp = spacy.blank(iso)
                result_sw |= nlp.Defaults.stop_words
            except Exception:  # noqa: BLE001
                # Unknown ISO code → fall back to English
                try:  # noqa: SIM105
                    result_sw |= spacy.blank("en").Defaults.stop_words
                except Exception:  # noqa: BLE001
                    pass
        return frozenset(result_sw)

    raise ValueError(f"Unknown stopword source: {source!r}.")  # pragma: no cover


# ---------------------------------------------------------------------------
# N-gram extraction — pure function
# ---------------------------------------------------------------------------


def _extract_ngrams(tokens: list[str], n: int) -> list[str]:
    """Extract n-grams from a token list.

    Parameters
    ----------
    tokens : list[str]
        Input tokens.
    n : int
        Gram size.

    Returns
    -------
    list[str]
        N-gram strings joined by ``_``.
    """
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Gensim BoW helper
# ---------------------------------------------------------------------------


def _to_gensim_bow(tokens: list[str], dictionary: Any) -> list[tuple[int, int]]:
    """Convert a token list to a Gensim BoW ``(id, count)`` vector.

    Parameters
    ----------
    tokens : list[str]
        Processed token list.
    dictionary : gensim.corpora.Dictionary
        Gensim dictionary mapping tokens to integer IDs.

    Returns
    -------
    list[tuple[int, int]]
        Sparse BoW vector.
    """
    return dictionary.doc2bow(tokens)


def _strip_unicode_punct(tok: str) -> str:
    """Strip all Unicode punctuation from *tok* using ``unicodedata`` categories.

    Removes characters whose Unicode category starts with ``"P"``
    (``Po``, ``Ps``, ``Pe``, ``Pi``, ``Pf``, ``Pd``, ``Pc``).
    This covers CJK punctuation (``。！？、``), Arabic punctuation
    (``،؟؛``), and all standard Latin punctuation — a strict superset
    of :data:`_PUNCT_TABLE`.

    Parameters
    ----------
    tok : str
        Input token.

    Returns
    -------
    str
        Token with all Unicode punctuation characters removed.

    Examples
    --------
    >>> _strip_unicode_punct("hello.")
    'hello'
    >>> _strip_unicode_punct("世界。")
    '世界'
    >>> _strip_unicode_punct("مرحبا،")
    'مرحبا'
    """  # noqa: RUF002
    return "".join(c for c in tok if not unicodedata.category(c).startswith("P"))


# ---------------------------------------------------------------------------
# Core processing pipeline — pure function
# ---------------------------------------------------------------------------


def _process_tokens(  # noqa: PLR0912
    tokens: list,
    cfg: WordChunkerConfig,
    stopwords: frozenset,
    spacy_doc: Any = None,
) -> list:
    """Apply the full normalisation pipeline to raw tokens.

    Processing order:
    1. Lowercase
    2. Remove punctuation-only tokens
    3. Remove numeric tokens (optional)
    4. Stopword removal
    5. Length filtering
    6. Stemming OR lemmatization (not both)

    Parameters
    ----------
    tokens : list[str]
        Raw tokens from the tokenizer.
    cfg : WordChunkerConfig
        Processing configuration.
    stopwords : frozenset[str]
        Active stopword set.
    spacy_doc : spacy.Doc, optional
        Pre-parsed spaCy Doc for lemmatization.  Required when
        ``cfg.lemmatizer == LemmatizationBackend.SPACY``.

    Returns
    -------
    list[str]
        Normalised token list.
    """
    result = []

    for tok in tokens:
        # 1. Lowercase.
        if cfg.lowercase:
            tok = tok.lower()  # noqa: PLW2901

        # 2a. Strip ALL Unicode punctuation (superset of ASCII).
        if cfg.strip_unicode_punctuation:
            tok = _strip_unicode_punct(tok)  # noqa: PLW2901
            if not tok:
                continue
        elif cfg.remove_punctuation:
            # 2b. ASCII-only punctuation strip (legacy behaviour).
            stripped = tok.translate(_PUNCT_TABLE)
            if not stripped:
                continue
            tok = stripped  # noqa: PLW2901

        # 3. Remove pure numeric tokens.
        if cfg.remove_numbers and tok.isnumeric():
            continue

        # 4. Stopword removal (check both base set and custom additions).
        tok_lower = tok.lower() if not cfg.lowercase else tok
        if tok_lower in stopwords:
            continue
        if cfg.custom_stopwords and tok_lower in cfg.custom_stopwords:
            continue

        # 5. Length filter.
        if len(tok) < cfg.min_token_length:
            continue
        if cfg.max_token_length is not None and len(tok) > cfg.max_token_length:
            continue

        result.append(tok)

    # 6a. Stemming (takes precedence over lemmatization).
    if cfg.stemmer != StemmingBackend.NONE:
        if cfg.stemmer == StemmingBackend.CUSTOM:
            stemmer_obj = _resolve_custom_stemmer(cfg)
            result = [stemmer_obj.stem(t) for t in result]
        else:
            stemmer_obj = _get_stemmer(cfg.stemmer, cfg.nltk_language)
            result = [stemmer_obj.stem(t) for t in result]
        return result  # noqa: RET504

    # 6b. Lemmatization (only when stemming is disabled).
    if cfg.lemmatizer == LemmatizationBackend.NLTK_WORDNET:
        lemmatizer_obj = _get_nltk_lemmatizer()
        result = [lemmatizer_obj.lemmatize(t) for t in result]
    elif cfg.lemmatizer == LemmatizationBackend.SPACY and spacy_doc is not None:
        spacy_tokens = [t for t in spacy_doc if not t.is_space]
        lemma_map = {t.text.lower(): t.lemma_.lower() for t in spacy_tokens}
        result = [lemma_map.get(t, t) for t in result]
    elif cfg.lemmatizer == LemmatizationBackend.CUSTOM:
        lemmatizer_obj = _resolve_custom_lemmatizer(cfg)
        result = [lemmatizer_obj.lemmatize(t) for t in result]

    return result


# ---------------------------------------------------------------------------
# Custom backend resolvers — pure functions
# ---------------------------------------------------------------------------


def _resolve_custom_tokenizer(cfg: WordChunkerConfig) -> TokenizerProtocol:
    """Resolve the custom tokenizer from *cfg*.

    Parameters
    ----------
    cfg : WordChunkerConfig
        Active configuration.

    Returns
    -------
    TokenizerProtocol
        Ready tokenizer.

    Raises
    ------
    ValueError
        If no custom_tokenizer is set.
    TypeError
        If the provided object does not satisfy TokenizerProtocol.
    """
    obj = cfg.custom_tokenizer
    if obj is None:
        raise ValueError(
            "WordChunkerConfig.custom_tokenizer must be set when "
            "tokenizer=TokenizerBackend.CUSTOM."
        )
    if callable(obj) and not hasattr(obj, "tokenize"):
        return FunctionTokenizer(obj, name="custom_tokenizer")
    if not isinstance(obj, TokenizerProtocol):
        raise TypeError(
            f"custom_tokenizer must satisfy TokenizerProtocol "
            f"(have a .tokenize(text) method), got {type(obj).__name__!r}."
        )
    return obj


def _resolve_custom_stemmer(cfg: WordChunkerConfig) -> StemmerProtocol:
    """Resolve the custom stemmer from *cfg*.

    Parameters
    ----------
    cfg : WordChunkerConfig
        Active configuration.

    Returns
    -------
    StemmerProtocol
        Ready stemmer.

    Raises
    ------
    ValueError
        If no custom_stemmer is set.
    TypeError
        If the provided object does not satisfy StemmerProtocol.
    """
    obj = cfg.custom_stemmer
    if obj is None:
        raise ValueError(
            "WordChunkerConfig.custom_stemmer must be set when "
            "stemmer=StemmingBackend.CUSTOM."
        )
    if callable(obj) and not hasattr(obj, "stem"):
        return FunctionStemmer(obj, name="custom_stemmer")
    if not isinstance(obj, StemmerProtocol):
        raise TypeError(
            f"custom_stemmer must satisfy StemmerProtocol "
            f"(have a .stem(word) method), got {type(obj).__name__!r}."
        )
    return obj


def _resolve_custom_lemmatizer(cfg: WordChunkerConfig) -> LemmatizerProtocol:
    """Resolve the custom lemmatizer from *cfg*.

    Parameters
    ----------
    cfg : WordChunkerConfig
        Active configuration.

    Returns
    -------
    LemmatizerProtocol
        Ready lemmatizer.

    Raises
    ------
    ValueError
        If no custom_lemmatizer is set.
    TypeError
        If the provided object does not satisfy LemmatizerProtocol.
    """
    obj = cfg.custom_lemmatizer
    if obj is None:
        raise ValueError(
            "WordChunkerConfig.custom_lemmatizer must be set when "
            "lemmatizer=LemmatizationBackend.CUSTOM."
        )
    if callable(obj) and not hasattr(obj, "lemmatize"):
        return FunctionLemmatizer(obj, name="custom_lemmatizer")
    if not isinstance(obj, LemmatizerProtocol):
        raise TypeError(
            f"custom_lemmatizer must satisfy LemmatizerProtocol "
            f"(have a .lemmatize(word) method), got {type(obj).__name__!r}."
        )
    return obj


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


class WordChunker:
    """Process a document at word level, producing normalised token chunks.

    Each output :class:`~.._types.Chunk` contains:

    * ``text`` — space-joined normalised tokens (with optional n-grams).
    * ``metadata`` — token list, n-grams, token count, processing flags,
      optional Gensim BoW vector.

    Parameters
    ----------
    config : WordChunkerConfig, optional
        Processing configuration.
    gensim_dictionary : gensim.corpora.Dictionary, optional
        Pre-built Gensim dictionary.  When provided (and
        ``cfg.build_gensim_corpus`` is ``True``), each chunk's metadata
        includes a ``"bow"`` Gensim BoW vector.

    Examples
    --------
    >>> cfg = WordChunkerConfig(stemmer=StemmingBackend.PORTER)
    >>> chunker = WordChunker(cfg)
    >>> result = chunker.chunk("The quick brown foxes are jumping over lazy dogs.")
    >>> "token_count" in result.chunks[0].metadata
    True
    """

    def __init__(
        self,
        config: WordChunkerConfig | None = None,
        gensim_dictionary: Any | None = None,
    ) -> None:
        self._cfg = config if config is not None else WordChunkerConfig()
        self._gensim_dict = gensim_dictionary
        self._stopwords: frozenset[str] = frozenset()  # loaded lazily
        self._spacy_nlp: Any | None = None  # loaded lazily
        self._validate_config()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:  # noqa: PLR0912
        """Validate configuration at construction time.

        Raises
        ------
        ValueError
            On invalid configuration values.
        """
        if self._cfg.min_token_length < 0:
            raise ValueError(
                f"WordChunkerConfig.min_token_length must be >= 0, "
                f"got {self._cfg.min_token_length}."
            )
        if self._cfg.max_token_length is not None:  # noqa: SIM102
            if self._cfg.max_token_length < self._cfg.min_token_length:
                raise ValueError(
                    "WordChunkerConfig.max_token_length must be >= min_token_length."
                )
        min_n, max_n = self._cfg.ngram_range
        if min_n < 1 or max_n < min_n:
            raise ValueError(
                f"WordChunkerConfig.ngram_range must satisfy "
                f"1 <= min_n <= max_n, got ({min_n}, {max_n})."
            )
        if self._cfg.chunk_by not in ("document", "sentence"):
            raise ValueError(
                f"WordChunkerConfig.chunk_by must be 'document' or "
                f"'sentence', got {self._cfg.chunk_by!r}."
            )
        if (
            self._cfg.tokenizer == TokenizerBackend.SPACY
            or self._cfg.lemmatizer == LemmatizationBackend.SPACY
        ) and not self._cfg.spacy_model:
            raise ValueError(
                "WordChunkerConfig.spacy_model must be set when using "
                "SPACY tokenizer or lemmatizer."
            )
        if (
            self._cfg.tokenizer == TokenizerBackend.CUSTOM
            and self._cfg.custom_tokenizer is None
        ):
            raise ValueError(
                "WordChunkerConfig.custom_tokenizer must be set when "
                "tokenizer=TokenizerBackend.CUSTOM."
            )
        if (
            self._cfg.stemmer == StemmingBackend.CUSTOM
            and self._cfg.custom_stemmer is None
        ):
            raise ValueError(
                "WordChunkerConfig.custom_stemmer must be set when "
                "stemmer=StemmingBackend.CUSTOM."
            )
        if (
            self._cfg.lemmatizer == LemmatizationBackend.CUSTOM
            and self._cfg.custom_lemmatizer is None
        ):
            raise ValueError(
                "WordChunkerConfig.custom_lemmatizer must be set when "
                "lemmatizer=LemmatizationBackend.CUSTOM."
            )
        # Guard SNOWBALL against unsupported languages (e.g. Arabic, CJK).
        if self._cfg.stemmer == StemmingBackend.SNOWBALL:
            langs = coerce_language(self._cfg.nltk_language, default="english")
            for lang in langs:
                if lang not in _SNOWBALL_SUPPORTED_LANGUAGES:
                    raise ValueError(
                        f"StemmingBackend.SNOWBALL does not support language "
                        f"{lang!r}. "
                        f"Supported: {sorted(_SNOWBALL_SUPPORTED_LANGUAGES)}. "
                        f"Use StemmingBackend.CUSTOM to supply your own stemmer."
                    )
        if self._cfg.build_gensim_corpus and self._gensim_dict is None:
            logger.warning(
                "WordChunkerConfig.build_gensim_corpus=True but no "
                "gensim_dictionary was provided. BoW vectors will be absent."
            )

    # ------------------------------------------------------------------
    # Lazy initialisation helpers
    # ------------------------------------------------------------------

    def _ensure_stopwords(self) -> None:
        """Load stopwords on first use (union across all specified languages)."""
        if not self._stopwords:
            base = _load_stopwords(self._cfg.stopwords, self._cfg.nltk_language)
            if self._cfg.custom_stopwords:
                base = base | frozenset(self._cfg.custom_stopwords)
            self._stopwords = base

    def _ensure_spacy(self) -> None:
        """Load the spaCy model on first use."""
        if self._spacy_nlp is None and self._cfg.spacy_model:
            try:
                import spacy  # type: ignore[import-untyped]  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "SPACY backend requires 'spacy'. Install with: pip install spacy"
                ) from exc
            self._spacy_nlp = spacy.load(self._cfg.spacy_model)

    # ------------------------------------------------------------------
    # Internal tokenize + process
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Run the configured tokenizer on *text*.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list[str]
            Raw tokens.
        """
        if self._cfg.tokenizer == TokenizerBackend.SIMPLE:
            return _tokenize_simple(text)
        if self._cfg.tokenizer == TokenizerBackend.NLTK:
            return _tokenize_nltk(text)
        if self._cfg.tokenizer == TokenizerBackend.SPACY:
            assert self._cfg.spacy_model is not None  # noqa: S101
            return _tokenize_spacy(text, self._cfg.spacy_model)
        if self._cfg.tokenizer == TokenizerBackend.CUSTOM:
            tok = _resolve_custom_tokenizer(self._cfg)
            return tok.tokenize(text)
        raise ValueError(
            f"Unsupported tokenizer: {self._cfg.tokenizer!r}."
        )  # pragma: no cover

    def _process_segment(self, text: str, doc_id: str | None, seg_index: int) -> Chunk:
        """Process a single text segment into a :class:`~.._types.Chunk`.

        Parameters
        ----------
        text : str
            Segment text (one document or one sentence).
        doc_id : str, optional
            Document identifier.
        seg_index : int
            Index of this segment within the document.

        Returns
        -------
        Chunk
            Processed word-level chunk.
        """
        self._ensure_stopwords()

        spacy_doc: Any | None = None
        if self._cfg.lemmatizer == LemmatizationBackend.SPACY:
            self._ensure_spacy()
            if self._spacy_nlp is not None:
                spacy_doc = self._spacy_nlp(text)

        raw_tokens = self._tokenize(text)
        processed = _process_tokens(
            raw_tokens, self._cfg, self._stopwords, spacy_doc=spacy_doc
        )

        # N-gram extraction.
        min_n, max_n = self._cfg.ngram_range
        all_terms: list[str] = list(processed)
        ngrams: list[str] = []
        for n in range(max(2, min_n), max_n + 1):
            ng = _extract_ngrams(processed, n)
            ngrams.extend(ng)
            all_terms.extend(ng)

        chunk_text = " ".join(all_terms)

        meta: dict[str, Any] = {
            "chunk_index": seg_index,
            "tokens": processed,
            "token_count": len(processed),
            "ngrams": ngrams,
            "ngram_range": list(self._cfg.ngram_range),
            "tokenizer": self._cfg.tokenizer.value,
            "stemmer": self._cfg.stemmer.value,
            "lemmatizer": self._cfg.lemmatizer.value,
        }
        if doc_id is not None:
            meta["doc_id"] = doc_id

        if self._cfg.build_gensim_corpus and self._gensim_dict is not None:
            meta["bow"] = _to_gensim_bow(processed, self._gensim_dict)

        return Chunk(
            text=chunk_text,
            start_char=0,  # word-level chunks do not track character offsets
            end_char=len(text),
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        doc_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkResult:
        """Process *text* into word-level chunks.

        Parameters
        ----------
        text : str
            Raw document text.
        doc_id : str, optional
            Document identifier stored in each chunk's metadata.
        extra_metadata : dict[str, Any], optional
            Additional key/value pairs merged into result metadata.

        Returns
        -------
        ChunkResult
            Chunks and aggregate metadata.

        Raises
        ------
        TypeError
            If *text* is not a ``str``.
        ValueError
            If *text* is empty or whitespace-only.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}.")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only.")

        if self._cfg.chunk_by == "sentence":
            # Split on sentence boundaries for sentence-level word chunks.
            from ._sentence import (  # noqa: PLC0415
                SentenceChunker,
                SentenceChunkerConfig,
            )

            sent_chunker = SentenceChunker(
                SentenceChunkerConfig(min_length=1, include_offsets=False)
            )
            sent_result = sent_chunker.chunk(text, doc_id=doc_id)
            segments = [c.text for c in sent_result.chunks]
        else:
            segments = [text]

        chunks: list[Chunk] = [
            self._process_segment(seg, doc_id, idx) for idx, seg in enumerate(segments)
        ]

        result_meta: dict[str, Any] = {
            "chunker": "word",
            "tokenizer": self._cfg.tokenizer.value,
            "stemmer": self._cfg.stemmer.value,
            "lemmatizer": self._cfg.lemmatizer.value,
            "stopwords": self._cfg.stopwords.value,
            "ngram_range": list(self._cfg.ngram_range),
            "total_chunks": len(chunks),
        }
        if doc_id is not None:
            result_meta["doc_id"] = doc_id
        if extra_metadata:
            result_meta.update(extra_metadata)

        return ChunkResult(chunks=chunks, metadata=result_meta)

    def chunk_batch(
        self,
        texts: list[str],
        doc_ids: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Process a list of documents into word-level chunks.

        Parameters
        ----------
        texts : list[str]
            Input documents.
        doc_ids : list[str], optional
            Parallel document identifiers.
        extra_metadata : dict[str, Any], optional
            Shared metadata merged into every result.

        Returns
        -------
        list[ChunkResult]
            One result per document.

        Raises
        ------
        TypeError
            If *texts* is not a list.
        ValueError
            If *doc_ids* length mismatches *texts*.
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list, got {type(texts).__name__!r}.")
        if doc_ids is not None and len(doc_ids) != len(texts):
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) must equal "
                f"texts length ({len(texts)})."
            )
        return [
            self.chunk(
                t,
                doc_id=doc_ids[i] if doc_ids else None,
                extra_metadata=extra_metadata,
            )
            for i, t in enumerate(texts)
        ]

    # ------------------------------------------------------------------
    # Corpus-level utilities
    # ------------------------------------------------------------------

    @staticmethod
    def build_gensim_dictionary(
        token_lists: list[list[str]],
        no_below: int = 2,
        no_above: float = 0.9,
        keep_n: int | None = None,
    ) -> Any:
        """Build a :class:`gensim.corpora.Dictionary` from token lists.

        Parameters
        ----------
        token_lists : list[list[str]]
            Processed token lists (one per document).
        no_below : int
            Filter tokens appearing in fewer than *no_below* documents.
        no_above : float
            Filter tokens appearing in more than *no_above* fraction of
            documents (0.0-1.0).
        keep_n : int, optional
            Retain only the top *keep_n* most frequent tokens after filtering.

        Returns
        -------
        gensim.corpora.Dictionary
            Built and filtered dictionary.

        Raises
        ------
        ImportError
            If Gensim is not installed.
        ValueError
            If *token_lists* is empty.
        """
        if not token_lists:
            raise ValueError("token_lists must not be empty.")
        try:
            from gensim.corpora import (  # type: ignore[import-untyped]  # noqa: PLC0415
                Dictionary,
            )
        except ImportError as exc:
            raise ImportError(
                "build_gensim_dictionary requires 'gensim'. "
                "Install with: pip install gensim"
            ) from exc

        dictionary = Dictionary(token_lists)
        dictionary.filter_extremes(
            no_below=no_below,
            no_above=no_above,
            keep_n=keep_n,
        )
        return dictionary

    @staticmethod
    def vocabulary_stats(token_lists: list[list[str]]) -> dict[str, Any]:
        """Compute vocabulary statistics over a corpus.

        Parameters
        ----------
        token_lists : list[list[str]]
            Processed token lists, one per document.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            ``vocab_size``, ``total_tokens``, ``unique_tokens``,
            ``avg_tokens_per_doc``, ``top_20_tokens``.

        Raises
        ------
        ValueError
            If *token_lists* is empty.
        """
        if not token_lists:
            raise ValueError("token_lists must not be empty.")

        from collections import Counter  # noqa: PLC0415

        all_tokens: list[str] = [t for tl in token_lists for t in tl]
        counter = Counter(all_tokens)
        return {
            "vocab_size": len(counter),
            "total_tokens": len(all_tokens),
            "unique_tokens": len(counter),
            "avg_tokens_per_doc": len(all_tokens) / len(token_lists),
            "top_20_tokens": counter.most_common(20),
        }
