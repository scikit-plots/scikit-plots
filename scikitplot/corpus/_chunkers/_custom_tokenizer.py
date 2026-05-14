# scikitplot/corpus/_chunkers/_custom_tokenizer.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Custom tokenizer / sentence-splitter protocol, registry, and script detection.

This module is the single extension point for plugging any third-party
NLP library into :class:`~._word.WordChunker` and
:class:`~._sentence.SentenceChunker` without modifying core code.

Supported extension points:

* **Tokenization** — ``WordChunkerConfig(tokenizer=TokenizerBackend.CUSTOM,
  custom_tokenizer=<TokenizerProtocol>)``.  Works with MeCab (Japanese),
  jieba (Chinese), camel-tools (Arabic/Ottoman), stanza (100+ languages),
  HuggingFace fast tokenizers, or any callable returning ``list[str]``.

* **Sentence splitting** — ``SentenceChunkerConfig(backend=SentenceBackend.CUSTOM,
  custom_splitter=<SentenceSplitterProtocol>)``.  Works with PySBD, Stanza
  sentence segmenters, CAMeL Tools, or any callable returning ``list[str]``.

* **Stemming** — ``WordChunkerConfig(stemmer=StemmingBackend.CUSTOM,
  custom_stemmer=<StemmerProtocol>)``.  Works with any callable
  ``Callable[[str], str]``.

* **Lemmatization** — ``WordChunkerConfig(lemmatizer=LemmatizationBackend.CUSTOM,
  custom_lemmatizer=<LemmatizerProtocol>)``.

Script detection:

:func:`detect_script` returns a :class:`ScriptType` member describing the
dominant Unicode script in a text sample.  Chunkers use this to auto-configure
sentence-boundary patterns and punctuation stripping for CJK / RTL / ancient
scripts without the caller needing to pass explicit hints.

Python compatibility:

Python 3.8-3.15.  No external dependencies.
``from __future__ import annotations`` for all annotations.

Examples
--------
Wrap a plain callable as a tokenizer:

>>> tok = FunctionTokenizer(str.split)
>>> tok.tokenize("hello world")
['hello', 'world']

Register a named tokenizer for reuse:

>>> register_tokenizer("whitespace", FunctionTokenizer(str.split))
>>> get_tokenizer("whitespace").tokenize("foo bar")
['foo', 'bar']

Detect script:

>>> detect_script("مرحبا بالعالم")
<ScriptType.ARABIC: 'arabic'>
>>> detect_script("Hello World")
<ScriptType.LATIN: 'latin'>
>>> detect_script("こんにちは")
<ScriptType.CJK: 'cjk'>
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import threading
import unicodedata  # noqa: F401

# Python 3.7 shim (not needed at 3.8+ but harmless)
# Only imports when type checking
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    Generator,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from typing_extensions import Self  # noqa: F401

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [  # noqa: RUF022
    # Protocols
    "TokenizerProtocol",
    "SentenceSplitterProtocol",
    "StemmerProtocol",
    "LemmatizerProtocol",
    # Callable wrappers
    "FunctionTokenizer",
    "FunctionSentenceSplitter",
    "FunctionStemmer",
    "FunctionLemmatizer",
    # Registry
    "CustomTokenizerRegistry",
    "register_tokenizer",
    "get_tokenizer",
    "register_sentence_splitter",
    "get_sentence_splitter",
    "register_stemmer",
    "get_stemmer",
    "register_lemmatizer",
    "get_lemmatizer",
    # Script detection
    "ScriptType",
    "detect_script",
    "is_cjk_char",
    "is_rtl_char",
    "split_cjk_chars",
    "_split_cjk_chars_legacy",
    "MULTI_SCRIPT_SENTENCE_RE_PATTERN",
    # Layer 1 — Script segmentation
    "ScriptSpan",
    "ScriptSegmenter",
]


# ===========================================================================
# Section 1 — Protocols (structural subtyping for type checkers)
# ===========================================================================


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Structural protocol for word tokenizers.

    Any object with a ``tokenize(text: str) -> list[str]`` method satisfies
    this protocol, regardless of inheritance.  This includes MeCab wrappers,
    jieba objects, camel-tools tokenizers, Stanza pipelines, HuggingFace
    fast tokenizers, and plain callable wrappers via :class:`FunctionTokenizer`.

    Parameters
    ----------
    (none at construction time — protocols define the *call* interface)

    Examples
    --------
    >>> class MyTok:
    ...     def tokenize(self, text: str) -> list:
    ...         return text.split()
    >>> isinstance(MyTok(), TokenizerProtocol)
    True
    """  # noqa: D205

    def tokenize(self, text: str) -> list[str]:
        """Tokenize *text* into a list of token strings.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        list[str]
            Token list.  Empty list for empty input.
        """
        ...  # pragma: no cover


@runtime_checkable
class SentenceSplitterProtocol(Protocol):
    """Structural protocol for sentence segmenters.

    Any object with a ``split(text: str) -> list[str]`` method satisfies
    this protocol.

    Examples
    --------
    >>> class MySplitter:
    ...     def split(self, text: str) -> list:
    ...         return text.split(". ")
    >>> isinstance(MySplitter(), SentenceSplitterProtocol)
    True
    """  # noqa: D205

    def split(self, text: str) -> list[str]:
        """Split *text* into sentences.

        Parameters
        ----------
        text : str
            Raw document text.

        Returns
        -------
        list[str]
            Sentence strings.
        """
        ...  # pragma: no cover


@runtime_checkable
class StemmerProtocol(Protocol):
    """Structural protocol for word stemmers.

    Examples
    --------
    >>> class MyStemmer:
    ...     def stem(self, word: str) -> str:
    ...         return word.rstrip("ing")
    >>> isinstance(MyStemmer(), StemmerProtocol)
    True
    """  # noqa: D205

    def stem(self, word: str) -> str:
        """Return the stem of *word*.

        Parameters
        ----------
        word : str
            Input word.

        Returns
        -------
        str
            Stemmed form.
        """
        ...  # pragma: no cover


@runtime_checkable
class LemmatizerProtocol(Protocol):
    """Structural protocol for word lemmatizers.

    The ``pos`` parameter is optional context (part-of-speech tag).
    Implementations that do not use ``pos`` can ignore it.

    Examples
    --------
    >>> class MyLemma:
    ...     def lemmatize(self, word: str, pos: str = None) -> str:
    ...         return word.lower()
    >>> isinstance(MyLemma(), LemmatizerProtocol)
    True
    """  # noqa: D205

    def lemmatize(self, word: str, pos: str | None = None) -> str:
        """Return the lemma of *word*.

        Parameters
        ----------
        word : str
            Input word.
        pos : str, optional
            Part-of-speech hint (e.g. ``"n"`` for noun, ``"v"`` for verb).

        Returns
        -------
        str
            Lemma form.
        """
        ...  # pragma: no cover


# ===========================================================================
# Section 2 — Callable wrappers (bridge any Callable to a Protocol)
# ===========================================================================


class FunctionTokenizer:
    r"""Wrap any ``Callable[[str], list[str]]`` as a :class:`TokenizerProtocol`.

    Parameters
    ----------
    fn : Callable[[str], list[str]]
        Tokenization function.  Must accept a single ``str`` argument
        and return a ``list[str]``.
    name : str, optional
        Human-readable name for logging and ``repr``.

    Notes
    -----
    **User note:** Use this to plug in any tokenization library:

    .. code-block:: python

        import MeCab

        tagger = MeCab.Tagger("-Owakati")
        tok = FunctionTokenizer(lambda text: tagger.parse(text).strip().split())

        import jieba

        tok = FunctionTokenizer(lambda text: list(jieba.cut(text)))

    **Developer note:** The wrapper stores only the callable; no model
    loading happens at construction time.

    Examples
    --------
    >>> tok = FunctionTokenizer(str.split)
    >>> tok.tokenize("hello world")
    ['hello', 'world']
    >>> tok = FunctionTokenizer(lambda t: list(t), name="char_splitter")
    >>> tok.tokenize("abc")
    ['a', 'b', 'c']
    """  # noqa: D205

    def __init__(
        self,
        fn: Callable[[str], list[str]],
        name: str = "custom",
    ) -> None:
        if not callable(fn):
            raise TypeError(
                f"FunctionTokenizer: fn must be callable, got {type(fn).__name__!r}."
            )
        self._fn = fn
        self._name = name

    def tokenize(self, text: str) -> list[str]:
        """Tokenize *text* using the wrapped callable.

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
        TypeError
            If the wrapped callable does not return a list.
        """
        if not text:
            return []
        result = self._fn(text)
        if not isinstance(result, list):
            raise TypeError(
                f"FunctionTokenizer({self._name!r}): callable must return list[str], "
                f"got {type(result).__name__!r}."
            )
        return result

    def __repr__(self) -> str:
        return f"FunctionTokenizer(name={self._name!r})"


class FunctionSentenceSplitter:
    r"""Wrap any ``Callable[[str], list[str]]`` as a :class:`SentenceSplitterProtocol`.

    Parameters
    ----------
    fn : Callable[[str], list[str]]
        Sentence-splitting function.
    name : str, optional
        Human-readable name for logging and ``repr``.

    Examples
    --------
    >>> sp = FunctionSentenceSplitter(lambda t: t.split(". "))
    >>> sp.split("Hello. World.")
    ['Hello', 'World.']
    """  # noqa: D205

    def __init__(
        self,
        fn: Callable[[str], list[str]],
        name: str = "custom",
    ) -> None:
        if not callable(fn):
            raise TypeError(
                f"FunctionSentenceSplitter: fn must be callable, "
                f"got {type(fn).__name__!r}."
            )
        self._fn = fn
        self._name = name

    def split(self, text: str) -> list[str]:
        """Split *text* into sentences.

        Parameters
        ----------
        text : str
            Input document text.

        Returns
        -------
        list[str]
            Sentence strings.

        Raises
        ------
        TypeError
            If the wrapped callable does not return a list.
        """
        if not text:
            return []
        result = self._fn(text)
        if not isinstance(result, list):
            raise TypeError(
                f"FunctionSentenceSplitter({self._name!r}): callable must return "
                f"list[str], got {type(result).__name__!r}."
            )
        return result

    def __repr__(self) -> str:
        return f"FunctionSentenceSplitter(name={self._name!r})"


class FunctionStemmer:
    r"""Wrap any ``Callable[[str], str]`` as a :class:`StemmerProtocol`.

    Parameters
    ----------
    fn : Callable[[str], str]
        Stemming function; takes a single word, returns its stem.
    name : str, optional
        Human-readable name.

    Examples
    --------
    >>> st = FunctionStemmer(lambda w: w[:4] if len(w) > 4 else w)
    >>> st.stem("running")
    'runn'
    """  # noqa: D205

    def __init__(
        self,
        fn: Callable[[str], str],
        name: str = "custom",
    ) -> None:
        if not callable(fn):
            raise TypeError(
                f"FunctionStemmer: fn must be callable, got {type(fn).__name__!r}."
            )
        self._fn = fn
        self._name = name

    def stem(self, word: str) -> str:
        """Stem *word*.

        Parameters
        ----------
        word : str
            Input word.

        Returns
        -------
        str
            Stemmed form.
        """
        return self._fn(word)

    def __repr__(self) -> str:
        return f"FunctionStemmer(name={self._name!r})"


class FunctionLemmatizer:
    r"""Wrap any ``Callable[[str, Optional[str]], str]`` as a :class:`LemmatizerProtocol`.

    Parameters
    ----------
    fn : Callable[[str], str] or Callable[[str, Optional[str]], str]
        Lemmatization function.  May accept an optional ``pos`` argument.
        If the function accepts only one argument the ``pos`` parameter is
        silently dropped.
    name : str, optional
        Human-readable name.

    Examples
    --------
    >>> lm = FunctionLemmatizer(lambda w: w.lower(), pos=None)
    >>> lm.lemmatize("Running", pos="v")
    'running'
    """  # noqa: D205

    def __init__(
        self,
        fn: Callable[..., str],
        name: str = "custom",
    ) -> None:
        if not callable(fn):
            raise TypeError(
                f"FunctionLemmatizer: fn must be callable, got {type(fn).__name__!r}."
            )
        self._fn = fn
        self._name = name
        # Detect whether the callable accepts a second (pos) argument.
        import inspect  # noqa: PLC0415

        try:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            self._accepts_pos = len(params) >= 2  # noqa: PLR2004
        except (ValueError, TypeError):
            self._accepts_pos = False

    def lemmatize(self, word: str, pos: str | None = None) -> str:
        """Lemmatize *word* with optional POS hint.

        Parameters
        ----------
        word : str
            Input word.
        pos : str, optional
            Part-of-speech tag.

        Returns
        -------
        str
            Lemma form.
        """
        if self._accepts_pos:
            return self._fn(word, pos)
        return self._fn(word)

    def __repr__(self) -> str:
        return f"FunctionLemmatizer(name={self._name!r})"


# ===========================================================================
# Section 3 — Module-level registries
# ===========================================================================


class CustomTokenizerRegistry:
    """Thread-safe module-level registry for named custom components.

    Each registry holds a ``dict[str, Protocol]`` accessible via module-level
    helpers (:func:`register_tokenizer`, :func:`get_tokenizer`, etc.).

    All ``register``, ``get``, ``names``, and ``__contains__`` operations
    acquire an internal :class:`threading.RLock` to guarantee safety when
    registering from worker threads (e.g. ``ThreadPoolExecutor`` batch jobs).

    Notes
    -----
    **User note:** Register all custom components at application startup, before
    spawning worker threads, to avoid any lock contention in hot paths.

    **Developer note:** :class:`threading.RLock` (reentrant) is used instead of
    :class:`threading.Lock` so that the same thread can call ``register`` from
    within a ``get`` callback without deadlocking.  This is safe on CPython,
    PyPy, and GraalPy.

    Raises
    ------
    KeyError
        :meth:`get` raises ``KeyError`` when the name is not registered.
    """  # noqa: D205

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._store: dict[str, Any] = {}
        self._lock: threading.RLock = threading.RLock()

    def register(self, name: str, instance: Any) -> None:
        """Register *instance* under *name*.

        Parameters
        ----------
        name : str
            Registry key.  Must be a non-empty string.
        instance : object
            The component instance to register.

        Raises
        ------
        TypeError
            If *name* is not a str.
        ValueError
            If *name* is empty.
        """
        if not isinstance(name, str):
            raise TypeError(
                f"CustomTokenizerRegistry({self._kind!r}): "
                f"name must be str, got {type(name).__name__!r}."
            )
        if not name:
            raise ValueError(
                f"CustomTokenizerRegistry({self._kind!r}): name must be non-empty."
            )
        with self._lock:
            if name in self._store:
                logger.debug(
                    "CustomTokenizerRegistry(%r): overwriting existing entry %r.",
                    self._kind,
                    name,
                )
            self._store[name] = instance
            logger.debug(
                "CustomTokenizerRegistry(%r): registered %r → %r.",
                self._kind,
                name,
                type(instance).__name__,
            )

    def get(self, name: str) -> Any:
        """Retrieve the component registered under *name*.

        Parameters
        ----------
        name : str
            Registry key.

        Returns
        -------
        object
            The registered component.

        Raises
        ------
        KeyError
            If *name* has not been registered.
        """
        with self._lock:
            if name not in self._store:
                available = list(self._store)
                raise KeyError(
                    f"CustomTokenizerRegistry({self._kind!r}): {name!r} not registered. "
                    f"Available: {available}."
                )
            return self._store[name]

    def names(self) -> list[str]:
        """Return all registered names.

        Returns
        -------
        list[str]
            Sorted list of registered keys.
        """
        with self._lock:
            return sorted(self._store)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._store

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"CustomTokenizerRegistry(kind={self._kind!r}, entries={self.names()!r})"


# Module-level singletons — one per component type.
_TOKENIZER_REGISTRY: Final[CustomTokenizerRegistry] = CustomTokenizerRegistry(
    "tokenizer"
)
_SPLITTER_REGISTRY: Final[CustomTokenizerRegistry] = CustomTokenizerRegistry(
    "sentence_splitter"
)
_STEMMER_REGISTRY: Final[CustomTokenizerRegistry] = CustomTokenizerRegistry("stemmer")
_LEMMATIZER_REGISTRY: Final[CustomTokenizerRegistry] = CustomTokenizerRegistry(
    "lemmatizer"
)


# ---------------------------------------------------------------------------
# Public helper functions — thin wrappers around the singletons
# ---------------------------------------------------------------------------


def register_tokenizer(name: str, tokenizer: TokenizerProtocol) -> None:
    """Register a named :class:`TokenizerProtocol` implementation.

    Parameters
    ----------
    name : str
        Registry key.
    tokenizer : TokenizerProtocol
        Tokenizer instance.

    Examples
    --------
    >>> register_tokenizer("jieba", FunctionTokenizer(lambda t: t.split()))
    >>> get_tokenizer("jieba").tokenize("hello world")
    ['hello', 'world']
    """
    _TOKENIZER_REGISTRY.register(name, tokenizer)


def get_tokenizer(name: str) -> TokenizerProtocol:
    """Retrieve a registered tokenizer by name.

    Parameters
    ----------
    name : str
        Registry key.

    Returns
    -------
    TokenizerProtocol
        The registered tokenizer.

    Raises
    ------
    KeyError
        If *name* has not been registered.
    """
    return _TOKENIZER_REGISTRY.get(name)


def register_sentence_splitter(name: str, splitter: SentenceSplitterProtocol) -> None:
    """Register a named :class:`SentenceSplitterProtocol` implementation.

    Parameters
    ----------
    name : str
        Registry key.
    splitter : SentenceSplitterProtocol
        Sentence splitter instance.

    Examples
    --------
    >>> register_sentence_splitter(
    ...     "pysbd_en", FunctionSentenceSplitter(lambda t: t.split(". "))
    ... )
    """
    _SPLITTER_REGISTRY.register(name, splitter)


def get_sentence_splitter(name: str) -> SentenceSplitterProtocol:
    """Retrieve a registered sentence splitter by name.

    Parameters
    ----------
    name : str
        Registry key.

    Returns
    -------
    SentenceSplitterProtocol
        The registered splitter.

    Raises
    ------
    KeyError
        If *name* has not been registered.
    """
    return _SPLITTER_REGISTRY.get(name)


def register_stemmer(name: str, stemmer: StemmerProtocol) -> None:
    """Register a named :class:`StemmerProtocol` implementation.

    Parameters
    ----------
    name : str
        Registry key.
    stemmer : StemmerProtocol
        Stemmer instance.
    """
    _STEMMER_REGISTRY.register(name, stemmer)


def get_stemmer(name: str) -> StemmerProtocol:
    """Retrieve a registered stemmer by name.

    Parameters
    ----------
    name : str
        Registry key.

    Returns
    -------
    StemmerProtocol
        The registered stemmer.

    Raises
    ------
    KeyError
        If *name* has not been registered.
    """
    return _STEMMER_REGISTRY.get(name)


def register_lemmatizer(name: str, lemmatizer: LemmatizerProtocol) -> None:
    """Register a named :class:`LemmatizerProtocol` implementation.

    Parameters
    ----------
    name : str
        Registry key.
    lemmatizer : LemmatizerProtocol
        Lemmatizer instance.
    """
    _LEMMATIZER_REGISTRY.register(name, lemmatizer)


def get_lemmatizer(name: str) -> LemmatizerProtocol:
    """Retrieve a registered lemmatizer by name.

    Parameters
    ----------
    name : str
        Registry key.

    Returns
    -------
    LemmatizerProtocol
        The registered lemmatizer.

    Raises
    ------
    KeyError
        If *name* has not been registered.
    """
    return _LEMMATIZER_REGISTRY.get(name)


# ===========================================================================
# Section 4 — Script detection
# ===========================================================================


from enum import Enum  # noqa: E402 (after __future__ guard)


class ScriptType(str, Enum):
    """Dominant Unicode script detected in a text sample.

    Attributes
    ----------
    LATIN
        Latin script (English, French, German, Spanish, Portuguese,
        Romanian, Turkish, Vietnamese, etc.) — left-to-right.
    CJK
        Chinese / Japanese / Korean ideographs and kana — traditionally
        top-to-bottom, rendered left-to-right in digital contexts.
    ARABIC
        Arabic, Persian (Farsi), Ottoman Turkish, Urdu — right-to-left.
    HEBREW
        Hebrew, Yiddish — right-to-left.
    DEVANAGARI
        Hindi, Sanskrit, Marathi, Nepali — left-to-right.
    GREEK
        Modern and ancient Greek — left-to-right.
    CYRILLIC
        Russian, Bulgarian, Serbian, Ukrainian, etc. — left-to-right.
    ETHIOPIC
        Amharic, Tigrinya — left-to-right.
    GEORGIAN
        Georgian — left-to-right.
    EGYPTIAN
        Coptic (no Unicode block for hieroglyphs with full coverage yet;
        Coptic block used as proxy for Coptic-script ancient Egyptian).
    THAI
        Thai — left-to-right.
    SOUTHEAST_ASIAN
        Lao, Khmer, Myanmar, Burmese — covers mainland Southeast Asian
        scripts that are not Thai.
    SOUTH_ASIAN
        Dravidian scripts: Tamil, Telugu, Kannada, Malayalam, Sinhala.
        Distinct from Devanagari which covers North Indian languages.
    ARMENIAN
        Armenian — left-to-right.
    TIBETAN
        Tibetan — left-to-right.
    MIXED
        Multiple scripts present in roughly equal proportions.
    UNKNOWN
        No script characters detected (empty, purely numeric, symbols).
    """

    LATIN = "latin"
    # ------------------------------------------------------------------
    # CJK — deprecated alias; kept for backward compatibility.
    # New Layer-2 dispatch MUST use HAN / HIRAGANA / KATAKANA / HANGUL.
    # ------------------------------------------------------------------
    CJK = "cjk"
    # ------------------------------------------------------------------
    # Granular East Asian scripts (Bug 2 fix — split from CJK bucket)
    # ------------------------------------------------------------------
    HAN = "han"
    HIRAGANA = "hiragana"
    KATAKANA = "katakana"
    HANGUL = "hangul"
    # ------------------------------------------------------------------
    # RTL scripts
    # ------------------------------------------------------------------
    ARABIC = "arabic"
    HEBREW = "hebrew"
    # ------------------------------------------------------------------
    # South and South-East Asian
    # ------------------------------------------------------------------
    DEVANAGARI = "devanagari"
    THAI = "thai"
    SOUTHEAST_ASIAN = "southeast_asian"
    MYANMAR = "myanmar"
    KHMER = "khmer"
    SOUTH_ASIAN = "south_asian"
    TIBETAN = "tibetan"
    # ------------------------------------------------------------------
    # European / Caucasian
    # ------------------------------------------------------------------
    GREEK = "greek"
    CYRILLIC = "cyrillic"
    ARMENIAN = "armenian"
    GEORGIAN = "georgian"
    # ------------------------------------------------------------------
    # African
    # ------------------------------------------------------------------
    ETHIOPIC = "ethiopic"
    # ------------------------------------------------------------------
    # Ancient / historic
    # ------------------------------------------------------------------
    EGYPTIAN = "egyptian"
    EGYPTIAN_HIEROGLYPHS = "egyptian_hieroglyphs"
    # ------------------------------------------------------------------
    # Central Asian
    # ------------------------------------------------------------------
    MONGOLIAN = "mongolian"
    # ------------------------------------------------------------------
    # Symbol / emoji classes (Bug 1 fix)
    # ------------------------------------------------------------------
    EMOJI = "emoji"
    SYMBOLIC = "symbolic"
    # ------------------------------------------------------------------
    # Meta values
    # ------------------------------------------------------------------
    MIXED = "mixed"
    UNKNOWN = "unknown"


def detect_script(  # noqa: PLR0912
    text: str,
    *,
    sample_size: int = 500,
    majority_threshold: float = 0.55,
) -> ScriptType:
    r"""Detect the dominant Unicode script in *text*.

    Samples up to *sample_size* characters for efficiency on long documents.
    Returns :attr:`ScriptType.MIXED` when no single script exceeds
    *majority_threshold* of all script characters found.

    Parameters
    ----------
    text : str
        Input text to analyse.  Any length — only the first *sample_size*
        characters are examined.
    sample_size : int, optional
        Maximum number of characters to inspect.  Default 500.
    majority_threshold : float, optional
        Fraction of script chars a single script must reach to be declared
        dominant.  Default 0.55 (55 %).

    Returns
    -------
    ScriptType
        Detected dominant script.

    Notes
    -----
    **User note:** The detection is based on Unicode code-point ranges and
    is heuristic — it does not use a language model.  For ambiguous texts
    (transliterated Arabic in Latin script, mixed-script social media posts)
    the result may be :attr:`ScriptType.MIXED`.  Pass an explicit
    ``script_hint`` to :class:`~._sentence.SentenceChunkerConfig` to
    override detection.

    Supported script families: Latin, CJK (Chinese/Japanese/Korean/Hangul),
    Arabic (including Persian/Ottoman/Urdu), Hebrew, Devanagari
    (Hindi/Sanskrit/Nepali), Greek, Cyrillic, Ethiopic, Georgian, Coptic
    (Egyptian proxy), Thai, Southeast Asian (Lao/Myanmar/Khmer), South Asian
    Dravidian (Tamil/Telugu/Kannada/Malayalam/Sinhala), Armenian, Tibetan.

    **Developer note:** Unicode categories ``unicodedata.category(c)``
    are *not* used here because they do not map cleanly to script families.
    Code-point ranges from the Unicode Standard are used instead.

    Examples
    --------
    >>> detect_script("Hello world")
    <ScriptType.LATIN: 'latin'>
    >>> detect_script("مرحبا بالعالم")
    <ScriptType.ARABIC: 'arabic'>
    >>> detect_script("こんにちは世界")
    <ScriptType.HIRAGANA: 'hiragana'>
    >>> detect_script("Ἡ γλῶσσα")
    <ScriptType.GREEK: 'greek'>
    >>> detect_script("12345 !@#$%")
    <ScriptType.UNKNOWN: 'unknown'>
    >>> detect_script("😀🎉")
    <ScriptType.EMOJI: 'emoji'>
    >>> detect_script("你好世界")
    <ScriptType.HAN: 'han'>
    >>> detect_script("안녕하세요")
    <ScriptType.HANGUL: 'hangul'>
    """
    if not text:
        return ScriptType.UNKNOWN

    sample = text[:sample_size]
    counts: dict[str, int] = {
        "latin": 0,
        # Granular East Asian (Bug 2 — CJK bucket split into four)
        "han": 0,
        "hiragana": 0,
        "katakana": 0,
        "hangul": 0,
        "arabic": 0,
        "hebrew": 0,
        "devanagari": 0,
        "greek": 0,
        "cyrillic": 0,
        "ethiopic": 0,
        "georgian": 0,
        "egyptian": 0,
        "egyptian_hieroglyphs": 0,
        "thai": 0,
        "southeast_asian": 0,
        "myanmar": 0,
        "khmer": 0,
        "south_asian": 0,
        "armenian": 0,
        "tibetan": 0,
        "mongolian": 0,
        # Symbol classes (Bug 1 — emoji-only text was UNKNOWN)
        "emoji": 0,
        "symbolic": 0,
    }

    for ch in sample:
        cp = ord(ch)

        # ── Han (Chinese logographs) ──────────────────────────────────
        if (
            0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs  # noqa: PLR2004
            or 0x3400 <= cp <= 0x4DBF  # Extension A  # noqa: PLR2004
            or 0x20000 <= cp <= 0x2A6DF  # Extension B  # noqa: PLR2004
            or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs  # noqa: PLR2004
            or 0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation  # noqa: PLR2004
        ):
            counts["han"] += 1

        # ── Hiragana ──────────────────────────────────────────────────
        elif 0x3040 <= cp <= 0x309F:  # noqa: PLR2004
            counts["hiragana"] += 1

        # ── Katakana (including half-width forms) ─────────────────────
        elif (
            0x30A0 <= cp <= 0x30FF  # noqa: PLR2004
            or 0xFF65 <= cp <= 0xFF9F  # Half-width Katakana  # noqa: PLR2004
        ):
            counts["katakana"] += 1

        # ── Hangul (syllables + Jamo + Extended A/B + Halfwidth) ──────
        elif (
            0xAC00 <= cp <= 0xD7AF  # Hangul Syllables  # noqa: PLR2004
            or 0x1100 <= cp <= 0x11FF  # Hangul Jamo  # noqa: PLR2004
            or 0xA960 <= cp <= 0xA97F  # Hangul Jamo Extended-A  # noqa: PLR2004
            or 0xD7B0 <= cp <= 0xD7FF  # Hangul Jamo Extended-B  # noqa: PLR2004
            or 0xFFA0 <= cp <= 0xFFDC  # Halfwidth Hangul  # noqa: PLR2004
        ):
            counts["hangul"] += 1

        # ── Arabic, Persian, Ottoman: base block + presentation forms ─
        elif (
            0x0600 <= cp <= 0x06FF  # Arabic  # noqa: PLR2004
            or 0x0750 <= cp <= 0x077F  # Arabic Supplement  # noqa: PLR2004
            or 0xFB50 <= cp <= 0xFDFF  # Arabic Presentation Forms A  # noqa: PLR2004
            or 0xFE70 <= cp <= 0xFEFF  # Arabic Presentation Forms B  # noqa: PLR2004
        ):
            counts["arabic"] += 1

        # ── Hebrew ────────────────────────────────────────────────────
        elif (  # noqa: PLR2004
            0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F  # noqa: PLR2004
        ):
            counts["hebrew"] += 1

        # ── Devanagari (Hindi, Sanskrit, Marathi, Nepali) ─────────────
        elif 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF:  # noqa: PLR2004
            counts["devanagari"] += 1

        # ── Greek (modern and ancient) ────────────────────────────────
        elif 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:  # noqa: PLR2004
            counts["greek"] += 1

        # ── Cyrillic ──────────────────────────────────────────────────
        elif 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:  # noqa: PLR2004
            counts["cyrillic"] += 1

        # ── Ethiopic (Amharic, Tigrinya) ──────────────────────────────
        elif 0x1200 <= cp <= 0x137F or 0x1380 <= cp <= 0x139F:  # noqa: PLR2004
            counts["ethiopic"] += 1

        # ── Georgian ──────────────────────────────────────────────────
        elif 0x10A0 <= cp <= 0x10FF:  # noqa: PLR2004
            counts["georgian"] += 1

        # ── Coptic (proxy for Coptic-script ancient Egyptian) ─────────
        elif 0x2C80 <= cp <= 0x2CFF:  # noqa: PLR2004
            counts["egyptian"] += 1

        # ── Egyptian Hieroglyphs ──────────────────────────────────────
        elif 0x13000 <= cp <= 0x1342F:  # noqa: PLR2004
            counts["egyptian_hieroglyphs"] += 1

        # ── Thai ──────────────────────────────────────────────────────
        elif 0x0E00 <= cp <= 0x0E7F:  # noqa: PLR2004
            counts["thai"] += 1

        # ── Myanmar / Burmese (split from southeast_asian) ────────────
        elif 0x1000 <= cp <= 0x109F:  # noqa: PLR2004
            counts["myanmar"] += 1

        # ── Khmer (split from southeast_asian) ────────────────────────
        elif 0x1780 <= cp <= 0x17FF:  # noqa: PLR2004
            counts["khmer"] += 1

        # ── Other Southeast Asian: Lao, Tai Tham ─────────────────────
        elif (
            0x0E80 <= cp <= 0x0EFF  # Lao  # noqa: PLR2004
            or 0x1A20 <= cp <= 0x1AAF  # Tai Tham  # noqa: PLR2004
        ):
            counts["southeast_asian"] += 1

        # ── Mongolian ─────────────────────────────────────────────────
        elif 0x1800 <= cp <= 0x18AF:  # noqa: PLR2004
            counts["mongolian"] += 1

        # ── South Asian Dravidian: Tamil, Telugu, Kannada, Malayalam, Sinhala
        elif (
            0x0B80 <= cp <= 0x0BFF  # Tamil  # noqa: PLR2004
            or 0x0C00 <= cp <= 0x0C7F  # Telugu  # noqa: PLR2004
            or 0x0C80 <= cp <= 0x0CFF  # Kannada  # noqa: PLR2004
            or 0x0D00 <= cp <= 0x0D7F  # Malayalam  # noqa: PLR2004
            or 0x0D80 <= cp <= 0x0DFF  # Sinhala  # noqa: PLR2004
        ):
            counts["south_asian"] += 1

        # ── Armenian ──────────────────────────────────────────────────
        elif 0x0530 <= cp <= 0x058F:  # noqa: PLR2004
            counts["armenian"] += 1

        # ── Tibetan ───────────────────────────────────────────────────
        elif 0x0F00 <= cp <= 0x0FFF:  # noqa: PLR2004
            counts["tibetan"] += 1

        # ── Emoji and Miscellaneous Symbols (Bug 1 fix) ───────────────
        elif (
            # Miscellaneous Symbols
            0x2600 <= cp <= 0x27BF  # noqa: PLR2004
            # Emoji/Misc Symbols & Pictographs
            or 0x1F300 <= cp <= 0x1F9FF  # noqa: PLR2004
            # Miscellaneous Technical
            or 0x2300 <= cp <= 0x23FF  # noqa: PLR2004
            # Chess / Symbols Extended-A
            or 0x1FA00 <= cp <= 0x1FA9F  # noqa: PLR2004
            # Symbols Extended-B
            or 0x1FAD0 <= cp <= 0x1FAFF  # noqa: PLR2004
        ):
            counts["emoji"] += 1

        # ── Mathematical / Musical Symbols ────────────────────────────
        elif (
            0x2100 <= cp <= 0x214F  # Letterlike Symbols  # noqa: PLR2004
            or 0x2200 <= cp <= 0x22FF  # Mathematical Operators  # noqa: PLR2004
            or 0x2A00 <= cp <= 0x2AFF  # Supplemental Math Operators  # noqa: PLR2004
            or 0x1D400 <= cp <= 0x1D7FF  # Mathematical Alphanumeric  # noqa: PLR2004
            or 0x1D100 <= cp <= 0x1D1FF  # Musical Symbols  # noqa: PLR2004
        ):
            counts["symbolic"] += 1

        # ── Latin: Basic + Extended A/B + Supplemental + IPA ─────────
        elif (
            0x0041 <= cp <= 0x007A  # A-Z a-z  # noqa: PLR2004
            or 0x00C0 <= cp <= 0x024F  # Latin Extended  # noqa: PLR2004
            or 0x0250 <= cp <= 0x02AF  # IPA Extensions  # noqa: PLR2004
            or 0x1E00 <= cp <= 0x1EFF  # Latin Extended Additional  # noqa: PLR2004
        ):
            counts["latin"] += 1

    total = sum(counts.values())
    if total == 0:
        return ScriptType.UNKNOWN

    # Find dominant script.
    dominant_key = max(counts, key=lambda k: counts[k])
    dominant_count = counts[dominant_key]

    if dominant_count / total >= majority_threshold:
        # ScriptType.CJK is a deprecated alias; detect_script now returns the
        # four granular East Asian values (HAN, HIRAGANA, KATAKANA, HANGUL).
        # Legacy callers comparing against ScriptType.CJK must migrate.
        return ScriptType(dominant_key)

    return ScriptType.MIXED


def is_cjk_char(ch: str) -> bool:
    """Return ``True`` if *ch* is a CJK / Japanese / Korean character.

    Parameters
    ----------
    ch : str
        Single character.

    Returns
    -------
    bool
        ``True`` for CJK ideographs, hiragana, katakana, hangul.

    Examples
    --------
    >>> is_cjk_char("字")
    True
    >>> is_cjk_char("A")
    False
    >>> is_cjk_char("あ")
    True
    """
    if len(ch) != 1:
        raise ValueError(f"is_cjk_char: expected a single character, got {ch!r}.")
    cp = ord(ch)
    return bool(
        0x4E00 <= cp <= 0x9FFF  # noqa: PLR2004
        or 0x3400 <= cp <= 0x4DBF  # noqa: PLR2004
        or 0x20000 <= cp <= 0x2A6DF  # noqa: PLR2004
        or 0xF900 <= cp <= 0xFAFF  # noqa: PLR2004
        or 0x3040 <= cp <= 0x309F  # Hiragana  # noqa: PLR2004
        or 0x30A0 <= cp <= 0x30FF  # Katakana  # noqa: PLR2004
        or 0xAC00 <= cp <= 0xD7AF  # Hangul  # noqa: PLR2004
        or 0x1100 <= cp <= 0x11FF  # noqa: PLR2004
    )


def is_rtl_char(ch: str) -> bool:
    """Return ``True`` if *ch* belongs to a right-to-left script.

    Covers Arabic, Persian, Ottoman, Hebrew, and related blocks.

    Parameters
    ----------
    ch : str
        Single character.

    Returns
    -------
    bool
        ``True`` for RTL script characters.

    Examples
    --------
    >>> is_rtl_char("م")
    True
    >>> is_rtl_char("A")
    False
    >>> is_rtl_char("ש")
    True
    """
    if len(ch) != 1:
        raise ValueError(f"is_rtl_char: expected a single character, got {ch!r}.")
    cp = ord(ch)
    return bool(
        0x0590 <= cp <= 0x05FF  # Hebrew  # noqa: PLR2004
        or 0x0600 <= cp <= 0x06FF  # Arabic  # noqa: PLR2004
        or 0x0750 <= cp <= 0x077F  # Arabic Supplement  # noqa: PLR2004
        or 0xFB00  # noqa: PLR2004
        <= cp
        <= 0xFDFF  # Hebrew + Arabic Presentation Forms A  # noqa: PLR2004
        or 0xFE70 <= cp <= 0xFEFF  # Arabic Presentation Forms B  # noqa: PLR2004
    )


def split_cjk_chars(text: str) -> list[str]:
    r"""Split *text* into individual CJK character tokens.

    Non-CJK runs (Latin words, numbers, spaces) are kept as contiguous
    tokens split on whitespace.  This produces a mixed token list where
    each CJK ideograph is its own token and each Latin/numeric word is
    its own token.

    When the ``regex`` library (PyPI) is installed, iteration is over
    grapheme clusters (``\X``), which is safe for ZWJ emoji sequences,
    Devanagari conjuncts, and any multi-codepoint grapheme that might
    occur in mixed-script text.  When ``regex`` is unavailable the
    legacy codepoint-level implementation is used automatically via
    :func:`_split_cjk_chars_legacy`.

    Parameters
    ----------
    text : str
        Input text that may contain CJK characters (NFC normalized for
        best results when non-CJK grapheme clusters are present).

    Returns
    -------
    list[str]
        Mixed token list; external API is unchanged from the legacy version.

    Notes
    -----
    **User note:** This is the recommended tokenization strategy for
    Chinese, Japanese (without furigana), and Korean when a dedicated
    morphological analyser (MeCab, jieba, kss) is not available.
    Character-level tokenization loses word-level semantics but ensures
    that CJK text is not treated as one giant "word" by whitespace
    splitters.

    **Developer note:** Bug 3 fix — the original implementation iterated
    raw codepoints via ``for ch in text:``, which is not grapheme-cluster-
    safe.  After Layer 0 (GraphemeClusterNormalizer) is applied, use the
    ``regex`` path.  The legacy path is preserved as
    :func:`_split_cjk_chars_legacy` for environments without ``regex``.

    Examples
    --------
    >>> split_cjk_chars("你好 world 再见")
    ['你', '好', 'world', '再', '见']
    >>> split_cjk_chars("Hello world")
    ['Hello', 'world']
    >>> split_cjk_chars("abc 日本語 123")
    ['abc', '日', '本', '語', '123']
    """
    try:
        import regex as _regex  # noqa: PLC0415
    except ImportError:
        return _split_cjk_chars_legacy(text)

    clusters = _regex.findall(r"\X", text)
    tokens: list[str] = []
    buf: list[str] = []

    for cluster in clusters:
        if cluster in (" ", "\t", "\n", "\r"):
            if buf:
                tokens.append("".join(buf))
                buf = []
        elif len(cluster) == 1 and is_cjk_char(cluster):
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(cluster)
        else:
            buf.append(cluster)

    if buf:
        tokens.append("".join(buf))

    return [t for t in tokens if t.strip()]


def _split_cjk_chars_legacy(text: str) -> list[str]:
    r"""Legacy codepoint-level CJK splitting (pre-Layer-0 implementation).

    Kept as a fallback when the ``regex`` library is not installed.
    Identical behaviour to the original ``split_cjk_chars`` — iterates
    raw codepoints and is NOT grapheme-cluster-safe.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        Mixed token list (CJK chars + non-CJK words).

    Notes
    -----
    **Developer note:** Do not call this directly in new code.
    :func:`split_cjk_chars` dispatches here automatically when ``regex``
    is unavailable.
    """
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch in (" ", "\t", "\n", "\r"):
            if buf:
                tokens.append("".join(buf))
                buf = []
        elif len(ch) == 1 and is_cjk_char(ch):
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if t.strip()]


# ===========================================================================
# Section 5 — Multi-script sentence boundary regex pattern
# ===========================================================================

#: Compiled regex source for multi-script sentence boundary detection.
#:
#: Matches the gap between two sentences across Latin, CJK, Arabic/Persian,
#: Devanagari, Ethiopic, Armenian, Khmer, Myanmar, and other scripts.
#: Intended to replace the Latin-only ``_SENTENCE_BOUNDARY_RE`` in
#: :mod:`._sentence` when the text is non-Latin or mixed.
#:
#: Terminal characters covered:
#:
#: * ``.!?``      — ASCII (Latin, English)
#: * ``。！？``   — CJK (Chinese, Japanese, Korean)  # noqa: RUF003
#: * ``؟``        — Arabic question mark (U+061F)
#: * ``।``        — Devanagari (Hindi) full stop (U+0964)
#: * ``۔``        — Urdu full stop (U+06D4)  # noqa: RUF003
#: * ``።``        — Ethiopic full stop (U+1362)
#: * ``…``        — Ellipsis (U+2026)
#: * ``‼``        — Double exclamation (U+203C)
#: * ``⁉``        — Exclamation question (U+2049)
#: * ``։``        — Armenian full stop (U+0589)  [Bug 6 fix]  # noqa: RUF003
#: * ``។``        — Khmer full stop (U+17D4)   [Bug 6 fix]
#: * ``၊``        — Myanmar comma (U+104A)     [Bug 6 fix]
#: * ``။``        — Myanmar full stop (U+104B)  [Bug 6 fix]
#:
#: .. note::
#:    Thai has no dedicated sentence-terminal character — word and sentence
#:    boundaries must be detected via dictionary-based segmentation
#:    (``DictionaryBoundaryStrategy`` in Layer 2).  Thai text is therefore
#:    handled correctly only after Layer 2 is applied; this pattern does
#:    NOT improve Thai sentence splitting.
#:
#: .. note::
#:    Pre-Layer-0 legacy pattern.  Uses stdlib ``re`` because it predates
#:    the ``regex``-for-all-Unicode mandate and contains no ``\\X`` or
#:    ``\\p{}`` syntax.  Do NOT add ``\\X`` or ``\\p{}`` here without
#:    switching the compilation site (in ``_sentence.py``) to
#:    ``regex.compile()`` first.  See Bug 5 in
#:    ``multilang_chunker_final_review.md``.
MULTI_SCRIPT_SENTENCE_RE_PATTERN: Final[str] = (
    r"(?<=[.!?。！？؟।۔።…‼⁉։។၊။])"  # sentence terminals  # noqa: RUF001
    r"[\s\u200b\u00a0]*"  # optional whitespace / ZWSP / NBSP
    r"(?=\S)"  # followed by any non-whitespace
)


# ===========================================================================
# Section 6 — Layer 1: ScriptSpan + ScriptSegmenter
# ===========================================================================

from dataclasses import dataclass  # noqa: E402 (after top-level imports)


@dataclass(frozen=True)
class ScriptSpan:
    r"""A contiguous span of text in a single Unicode script.

    Produced by :class:`ScriptSegmenter` from NFC-normalised text.
    All index fields refer to grapheme cluster indices (as produced by
    ``regex.findall(r'\X', nfc_text)``), not raw codepoint offsets.

    Parameters
    ----------
    text : str
        The span text (NFC normalised).
    script : ScriptType
        Detected script for this span.
    direction : str
        Writing direction: ``"ltr"`` | ``"rtl"`` | ``"ttb"``.
    start : int
        Grapheme cluster index in the parent document (inclusive).
    end : int
        Grapheme cluster index in the parent document (exclusive).

    Notes
    -----
    **Developer note:** Grapheme cluster indices (start, end) refer to the
    ``regex.findall(r'\\X', nfc_text)`` list produced by
    :class:`~._normalizers._normalizer.GraphemeClusterNormalizer`, not
    codepoint offsets.  A single emoji with ZWJ may span 3+ codepoints
    but occupies exactly 1 grapheme cluster index slot.

    Examples
    --------
    >>> span = ScriptSpan(
    ...     text="Hello", script=ScriptType.LATIN, direction="ltr", start=0, end=5
    ... )
    >>> span.script
    <ScriptType.LATIN: 'latin'>
    """

    text: str
    script: ScriptType
    direction: str
    start: int
    end: int


class ScriptSegmenter:
    r"""Segment NFC-normalised text into contiguous Unicode script spans.

    Uses ``regex`` Unicode Script property matching (``\p{Script=X}``)
    rather than hard-coded codepoint ranges, so it automatically supports
    new scripts added by future Unicode versions.

    Common and Inherited codepoints (punctuation, combining marks) attach
    to the preceding script span. If no preceding span exists they attach
    to the following span.

    Parameters
    ----------
    min_span_chars : int, optional
        Minimum grapheme count for a span to be emitted as a standalone
        span.  Spans shorter than this are merged into the adjacent
        dominant span.  Default 1 (no merging).
    inherit_direction : bool, optional
        Common / Inherited codepoints inherit direction from the adjacent
        span.  Default ``True``.
    unknown_script_warning : bool, optional
        Emit a ``logger.warning`` when a codepoint's script property is not
        in the known ``ScriptType`` enum.  Default ``True``.
        The span is still produced with :attr:`ScriptType.UNKNOWN`.

    Raises
    ------
    ImportError
        If the ``regex`` (PyPI) library is not installed.  Message includes
        the exact ``pip install regex`` command.

    Notes
    -----
    **User note:** Install ``regex`` with ``pip install regex`` before using
    this class.  It is the only external dependency of Layer 1.

    **Developer note:** Uses ``regex`` Unicode property syntax.  All spans
    are produced in logical (storage) order — never reordered.
    ``ScriptSegmenter`` is stateless; all state lives in local variables
    within :meth:`segment`.

    Idempotency guarantee:
        Segmenting already-NFC text twice produces identical spans.

    References
    ----------
    UAX #24 (Script Property): https://unicode.org/reports/tr24/
    regex library: https://pypi.org/project/regex/

    Examples
    --------
    >>> seg = ScriptSegmenter()
    >>> spans = seg.segment("Hello مرحبا world")
    >>> [s.script.value for s in spans]
    ['latin', 'arabic', 'latin']
    """

    # Direction map: ScriptType value string → canonical direction string.
    # TTB is stored as logical LTR (Unicode always stores in logical order).
    DIRECTION_MAP: ClassVar[dict[str, str]] = {
        "latin": "ltr",
        "cyrillic": "ltr",
        "greek": "ltr",
        "armenian": "ltr",
        "georgian": "ltr",
        "ethiopic": "ltr",
        "devanagari": "ltr",
        "thai": "ltr",
        "tibetan": "ltr",
        "southeast_asian": "ltr",
        "south_asian": "ltr",
        # East Asian — modern horizontal; TTB is a rendering concern only
        "han": "ltr",
        "hiragana": "ltr",
        "katakana": "ltr",
        "hangul": "ltr",
        "cjk": "ltr",  # legacy alias
        # Central Asian
        "mongolian": "ttb",  # traditional; stored LTR, rendered TTB
        # RTL
        "arabic": "rtl",
        "hebrew": "rtl",
        # Symbol / emoji
        "emoji": "ltr",
        "symbolic": "ltr",
        # Ancient
        "egyptian": "ltr",
        "egyptian_hieroglyphs": "ltr",
        "myanmar": "ltr",
        "khmer": "ltr",
        # Meta
        "mixed": "ltr",
        "unknown": "ltr",
    }

    # Mapping from regex Unicode Script property name → ScriptType value string.
    SCRIPT_PROPERTY_MAP: ClassVar[dict[str, str]] = {
        "Latin": "latin",
        "Cyrillic": "cyrillic",
        "Greek": "greek",
        "Arabic": "arabic",
        "Hebrew": "hebrew",
        "Devanagari": "devanagari",
        "Han": "han",
        "Hiragana": "hiragana",
        "Katakana": "katakana",
        "Hangul": "hangul",
        "Thai": "thai",
        "Tibetan": "tibetan",
        "Georgian": "georgian",
        "Armenian": "armenian",
        "Ethiopic": "ethiopic",
        "Myanmar": "myanmar",
        "Khmer": "khmer",
        "Mongolian": "mongolian",
        "Egyptian_Hieroglyphs": "egyptian_hieroglyphs",
        "Coptic": "egyptian",
        # Common / Inherited are handled separately
        "Common": "_common",
        "Inherited": "_inherited",
    }

    def __init__(
        self,
        *,
        min_span_chars: int = 1,
        inherit_direction: bool = True,
        unknown_script_warning: bool = True,
    ) -> None:
        try:
            import regex as _regex  # noqa: PLC0415

            self._regex = _regex
        except ImportError as exc:
            raise ImportError(
                "ScriptSegmenter requires the `regex` library (Layer 1). "
                "Install it with: pip install regex"
            ) from exc

        if min_span_chars < 1:
            raise ValueError(
                f"ScriptSegmenter: min_span_chars must be >= 1, got {min_span_chars!r}."
            )

        self._min_span_chars = min_span_chars
        self._inherit_direction = inherit_direction
        self._unknown_script_warning = unknown_script_warning

    def _get_cluster_script(self, cluster: str) -> str:
        """Return the Script property name for the first codepoint of *cluster*.

        Parameters
        ----------
        cluster : str
            A single grapheme cluster (one or more codepoints).

        Returns
        -------
        str
            One of the ``SCRIPT_PROPERTY_MAP`` keys, ``"_common"``,
            ``"_inherited"``, or ``"_unknown"`` for unmapped scripts.
        """
        if not cluster:
            return "_common"
        first_cp = cluster[0]
        try:
            prop = self._regex.regex.get_script(first_cp)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            prop = None

        if prop is None:
            # Fallback: use Unicode general category for Common/Inherited.
            import unicodedata  # noqa: PLC0415

            cat = unicodedata.category(first_cp)
            if cat in ("Cf", "Mn", "Mc", "Me"):
                return "_inherited"
            return "_common"

        return self.SCRIPT_PROPERTY_MAP.get(prop, "_unknown")

    def segment(self, text: str) -> list[ScriptSpan]:  # noqa: PLR0912
        r"""Segment *text* into contiguous :class:`ScriptSpan` objects.

        Parameters
        ----------
        text : str
            NFC-normalised input text. Run
            :class:`~._normalizers._normalizer.GraphemeClusterNormalizer`
            first for correctness on non-Latin scripts.

        Returns
        -------
        list[ScriptSpan]
            Non-empty list of spans in logical order.  Returns a single
            :attr:`ScriptType.UNKNOWN` span for empty input.

        Notes
        -----
        **Developer note:** Common / Inherited codepoints (punctuation,
        diacritics) are attributed to the preceding span; if there is no
        preceding span they are deferred and attributed to the next span.
        Whitespace-only spans are attributed to ``_common`` and merged
        into adjacent spans.
        """
        if not text:
            return [
                ScriptSpan(
                    text="",
                    script=ScriptType.UNKNOWN,
                    direction="ltr",
                    start=0,
                    end=0,
                )
            ]

        clusters: list[str] = self._regex.findall(r"\X", text)
        n_clusters = len(clusters)

        # Build a parallel script-key list — one entry per grapheme cluster.
        cluster_scripts: list[str] = []
        for cluster in clusters:
            cluster_scripts.append(self._get_cluster_script(cluster))

        # Resolve Common / Inherited: attach to adjacent known script.
        # Pass 1 — forward pass: attach to preceding.
        resolved: list[str] = list(cluster_scripts)
        last_known = "_common"
        for i, sc in enumerate(cluster_scripts):
            if sc not in ("_common", "_inherited", "_unknown"):
                last_known = sc
            else:
                resolved[i] = last_known  # may still be "_common" at start

        # Pass 2 — backward pass: attach leading Common/Inherited to first
        # known script that follows.
        first_known = "_common"
        for sc in reversed(resolved):
            if sc not in ("_common", "_inherited", "_unknown"):
                first_known = sc
                break
        for i, sc in enumerate(resolved):
            if sc in ("_common", "_inherited", "_unknown"):
                resolved[i] = first_known
            else:
                break

        # Edge case: if first_known is still "_common" after both passes, the
        # ENTIRE text contains only Common/Inherited characters (all punctuation,
        # digits, spaces, symbols — no letters from any known script).
        # Replace remaining "_common" sentinels with "_unknown" so the span-build
        # loop below maps them to ScriptType.UNKNOWN cleanly without emitting the
        # "unrecognised script property" warning (which was designed for NEW Unicode
        # scripts, not for this expected Common-only edge case).
        if first_known == "_common":
            resolved = [
                "_unknown" if sc in ("_common", "_inherited", "_unknown") else sc
                for sc in resolved
            ]

        # Build runs of identical resolved script keys.
        spans: list[ScriptSpan] = []
        i = 0
        while i < n_clusters:
            run_script_key = resolved[i]
            j = i
            while j < n_clusters and resolved[j] == run_script_key:
                j += 1

            run_text = "".join(clusters[i:j])

            # Map script key → ScriptType (with unknown-script warning).
            # NOTE: "_common" should no longer appear here after the resolution
            # passes above.  If it does, it means a NEWLY ADDED Unicode script
            # that shares the "Common" script property — warn as before.
            try:
                script_enum = ScriptType(run_script_key)
            except ValueError:
                if self._unknown_script_warning:
                    logger.warning(
                        "ScriptSegmenter: unrecognised script property %r "
                        "at grapheme cluster index %d. Falling back to "
                        "ScriptType.UNKNOWN. If this is a recently added "
                        "Unicode script, update SCRIPT_PROPERTY_MAP.",
                        run_script_key,
                        i,
                    )
                script_enum = ScriptType.UNKNOWN

            direction = self.DIRECTION_MAP.get(script_enum.value, "ltr")
            spans.append(
                ScriptSpan(
                    text=run_text,
                    script=script_enum,
                    direction=direction,
                    start=i,
                    end=j,
                )
            )
            i = j

        # Merge very short spans into the adjacent dominant span.
        if self._min_span_chars > 1 and len(spans) > 1:
            spans = self._merge_short_spans(spans, self._min_span_chars)

        return spans

    @staticmethod
    def _merge_short_spans(
        spans: list[ScriptSpan],
        min_chars: int,
    ) -> list[ScriptSpan]:
        """Merge spans shorter than *min_chars* into adjacent dominant span.

        Parameters
        ----------
        spans : list[ScriptSpan]
            Spans to merge.
        min_chars : int
            Minimum grapheme count threshold.

        Returns
        -------
        list[ScriptSpan]
            Merged span list (always non-empty).
        """
        merged: list[ScriptSpan] = []
        for span in spans:
            span_len = span.end - span.start
            if span_len < min_chars and merged:
                prev = merged[-1]
                # Merge into previous span, keep previous script/direction.
                merged[-1] = ScriptSpan(
                    text=prev.text + span.text,
                    script=prev.script,
                    direction=prev.direction,
                    start=prev.start,
                    end=span.end,
                )
            else:
                merged.append(span)
        return merged
