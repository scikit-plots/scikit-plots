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

Supported extension points
--------------------------
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

Script detection
----------------
:func:`detect_script` returns a :class:`ScriptType` member describing the
dominant Unicode script in a text sample.  Chunkers use this to auto-configure
sentence-boundary patterns and punctuation stripping for CJK / RTL / ancient
scripts without the caller needing to pass explicit hints.

Usage examples
--------------
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

Python compatibility
--------------------
Python 3.8-3.15.  No external dependencies.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import unicodedata  # noqa: F401
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Type  # noqa: F401

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python 3.7 shim (not needed at 3.8+ but harmless)
    from typing_extensions import (  # type: ignore[assignment]
        Protocol,
        runtime_checkable,
    )

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
    "MULTI_SCRIPT_SENTENCE_RE_PATTERN",
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
    """Thread-safe(ish) module-level registry for named custom components.

    Each registry holds a ``dict[str, Protocol]`` accessible via module-level
    helpers (:func:`register_tokenizer`, :func:`get_tokenizer`, etc.).

    Notes
    -----
    **Developer note:** The registry is intentionally not thread-locked.
    Registration happens at import/startup time; concurrent reads during
    inference are safe because dict lookups in CPython are atomic under the
    GIL.  If you register from a worker thread, synchronize externally.

    Raises
    ------
    KeyError
        :meth:`get` raises ``KeyError`` when the name is not registered.
    """  # noqa: D205

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._store: dict[str, Any] = {}

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
        return sorted(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
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
    CJK = "cjk"
    ARABIC = "arabic"
    HEBREW = "hebrew"
    DEVANAGARI = "devanagari"
    GREEK = "greek"
    CYRILLIC = "cyrillic"
    ETHIOPIC = "ethiopic"
    GEORGIAN = "georgian"
    EGYPTIAN = "egyptian"
    THAI = "thai"
    SOUTHEAST_ASIAN = "southeast_asian"
    SOUTH_ASIAN = "south_asian"
    ARMENIAN = "armenian"
    TIBETAN = "tibetan"
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
    <ScriptType.CJK: 'cjk'>
    >>> detect_script("Ἡ γλῶσσα")
    <ScriptType.GREEK: 'greek'>
    >>> detect_script("12345 !@#$%")
    <ScriptType.UNKNOWN: 'unknown'>
    """
    if not text:
        return ScriptType.UNKNOWN

    sample = text[:sample_size]
    counts: dict[str, int] = {
        "latin": 0,
        "cjk": 0,
        "arabic": 0,
        "hebrew": 0,
        "devanagari": 0,
        "greek": 0,
        "cyrillic": 0,
        "ethiopic": 0,
        "georgian": 0,
        "egyptian": 0,
        "thai": 0,
        "southeast_asian": 0,
        "south_asian": 0,
        "armenian": 0,
        "tibetan": 0,
    }

    for ch in sample:
        cp = ord(ch)

        # CJK Unified Ideographs + Extension A/B/C/D/E/F
        if (
            0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs  # noqa: PLR2004
            or 0x3400 <= cp <= 0x4DBF  # Extension A  # noqa: PLR2004
            or 0x20000 <= cp <= 0x2A6DF  # Extension B  # noqa: PLR2004
            or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs  # noqa: PLR2004
            or 0x3040 <= cp <= 0x309F  # Hiragana  # noqa: PLR2004
            or 0x30A0 <= cp <= 0x30FF  # Katakana  # noqa: PLR2004
            or 0xAC00 <= cp <= 0xD7AF  # Hangul Syllables  # noqa: PLR2004
            or 0x1100 <= cp <= 0x11FF  # Hangul Jamo  # noqa: PLR2004
            or 0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation  # noqa: PLR2004
        ):
            counts["cjk"] += 1

        # Arabic, Persian, Ottoman: base block + presentation forms
        elif (
            0x0600 <= cp <= 0x06FF  # Arabic  # noqa: PLR2004
            or 0x0750 <= cp <= 0x077F  # Arabic Supplement  # noqa: PLR2004
            or 0xFB50 <= cp <= 0xFDFF  # Arabic Presentation Forms A  # noqa: PLR2004
            or 0xFE70 <= cp <= 0xFEFF  # Arabic Presentation Forms B  # noqa: PLR2004
        ):
            counts["arabic"] += 1

        # Hebrew
        elif 0x0590 <= cp <= 0x05FF or 0xFB1D <= cp <= 0xFB4F:  # noqa: PLR2004
            counts["hebrew"] += 1

        # Devanagari (Hindi, Sanskrit, Marathi, Nepali)
        elif 0x0900 <= cp <= 0x097F or 0xA8E0 <= cp <= 0xA8FF:  # noqa: PLR2004
            counts["devanagari"] += 1

        # Greek (modern and ancient)
        elif 0x0370 <= cp <= 0x03FF or 0x1F00 <= cp <= 0x1FFF:  # noqa: PLR2004
            counts["greek"] += 1

        # Cyrillic
        elif 0x0400 <= cp <= 0x04FF or 0x0500 <= cp <= 0x052F:  # noqa: PLR2004
            counts["cyrillic"] += 1

        # Ethiopic (Amharic, Tigrinya)
        elif 0x1200 <= cp <= 0x137F or 0x1380 <= cp <= 0x139F:  # noqa: PLR2004
            counts["ethiopic"] += 1

        # Georgian
        elif 0x10A0 <= cp <= 0x10FF:  # noqa: PLR2004
            counts["georgian"] += 1

        # Coptic (proxy for Coptic-script ancient Egyptian)
        elif 0x2C80 <= cp <= 0x2CFF:  # noqa: PLR2004
            counts["egyptian"] += 1

        # Thai
        elif 0x0E00 <= cp <= 0x0E7F:  # noqa: PLR2004
            counts["thai"] += 1

        # Southeast Asian: Lao, Myanmar/Burmese, Khmer
        elif (
            0x0E80 <= cp <= 0x0EFF  # Lao  # noqa: PLR2004
            or 0x1000 <= cp <= 0x109F  # Myanmar (Burmese)  # noqa: PLR2004
            or 0x1780 <= cp <= 0x17FF  # Khmer  # noqa: PLR2004
            or 0x1A20 <= cp <= 0x1AAF  # Tai Tham  # noqa: PLR2004
        ):
            counts["southeast_asian"] += 1

        # South Asian Dravidian: Tamil, Telugu, Kannada, Malayalam, Sinhala
        elif (
            0x0B80 <= cp <= 0x0BFF  # Tamil  # noqa: PLR2004
            or 0x0C00 <= cp <= 0x0C7F  # Telugu  # noqa: PLR2004
            or 0x0C80 <= cp <= 0x0CFF  # Kannada  # noqa: PLR2004
            or 0x0D00 <= cp <= 0x0D7F  # Malayalam  # noqa: PLR2004
            or 0x0D80 <= cp <= 0x0DFF  # Sinhala  # noqa: PLR2004
        ):
            counts["south_asian"] += 1

        # Armenian
        elif 0x0530 <= cp <= 0x058F:  # noqa: PLR2004
            counts["armenian"] += 1

        # Tibetan
        elif 0x0F00 <= cp <= 0x0FFF:  # noqa: PLR2004
            counts["tibetan"] += 1

        # Latin: Basic + Extended A/B + Supplemental + IPA Extensions
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

    Parameters
    ----------
    text : str
        Input text that may contain CJK characters.

    Returns
    -------
    list[str]
        Mixed token list.

    Notes
    -----
    **User note:** This is the recommended tokenization strategy for
    Chinese, Japanese (without furigana), and Korean when a dedicated
    morphological analyser (MeCab, jieba, kss) is not available.
    Character-level tokenization loses word-level semantics but ensures
    that CJK text is not treated as one giant "word" by whitespace
    splitters.

    **Developer note:** Used by :func:`~._fixed_window._tokenize_whitespace`
    when ``unit=TOKENS`` and the text is detected as CJK.

    Examples
    --------
    >>> split_cjk_chars("你好 world 再见")
    ['你', '好', 'world', '再', '见']
    >>> split_cjk_chars("Hello world")
    ['Hello', 'world']
    >>> split_cjk_chars("abc 日本語 123")
    ['abc', '日', '本', '語', '123']
    """
    tokens: list[str] = []
    buf: list[str] = []

    for ch in text:
        if ch in (" ", "\t", "\n", "\r"):
            if buf:
                tokens.append("".join(buf))
                buf = []
        elif is_cjk_char(ch):
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
#: Devanagari, Ethiopic, and other scripts.  Intended to replace the
#: Latin-only ``_SENTENCE_BOUNDARY_RE`` in :mod:`._sentence` when the
#: text is non-Latin or mixed.
#:
#: Terminal characters covered:
#:
#: * ``.!?``      — ASCII (Latin, English)
#: * ``。！？``   — CJK (Chinese, Japanese, Korean)  # noqa: RUF003
#: * ``؟``        — Arabic question mark (U+061F)
#: * ``。``        — Chinese/Japanese period (U+3002)
#: * ``।``        — Devanagari (Hindi) full stop (U+0964)
#: * ``۔``        — Urdu full stop (U+06D4)  # noqa: RUF003
#: * ``።``        — Ethiopic full stop (U+1362)
#: * ``…``        — Ellipsis (U+2026)
#: * ``‼``        — Double exclamation (U+203C)
#: * ``⁉``        — Exclamation question (U+2049)
MULTI_SCRIPT_SENTENCE_RE_PATTERN: Final[str] = (
    r"(?<=[.!?。！？؟।۔።…‼⁉])"  # preceded by a sentence terminal  # noqa: RUF001
    r"[\s\u200b\u00a0]*"  # optional whitespace / ZWSP / NBSP
    r"(?=\S)"  # followed by any non-whitespace
)
