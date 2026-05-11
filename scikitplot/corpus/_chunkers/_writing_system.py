# scikitplot/corpus/_chunkers/_writing_system.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._chunkers._writing_system
============================================
Layer 2 — Writing-system-aware segmentation strategies.

Each :class:`SegmentationStrategy` receives a :class:`ScriptSpan` (produced by
Layer 1 :class:`~._custom_tokenizer.ScriptSegmenter`) and returns a list of
:class:`~._types.Chunk` objects segmented according to the linguistic
conventions of that script.

Built-in strategy priority (highest → lowest):

* Custom strategy registered under the span's script key in
  :class:`~._custom_tokenizer.CustomTokenizerRegistry`
* Script-specific primary strategy (MeCab, PyThaiNLP, jieba …)
* :class:`GraphemeClusterStrategy` — guaranteed fallback for any script

Q3 compliance — zero failures:
    Every strategy that cannot load its optional dependency logs a
    ``logger.warning`` with the exact ``pip install`` command and falls
    back to :class:`GraphemeClusterStrategy`.  No hard crash, no silent
    failure.

Q4 compliance — Japanese multi-tier probe:
    :class:`JapaneseStrategy` probes MeCab → fugashi → SudachiPy →
    grapheme-cluster at **__init__ time** (once).  Per-segment calls are
    O(1) dispatch.

Python compatibility: 3.8-3.15.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from dataclasses import dataclass, field  # noqa: F401
from typing import (  # noqa: F401
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .._types import Chunk, ChunkResult  # noqa: F401
from ._custom_tokenizer import (  # noqa: F401
    CustomTokenizerRegistry,
    ScriptSegmenter,
    ScriptSpan,
    ScriptType,
    detect_script,
)

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    # Built-in strategies (also useful standalone)
    "ArabicMorphologicalStrategy",
    "CJKCharacterStrategy",
    "DeterminativeGroupStrategy",
    "DictionaryBoundaryStrategy",
    "EthiopicStrategy",
    "GraphemeClusterStrategy",
    "IndicAksharaStrategy",
    "JapaneseStrategy",
    "KoreanSyllableStrategy",
    "MongolianStrategy",
    # Protocol
    "SegmentationStrategy",
    "SpacePunctuationStrategy",
    "TibetanStrategy",
    # Adapter (Layer 2 public entry point)
    "WritingSystemAdapter",
    # Config
    "WritingSystemAdapterConfig",
]


# ===========================================================================
# Section 1 — SegmentationStrategy Protocol
# ===========================================================================


@runtime_checkable
class SegmentationStrategy(Protocol):
    """Protocol for per-script text segmentation strategies.

    All implementations MUST be stateless with respect to :meth:`segment`
    — only ``__init__`` may cause side effects (model loading, probe chain
    evaluation, etc.).

    Examples
    --------
    >>> class MyStrategy:
    ...     def segment(self, span, config):
    ...         return [Chunk(text=span.text, start_char=0, end_char=len(span.text))]
    >>> isinstance(MyStrategy(), SegmentationStrategy)
    True
    """

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment *span* into :class:`~._types.Chunk` objects.

        Parameters
        ----------
        span : ScriptSpan
            A contiguous script span from :class:`ScriptSegmenter`.
        config : WritingSystemAdapterConfig
            Adapter-level configuration (max_chunk_size, overlap, unit …).

        Returns
        -------
        list[Chunk]
            Non-empty list.  MUST return at least one ``Chunk`` for any
            non-empty ``span.text``.
        """
        ...  # pragma: no cover


# ===========================================================================
# Section 2 — WritingSystemAdapterConfig
# ===========================================================================


@dataclass(frozen=True)
class WritingSystemAdapterConfig:
    """Configuration for :class:`WritingSystemAdapter`.

    Parameters
    ----------
    unit : {"word", "sentence", "grapheme_cluster"}, optional
        Segmentation granularity requested from each strategy.
        Default ``"word"``.
    max_chunk_size : int or None, optional
        Hard cap on chunk length in grapheme clusters.  Chunks exceeding
        this limit are split further by the strategy.  Default ``None``.
    overlap : int, optional
        Number of chunks of overlap between adjacent chunks.  Default 0.
    strip_whitespace : bool, optional
        Strip leading/trailing whitespace from each chunk.  Default ``True``.
    min_chunk_length : int, optional
        Discard chunks shorter than this grapheme count.  Default 1.
    """

    unit: str = "word"
    max_chunk_size: int | None = None
    overlap: int = 0
    strip_whitespace: bool = True
    min_chunk_length: int = 1


# ===========================================================================
# Section 3 — Guaranteed fallback: GraphemeClusterStrategy
# ===========================================================================


class GraphemeClusterStrategy:
    r"""Grapheme-cluster-level fallback strategy for any script.

    Splits text into individual grapheme clusters (UAX #29 ``\X``).
    When ``unit="word"`` the clusters are joined in runs separated by
    Unicode whitespace; when ``unit="grapheme_cluster"`` every cluster
    is its own chunk.

    This is the **mandatory terminal fallback** — it never depends on any
    optional library, so it always succeeds.

    Parameters
    ----------
    (none)

    Notes
    -----
    **Developer note:** Used as the fallback by every strategy that cannot
    load its primary optional dependency.  May also be used directly for
    EMOJI, SYMBOLIC, UNKNOWN, and MIXED script spans.

    Examples
    --------
    >>> from scikitplot.corpus._chunkers._custom_tokenizer import ScriptSpan, ScriptType
    >>> cfg = WritingSystemAdapterConfig(unit="grapheme_cluster")
    >>> strategy = GraphemeClusterStrategy()
    >>> span = ScriptSpan("abc", ScriptType.LATIN, "ltr", 0, 3)
    >>> [c.text for c in strategy.segment(span, cfg)]
    ['a', 'b', 'c']
    """

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Split *span* into grapheme-cluster or whitespace-word chunks.

        Parameters
        ----------
        span : ScriptSpan
            Script span to segment.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Non-empty list; at minimum a single chunk with ``span.text``.
        """
        text = span.text
        if not text:
            return [Chunk(text="", start_char=span.start, end_char=span.start)]

        try:
            import regex as _regex  # noqa: PLC0415

            clusters: list[str] = _regex.findall(r"\X", text)
        except ImportError:
            clusters = list(text)  # codepoint fallback

        if config.unit == "grapheme_cluster":
            return self._clusters_to_chunks(clusters, span.start)

        # Default: whitespace-word grouping
        words: list[str] = []
        buf: list[str] = []
        for cluster in clusters:
            if cluster.strip() == "":
                if buf:
                    words.append("".join(buf))
                    buf = []
            else:
                buf.append(cluster)
        if buf:
            words.append("".join(buf))

        if not words:
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        return self._words_to_chunks(words, text, span.start, config)

    @staticmethod
    def _clusters_to_chunks(clusters: list[str], base: int) -> list[Chunk]:
        """Convert a flat cluster list into single-cluster Chunks."""
        result: list[Chunk] = []
        offset = base
        for cl in clusters:
            result.append(Chunk(text=cl, start_char=offset, end_char=offset + 1))
            offset += 1
        return result or [Chunk(text="", start_char=base, end_char=base)]

    @staticmethod
    def _words_to_chunks(
        words: list[str],
        source: str,
        base: int,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Convert a word list into Chunks with forward-cursor offsets."""
        chunks: list[Chunk] = []
        cursor = 0
        for word in words:
            if config.strip_whitespace:
                word = word.strip()  # noqa: PLW2901
            if len(word) < config.min_chunk_length:
                continue
            idx = source.find(word, cursor)
            if idx == -1:
                idx = cursor
            chunks.append(
                Chunk(
                    text=word,
                    start_char=base + idx,
                    end_char=base + idx + len(word),
                )
            )
            cursor = idx + len(word)
        return chunks or [
            Chunk(text=source, start_char=base, end_char=base + len(source))
        ]


# ===========================================================================
# Section 4 — Latin/Cyrillic/Greek/Armenian/Georgian whitespace strategy
# ===========================================================================


class SpacePunctuationStrategy:
    """Whitespace-and-punctuation segmentation for space-delimited scripts.

    Covers: Latin, Cyrillic, Greek, Armenian, Georgian, Ethiopic, and any
    other script where words are delimited by Unicode whitespace.

    Sentence-boundary splitting is delegated to
    :class:`~._sentence.SentenceChunker` (REGEX backend) when
    ``config.unit == "sentence"``.

    Parameters
    ----------
    (none)
    """

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment *span* by whitespace (words) or sentence boundary (sentences).

        Parameters
        ----------
        span : ScriptSpan
            Script span to segment.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Non-empty list of word or sentence chunks.
        """
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        if config.unit == "sentence":
            return self._sentence_split(text, span.start, config)

        # Word splitting via whitespace
        words = text.split()
        return GraphemeClusterStrategy._words_to_chunks(words, text, span.start, config)

    @staticmethod
    def _sentence_split(
        text: str, base: int, config: WritingSystemAdapterConfig
    ) -> list[Chunk]:
        """Split *text* into sentences using SentenceChunker REGEX backend."""
        try:
            from ._sentence import (  # noqa: PLC0415
                SentenceBackend,  # noqa: PLC0415
                SentenceChunker,
                SentenceChunkerConfig,
            )

            sc = SentenceChunker(
                SentenceChunkerConfig(
                    backend=SentenceBackend.REGEX, min_length=1, include_offsets=True
                )
            )
            result = sc.chunk(text)
            chunks: list[Chunk] = []
            for ch in result.chunks:
                start = base + (ch.start_char or 0)
                end = base + (ch.end_char or len(text))
                chunks.append(Chunk(text=ch.text, start_char=start, end_char=end))
            return chunks or [
                Chunk(text=text, start_char=base, end_char=base + len(text))
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SpacePunctuationStrategy: sentence split failed (%s). "
                "Falling back to whitespace-word split.",
                exc,
            )
            return GraphemeClusterStrategy._words_to_chunks(
                text.split(), text, base, config
            )


# ===========================================================================
# Section 5 — Arabic / Hebrew morphological strategy
# ===========================================================================


class ArabicMorphologicalStrategy:
    """Arabic and Hebrew morphological segmentation strategy.

    Primary: camel-tools (``pip install camel-tools``) for Arabic.
    Fallback: whitespace-word split (spaCy tokenisation not loaded here
    to keep the dependency chain minimal).

    For Hebrew the primary backend is ``camel-tools`` if the Hebrew
    lexicon is available; otherwise whitespace split.

    Parameters
    ----------
    (none)

    Notes
    -----
    **Developer note:** camel-tools contains a MorphAnalyzer that also
    covers Ottoman Turkish.  Load is deferred to first :meth:`segment`
    call to keep ``__init__`` fast.
    """

    _HAS_CAMEL: ClassVar[bool | None] = None  # None = not yet probed

    def _probe_camel(self) -> bool:
        if self.__class__._HAS_CAMEL is None:
            try:
                import camel_tools  # type: ignore[] # noqa: PLC0415, F401

                self.__class__._HAS_CAMEL = True
            except ImportError:
                self.__class__._HAS_CAMEL = False
                logger.warning(
                    "ArabicMorphologicalStrategy: camel-tools not installed. "
                    "Falling back to whitespace segmentation. "
                    "Install with: pip install camel-tools"
                )
        return bool(self.__class__._HAS_CAMEL)

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment Arabic/Hebrew *span* morphologically.

        Parameters
        ----------
        span : ScriptSpan
            Script span (Arabic or Hebrew).
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Morpheme or word chunks.
        """
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        if self._probe_camel() and config.unit in ("word", "morpheme"):
            try:
                return self._camel_segment(text, span.start, config)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ArabicMorphologicalStrategy: camel-tools segmentation "
                    "failed (%s). Falling back to whitespace split.",
                    exc,
                )

        # Fallback: space-delimited words
        return GraphemeClusterStrategy._words_to_chunks(
            text.split(), text, span.start, config
        )

    @staticmethod
    def _camel_segment(
        text: str, base: int, config: WritingSystemAdapterConfig
    ) -> list[Chunk]:
        """Use camel-tools simple word tokenisation as a morpheme approximation."""
        from camel_tools.tokenizers.word import (  # type: ignore[import-not-found]  # noqa: PLC0415
            simple_word_tokenize,
        )

        tokens: list[str] = simple_word_tokenize(text)
        return GraphemeClusterStrategy._words_to_chunks(tokens, text, base, config)


# ===========================================================================
# Section 6 — Han (Chinese logograph) character-level strategy
# ===========================================================================


class CJKCharacterStrategy:
    """Character-level segmentation for Han (Chinese logograph) script.

    Each Han ideograph is its own token.  Non-Han runs (Latin digits,
    punctuation) are kept as whitespace-split words.

    Optional jieba / pkuseg word segmentation is attempted when
    ``config.unit == "word"`` and the library is available.

    Parameters
    ----------
    prefer_jieba : bool, optional
        Attempt jieba segmentation for word-level chunks.  Default ``True``.
    """

    _HAS_JIEBA: ClassVar[bool | None] = None

    def __init__(self, *, prefer_jieba: bool = True) -> None:
        self._prefer_jieba = prefer_jieba
        self._probe_jieba()

    def _probe_jieba(self) -> None:
        if self.__class__._HAS_JIEBA is None:
            try:
                import jieba  # type: ignore[] # noqa: PLC0415, F401

                self.__class__._HAS_JIEBA = True
            except ImportError:
                self.__class__._HAS_JIEBA = False
                if self._prefer_jieba:
                    logger.warning(
                        "CJKCharacterStrategy: jieba not installed. "
                        "Falling back to character-level splitting. "
                        "Install with: pip install jieba"
                    )

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment Han *span* at character or word level.

        Parameters
        ----------
        span : ScriptSpan
            Han script span.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Character or jieba-word chunks.
        """
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        if self._prefer_jieba and self.__class__._HAS_JIEBA and config.unit == "word":
            try:
                return self._jieba_segment(text, span.start, config)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CJKCharacterStrategy: jieba failed (%s). "
                    "Falling back to character-level splitting.",
                    exc,
                )

        # Character-level split
        from ._custom_tokenizer import split_cjk_chars  # noqa: PLC0415

        tokens = split_cjk_chars(text)
        return GraphemeClusterStrategy._words_to_chunks(
            tokens, text, span.start, config
        )

    @staticmethod
    def _jieba_segment(
        text: str, base: int, config: WritingSystemAdapterConfig
    ) -> list[Chunk]:
        import jieba  # type: ignore[import-not-found]  # noqa: PLC0415

        tokens: list[str] = list(jieba.cut(text, cut_all=False))
        return GraphemeClusterStrategy._words_to_chunks(tokens, text, base, config)


# ===========================================================================
# Section 7 — Japanese multi-tier strategy (Q4)
# ===========================================================================


class JapaneseStrategy:
    """Multi-tier Japanese segmentation strategy.

    Probe order at ``__init__`` time (first available wins, probed once):

    1. Custom tokenizer registered in :data:`~._custom_tokenizer._TOKENIZER_REGISTRY`
       under key ``"japanese"`` — **highest priority**.
    2. MeCab via ``mecab-python3`` (requires system MeCab binary).
    3. fugashi (pure-Python MeCab wrapper).
    4. SudachiPy (pure-Python).
    5. :class:`GraphemeClusterStrategy` — guaranteed fallback.

    Parameters
    ----------
    tagger_args : str, optional
        Arguments forwarded to MeCab Tagger constructor.  Default
        ``"-Owakati"`` (space-delimited output).
    sudachi_mode : {"A", "B", "C"}, optional
        SudachiPy split mode.  A = shortest, C = longest morphemes.
        Default ``"C"``.
    custom_tokenizer_key : str, optional
        Key to look up in the custom tokenizer registry.  Default
        ``"japanese"``.

    Notes
    -----
    **Developer note:** The probe chain runs exactly once at
    ``__init__`` time and caches the result in ``self._impl``.  Per-segment
    calls are an O(1) dispatch to the cached implementation.

    Examples
    --------
    >>> strategy = JapaneseStrategy()
    >>> # strategy._impl is set to whichever backend was found
    """

    _SUDACHI_MODES: frozenset[str] = frozenset({"A", "B", "C"})

    def __init__(
        self,
        *,
        tagger_args: str = "-Owakati",
        sudachi_mode: str = "C",
        custom_tokenizer_key: str = "japanese",
    ) -> None:
        if sudachi_mode not in self._SUDACHI_MODES:
            raise ValueError(
                f"JapaneseStrategy: sudachi_mode must be one of "
                f"{sorted(self._SUDACHI_MODES)}, got {sudachi_mode!r}."
            )

        self._tagger_args = tagger_args
        self._sudachi_mode = sudachi_mode
        self._custom_key = custom_tokenizer_key
        self._impl: str = (
            self._probe()
        )  # "custom" | "mecab" | "fugashi" | "sudachi" | "grapheme"
        self._mecab_tagger: Any = None
        self._fugashi_tagger: Any = None
        self._sudachi_tokenizer: Any = None
        self._grapheme_fallback = GraphemeClusterStrategy()

        self._init_impl()

    def _probe(self) -> str:
        """Return the name of the first available backend."""
        from ._custom_tokenizer import _TOKENIZER_REGISTRY  # noqa: PLC0415

        if self._custom_key in _TOKENIZER_REGISTRY:
            logger.debug(
                "JapaneseStrategy: using custom tokenizer %r.", self._custom_key
            )
            return "custom"

        try:
            import MeCab  # type: ignore[import-not-found]  # noqa: PLC0415, N813, F401

            return "mecab"
        except ImportError:
            pass

        try:
            import fugashi  # type: ignore[import-not-found]  # noqa: PLC0415, F401

            return "fugashi"
        except ImportError:
            pass

        try:
            import sudachipy  # type: ignore[import-not-found]  # noqa: PLC0415, F401

            return "sudachi"
        except ImportError:
            pass

        logger.warning(
            "JapaneseStrategy: no Japanese morphological library found "
            "(mecab-python3, fugashi, or sudachipy). "
            "Falling back to grapheme-cluster splitting. "
            "Install one of: pip install mecab-python3 OR pip install fugashi "
            "OR pip install sudachipy sudachi-dictionary-full"
        )
        return "grapheme"

    def _init_impl(self) -> None:
        """Initialise the selected backend object once."""
        try:
            if self._impl == "mecab":
                import MeCab  # type: ignore[import-not-found]  # noqa: PLC0415, N813

                self._mecab_tagger = MeCab.Tagger(self._tagger_args)

            elif self._impl == "fugashi":
                import fugashi  # type: ignore[import-not-found]  # noqa: PLC0415

                self._fugashi_tagger = fugashi.Tagger(self._tagger_args)

            elif self._impl == "sudachi":
                import sudachipy  # type: ignore[import-not-found]  # noqa: PLC0415
                import sudachipy.dictionary  # type: ignore[import-not-found]  # noqa: PLC0415

                mode_map = {
                    "A": sudachipy.SplitMode.A,
                    "B": sudachipy.SplitMode.B,
                    "C": sudachipy.SplitMode.C,
                }
                dict_obj = sudachipy.dictionary.Dictionary()
                self._sudachi_tokenizer = (
                    dict_obj.create(),
                    mode_map[self._sudachi_mode],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "JapaneseStrategy: failed to initialise %r backend (%s). "
                "Falling back to grapheme-cluster splitting.",
                self._impl,
                exc,
            )
            self._impl = "grapheme"

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment Japanese *span* using the probed backend.

        Parameters
        ----------
        span : ScriptSpan
            Hiragana or Katakana script span.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Non-empty list of morpheme/word chunks.
        """
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        try:
            tokens = self._tokenize(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "JapaneseStrategy: tokenisation failed (%s). "
                "Falling back to grapheme-cluster splitting.",
                exc,
            )
            return self._grapheme_fallback.segment(span, config)

        return GraphemeClusterStrategy._words_to_chunks(
            tokens, text, span.start, config
        )

    def _tokenize(self, text: str) -> list[str]:
        """Run the selected backend and return a flat token list."""
        from ._custom_tokenizer import _TOKENIZER_REGISTRY  # noqa: PLC0415

        if self._impl == "custom":
            tok = _TOKENIZER_REGISTRY.get(self._custom_key)
            return tok.tokenize(text)

        if self._impl == "mecab" and self._mecab_tagger is not None:
            parsed = self._mecab_tagger.parse(text)
            return [t for t in parsed.strip().split() if t]

        if self._impl == "fugashi" and self._fugashi_tagger is not None:
            return [str(w) for w in self._fugashi_tagger(text)]

        if self._impl == "sudachi" and self._sudachi_tokenizer is not None:
            tokenizer_obj, mode = self._sudachi_tokenizer
            morphemes = tokenizer_obj.tokenize(text, mode)
            return [m.surface() for m in morphemes]

        # Grapheme fallback
        try:
            import regex as _regex  # noqa: PLC0415

            return _regex.findall(r"\X", text)
        except ImportError:
            return list(text)


# ===========================================================================
# Section 8 — Korean syllable strategy
# ===========================================================================


class KoreanSyllableStrategy:
    """Space-delimited word segmentation for Korean Hangul.

    Korean uses spaces to delimit *eojeols* (spacing units). Each eojeol is
    kept as one token.  When ``config.unit == "grapheme_cluster"`` the
    strategy delegates to :class:`GraphemeClusterStrategy`.

    Optional: KSS (Korean Sentence Splitter) for ``unit="sentence"``.

    Parameters
    ----------
    (none)
    """

    _HAS_KSS: ClassVar[bool | None] = None
    _grapheme_fallback: ClassVar[GraphemeClusterStrategy] = GraphemeClusterStrategy()

    def _probe_kss(self) -> bool:
        if self.__class__._HAS_KSS is None:
            try:
                import kss  # type: ignore[import-not-found]  # noqa: PLC0415, F401

                self.__class__._HAS_KSS = True
            except ImportError:
                self.__class__._HAS_KSS = False
                logger.warning(
                    "KoreanSyllableStrategy: kss not installed. "
                    "Sentence splitting unavailable for Korean. "
                    "Install with: pip install kss"
                )
        return bool(self.__class__._HAS_KSS)

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment Korean *span* by eojeol (space) or syllable.

        Parameters
        ----------
        span : ScriptSpan
            Hangul script span.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Eojeol, sentence, or grapheme-cluster chunks.
        """
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        if config.unit == "grapheme_cluster":
            return self._grapheme_fallback.segment(span, config)

        if config.unit == "sentence" and self._probe_kss():
            try:
                return self._kss_segment(text, span.start, config)
            except Exception as exc:  # noqa: BLE001
                logger.warning("KoreanSyllableStrategy: kss failed (%s).", exc)

        words = text.split()
        return GraphemeClusterStrategy._words_to_chunks(words, text, span.start, config)

    @staticmethod
    def _kss_segment(
        text: str, base: int, config: WritingSystemAdapterConfig
    ) -> list[Chunk]:
        import kss  # type: ignore[import-not-found]  # noqa: PLC0415

        sentences: list[str] = kss.split_sentences(text)
        return GraphemeClusterStrategy._words_to_chunks(sentences, text, base, config)


# ===========================================================================
# Section 9 — Indic aksara (grapheme-cluster) strategy
# ===========================================================================


class IndicAksharaStrategy:
    r"""Grapheme-cluster segmentation for Indic scripts.

    Devanagari, Tamil, Telugu, Kannada, Malayalam, Sinhala, and other Indic
    scripts form *aksharas* (syllabic units) that may span multiple codepoints.
    Correct segmentation REQUIRES grapheme-cluster iteration (``\X``).

    Parameters
    ----------
    (none)
    """

    _grapheme_strategy: ClassVar[GraphemeClusterStrategy] = GraphemeClusterStrategy()

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment Indic *span* at grapheme-cluster (aksara) granularity.

        Parameters
        ----------
        span : ScriptSpan
            Devanagari or other Indic script span.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Grapheme-cluster or whitespace-word chunks.
        """
        # Word-level: split on whitespace, each word is one chunk.
        # Grapheme-cluster level: each akshara is one chunk.
        if config.unit in ("grapheme_cluster", "morpheme"):
            return self._grapheme_strategy.segment(span, config)

        text = span.text
        words = text.split()
        return GraphemeClusterStrategy._words_to_chunks(words, text, span.start, config)


# ===========================================================================
# Section 10 — Dictionary boundary strategy (Thai, Lao, Khmer, Chinese)
# ===========================================================================


class DictionaryBoundaryStrategy:
    """Dictionary-based word boundary detection for no-space scripts.

    Covers:

    * **Thai** — PyThaiNLP ``newmm`` engine (``pip install pythainlp``).
    * **Lao** — PyThaiNLP ``lao`` engine.
    * **Khmer** — khmer-nltk (``pip install khmer-nltk``).
    * **Han (Chinese)** — jieba (``pip install jieba``).

    A custom callable registered under
    ``"<script_value>_boundary"`` (e.g. ``"thai_boundary"``) in the
    tokenizer registry takes priority over the probe chain.

    Parameters
    ----------
    (none)

    Notes
    -----
    **User note (Q3):** Per architecture invariant Q3, all known languages
    are attempted.  If no dictionary library is available for the detected
    script, a ``logger.warning`` is emitted and
    :class:`GraphemeClusterStrategy` is used.

    **Developer note:** Thai has no sentence-terminal character; sentence
    splitting for Thai is handled here via PyThaiNLP ``sent_tokenize``.
    """

    _probe_cache: ClassVar[dict[str, bool]] = {}
    _grapheme_fallback: ClassVar[GraphemeClusterStrategy] = GraphemeClusterStrategy()

    def _probe(self, lib: str) -> bool:
        if lib not in self._probe_cache:
            try:
                __import__(lib)
                self._probe_cache[lib] = True
            except ImportError:
                self._probe_cache[lib] = False
        return self._probe_cache[lib]

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment no-space *span* using dictionary lookup.

        Parameters
        ----------
        span : ScriptSpan
            Thai, Lao, Khmer, or Han script span.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Dictionary-word or grapheme-cluster chunks.
        """
        text = span.text
        script = span.script

        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        # Check custom registry first.
        registry_key = f"{script.value}_boundary"
        try:
            from ._custom_tokenizer import _TOKENIZER_REGISTRY  # noqa: PLC0415

            if registry_key in _TOKENIZER_REGISTRY:
                tok = _TOKENIZER_REGISTRY.get(registry_key)
                tokens = tok.tokenize(text)
                return GraphemeClusterStrategy._words_to_chunks(
                    tokens, text, span.start, config
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DictionaryBoundaryStrategy: custom registry call failed (%s).", exc
            )

        # Script-specific dictionary backends.
        if script in (ScriptType.THAI,):  # noqa: FURB171
            return self._thai_segment(text, span.start, config)

        if script in (ScriptType.SOUTHEAST_ASIAN, ScriptType.KHMER):
            return self._khmer_lao_segment(text, span.start, config, script)

        if script in (ScriptType.HAN, ScriptType.CJK):
            return self._han_segment(text, span.start, config)

        # No strategy for this script → grapheme fallback.
        logger.warning(
            "DictionaryBoundaryStrategy: no dictionary backend for %r. "
            "Falling back to GraphemeClusterStrategy.",
            script.value,
        )
        return self._grapheme_fallback.segment(span, config)

    def _thai_segment(
        self, text: str, base: int, config: WritingSystemAdapterConfig
    ) -> list[Chunk]:
        if not self._probe("pythainlp"):
            logger.warning(
                "DictionaryBoundaryStrategy: PyThaiNLP not installed. "
                "Falling back to grapheme-cluster splitting. "
                "Install with: pip install pythainlp"
            )
            return self._grapheme_fallback.segment(
                ScriptSpan(text, ScriptType.THAI, "ltr", base, base + len(text)),
                config,
            )
        try:
            if config.unit == "sentence":
                from pythainlp.tokenize import (  # type: ignore[import-not-found]  # noqa: PLC0415
                    sent_tokenize,
                )

                tokens: list[str] = sent_tokenize(text)
            else:
                from pythainlp.tokenize import (  # type: ignore[import-not-found]  # noqa: PLC0415
                    word_tokenize,
                )

                tokens = word_tokenize(text, engine="newmm")
            return GraphemeClusterStrategy._words_to_chunks(tokens, text, base, config)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DictionaryBoundaryStrategy: Thai tokenisation failed (%s).", exc
            )
            return self._grapheme_fallback.segment(
                ScriptSpan(text, ScriptType.THAI, "ltr", base, base + len(text)), config
            )

    def _khmer_lao_segment(
        self,
        text: str,
        base: int,
        config: WritingSystemAdapterConfig,
        script: ScriptType,
    ) -> list[Chunk]:
        if script == ScriptType.KHMER and self._probe("khmernltk"):
            try:
                from khmernltk import (  # type: ignore[import-not-found]  # noqa: PLC0415
                    word_tokenize as khmer_tok,
                )

                tokens: list[str] = khmer_tok(text)
                return GraphemeClusterStrategy._words_to_chunks(
                    tokens, text, base, config
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DictionaryBoundaryStrategy: khmer-nltk failed (%s).", exc
                )

        if self._probe("pythainlp"):
            try:
                from pythainlp.tokenize import (  # type: ignore[import-not-found]  # noqa: PLC0415
                    word_tokenize,
                )

                tokens = word_tokenize(text, engine="newmm")
                return GraphemeClusterStrategy._words_to_chunks(
                    tokens, text, base, config
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DictionaryBoundaryStrategy: PyThaiNLP Lao failed (%s).", exc
                )

        logger.warning(
            "DictionaryBoundaryStrategy: no dictionary for %r. "
            "Falling back to grapheme-cluster. "
            "Install: pip install khmer-nltk  OR  pip install pythainlp",
            script.value,
        )
        return self._grapheme_fallback.segment(
            ScriptSpan(text, script, "ltr", base, base + len(text)), config
        )

    def _han_segment(
        self, text: str, base: int, config: WritingSystemAdapterConfig
    ) -> list[Chunk]:
        if self._probe("jieba"):
            try:
                import jieba  # type: ignore[import-not-found]  # noqa: PLC0415

                tokens: list[str] = list(jieba.cut(text, cut_all=False))
                return GraphemeClusterStrategy._words_to_chunks(
                    tokens, text, base, config
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("DictionaryBoundaryStrategy: jieba failed (%s).", exc)

        logger.warning(
            "DictionaryBoundaryStrategy: jieba not installed for Chinese. "
            "Falling back to character-level. Install: pip install jieba"
        )
        from ._custom_tokenizer import split_cjk_chars  # noqa: PLC0415

        tokens = split_cjk_chars(text)
        return GraphemeClusterStrategy._words_to_chunks(tokens, text, base, config)


# ===========================================================================
# Section 11 — Ethiopic strategy
# ===========================================================================


class EthiopicStrategy:
    """Ethiopic (Amharic, Tigrinya) segmentation strategy.

    Sentence boundary: ``።`` (U+1362 Ethiopic Full Stop).
    Word boundary: Unicode whitespace.

    Parameters
    ----------
    (none)
    """

    _SENT_TERM = "\u1362"  # ።

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment Ethiopic *span* by sentence or word.

        Parameters
        ----------
        span : ScriptSpan
            Ethiopic script span.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Sentence or word chunks.
        """
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        if config.unit == "sentence":
            import re  # noqa: PLC0415

            parts = re.split(r"(?<=[።])\s*", text)
            parts = [p.strip() for p in parts if p.strip()]
            return GraphemeClusterStrategy._words_to_chunks(
                parts, text, span.start, config
            )

        words = text.split()
        return GraphemeClusterStrategy._words_to_chunks(words, text, span.start, config)


# ===========================================================================
# Section 12 — Tibetan strategy
# ===========================================================================


class TibetanStrategy:
    """Tibetan segmentation strategy.

    Word boundary: tsek ``·`` (U+0F0B Tibetan Mark Intersyllabic Tsheg).
    Sentence boundary: shad ``།`` (U+0F0D Tibetan Mark Shad).

    Parameters
    ----------
    (none)
    """

    _TSEK = "\u0f0b"  # ·  word boundary
    _SHAD = "\u0f0d"  # །  sentence boundary

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment Tibetan *span* by tsek (word) or shad (sentence).

        Parameters
        ----------
        span : ScriptSpan
            Tibetan script span.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Syllable, word, or sentence chunks.
        """
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        import re  # noqa: PLC0415

        if config.unit == "sentence":
            parts = re.split(r"(?<=།)\s*", text)
        else:
            # Split on tsek (syllable boundary) for word-level
            parts = re.split(rf"[{self._TSEK}\s]+", text)

        parts = [p.strip() for p in parts if p.strip()]
        return GraphemeClusterStrategy._words_to_chunks(parts, text, span.start, config)


# ===========================================================================
# Section 13 — Mongolian strategy
# ===========================================================================


class MongolianStrategy:
    """Traditional Mongolian segmentation strategy.

    Traditional Mongolian is stored left-to-right and rendered top-to-bottom.
    Word boundaries are marked by Unicode whitespace.

    Parameters
    ----------
    (none)
    """

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment Mongolian *span* by whitespace.

        Parameters
        ----------
        span : ScriptSpan
            Mongolian script span.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Word chunks (whitespace-delimited).
        """
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]
        words = text.split()
        return GraphemeClusterStrategy._words_to_chunks(words, text, span.start, config)


# ===========================================================================
# Section 14 — Egyptian Hieroglyphs determinative-group strategy
# ===========================================================================


class DeterminativeGroupStrategy:
    """Egyptian Hieroglyph segmentation into determinative groups.

    Operates on Unicode-digitised hieroglyphic text (U+13000-U+1342F).
    A determinative group is a cluster of sign codepoints terminated by
    U+13430 (EGYPTIAN HIEROGLYPH VERTICAL JOINER) or U+13432
    (EGYPTIAN HIEROGLYPH INSERT AT MIDDLE).

    For OCR output, attach the OCR pre-processor hook before calling this
    strategy via :class:`WritingSystemAdapter`.

    Parameters
    ----------
    (none)

    Notes
    -----
    **Developer note:** This strategy populates ``chunk.metadata`` with
    ``{"determinative": "<sign>", "category": "hieroglyph"}`` for each
    group, enabling downstream population of
    ``CorpusDocument.determinative_groups``.
    """

    _JOINER_CPS: frozenset[int] = frozenset({0x13430, 0x13431, 0x13432, 0x13433})

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment hieroglyphic *span* into determinative groups.

        Parameters
        ----------
        span : ScriptSpan
            Egyptian Hieroglyphs script span.
        config : WritingSystemAdapterConfig
            Adapter configuration.

        Returns
        -------
        list[Chunk]
            Determinative-group or grapheme-cluster chunks.
        """
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        if config.unit == "grapheme_cluster":
            return GraphemeClusterStrategy().segment(span, config)

        groups: list[str] = []
        buf: list[str] = []
        for ch in text:
            cp = ord(ch)
            if cp in self._JOINER_CPS:
                if buf:
                    groups.append("".join(buf))
                    buf = []
            else:
                buf.append(ch)
        if buf:
            groups.append("".join(buf))

        if not groups:
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]

        chunks: list[Chunk] = []
        cursor = 0
        for group in groups:
            idx = text.find(group, cursor)
            if idx == -1:
                idx = cursor
            chunks.append(
                Chunk(
                    text=group,
                    start_char=span.start + idx,
                    end_char=span.start + idx + len(group),
                    metadata={"determinative": group, "category": "hieroglyph"},
                )
            )
            cursor = idx + len(group)
        return chunks


# ===========================================================================
# Section 15 — WritingSystemAdapter (Layer 2 public entry point)
# ===========================================================================


class WritingSystemAdapter:
    """Dispatch per-script segmentation to the correct strategy.

    Receives a list of :class:`ScriptSpan` objects (produced by
    :class:`~._custom_tokenizer.ScriptSegmenter`) and routes each span to
    the matching :class:`SegmentationStrategy`.

    Strategy registry (priority):

    1. Custom strategy registered in
       :data:`~._custom_tokenizer._TOKENIZER_REGISTRY` under the script
       value string (e.g. ``"japanese"``, ``"arabic"``).
    2. Built-in strategy for the detected :class:`ScriptType`.
    3. :class:`GraphemeClusterStrategy` — unconditional fallback.

    Parameters
    ----------
    config : WritingSystemAdapterConfig or None, optional
        Adapter-level configuration.  Default ``None`` →
        ``WritingSystemAdapterConfig()`` (word-level, no max size).
    extra_strategies : dict[str, SegmentationStrategy] or None, optional
        Additional ``{script_value: strategy}`` mappings that override the
        built-in registry for this adapter instance only.  Useful for
        testing or domain-specific customisation without touching the
        global registry.

    Notes
    -----
    **User note:** Use :class:`WritingSystemAdapter` as the ``adapter``
    parameter of :class:`~._semantic.SemanticChunker` (Layer 3).  You can
    also call it standalone:

    .. code-block:: python

        adapter = WritingSystemAdapter(WritingSystemAdapterConfig(unit="word"))
        spans = ScriptSegmenter().segment(text)
        chunks = adapter.adapt(spans)

    **Developer note (Q3 compliance):** Every strategy that cannot load
    its optional dependency logs a warning and falls back to
    :class:`GraphemeClusterStrategy`.  This adapter additionally wraps
    every ``strategy.segment()`` call in a try/except so a bug in a
    custom strategy cannot crash the pipeline.

    Examples
    --------
    >>> adapter = WritingSystemAdapter()
    >>> spans = ScriptSegmenter().segment("Hello 世界")
    >>> chunks = adapter.adapt(spans)
    >>> [c.text for c in chunks]
    ['Hello', '世', '界']
    """

    def __init__(
        self,
        config: WritingSystemAdapterConfig | None = None,
        *,
        extra_strategies: dict[str, SegmentationStrategy] | None = None,
    ) -> None:
        self._config = config if config is not None else WritingSystemAdapterConfig()
        self._extra = dict(extra_strategies) if extra_strategies else {}
        # _fallback: GraphemeClusterStrategy is used ONLY when a registered strategy
        # raises an exception (the except branch in adapt()). It is NOT the default
        # routing for unknown/mixed scripts — that caused every unrecognised script
        # span (including Latin+punctuation text with _common script chars) to be
        # split into individual grapheme clusters, producing ~N single-char chunks
        # where N = non-whitespace character count.  All single-char chunks then fail
        # DefaultFilter(min_words=3) → 0 documents yielded by SemanticChunker.
        self._fallback = GraphemeClusterStrategy()
        # _unknown_strategy: used for UNKNOWN and MIXED script spans in normal routing.
        # SpacePunctuationStrategy handles any whitespace-delimited text safely:
        # - Splits by whitespace when spaces are present (most Latin/mixed-script text).
        # - Returns the full span as one chunk when no whitespace found (safe fallback).
        # - Respects config.unit="grapheme_cluster" if explicitly requested.
        # This is correct for UNKNOWN text: we don't know the script but whitespace
        # splitting is always a safe minimum strategy.
        self._unknown_strategy: SegmentationStrategy = SpacePunctuationStrategy()

        # Build built-in strategy map — instantiated once.
        self._builtin: dict[str, SegmentationStrategy] = {
            # Space-delimited scripts
            "latin": SpacePunctuationStrategy(),
            "cyrillic": SpacePunctuationStrategy(),
            "greek": SpacePunctuationStrategy(),
            "armenian": SpacePunctuationStrategy(),
            "georgian": SpacePunctuationStrategy(),
            "ethiopic": EthiopicStrategy(),
            # RTL scripts
            "arabic": ArabicMorphologicalStrategy(),
            "hebrew": ArabicMorphologicalStrategy(),
            # Indic
            "devanagari": IndicAksharaStrategy(),
            "south_asian": IndicAksharaStrategy(),
            # East Asian
            "han": CJKCharacterStrategy(),
            "cjk": CJKCharacterStrategy(),  # deprecated alias
            "hiragana": JapaneseStrategy(),
            "katakana": JapaneseStrategy(),
            "hangul": KoreanSyllableStrategy(),
            # No-space Southeast Asian
            "thai": DictionaryBoundaryStrategy(),
            "khmer": DictionaryBoundaryStrategy(),
            "southeast_asian": DictionaryBoundaryStrategy(),
            "myanmar": IndicAksharaStrategy(),  # grapheme-cluster safe
            # Other
            "tibetan": TibetanStrategy(),
            "mongolian": MongolianStrategy(),
            # Ancient
            "egyptian": SpacePunctuationStrategy(),  # Coptic: space-delimited
            "egyptian_hieroglyphs": DeterminativeGroupStrategy(),
            # Symbol / emoji — pure grapheme-cluster splitting (no whitespace semantics)
            "emoji": self._fallback,
            "symbolic": self._fallback,
            # UNKNOWN / MIXED: use whitespace-word splitting, not grapheme clusters.
            # BUG FIX: previously both used self._fallback (GraphemeClusterStrategy),
            # which split every character into a separate chunk.  SpacePunctuationStrategy
            # is the correct minimum: word-split when spaces exist, full-span otherwise.
            "mixed": self._unknown_strategy,
            "unknown": self._unknown_strategy,
        }

    def adapt(self, spans: Sequence[ScriptSpan]) -> list[Chunk]:
        """Segment *spans* using the per-script strategy registry.

        Parameters
        ----------
        spans : sequence of ScriptSpan
            Script spans from :class:`ScriptSegmenter`.

        Returns
        -------
        list[Chunk]
            Flat, ordered list of all chunks across all spans.
            Always non-empty for non-empty input.
        """
        all_chunks: list[Chunk] = []
        for span in spans:
            strategy = self._resolve_strategy(span.script)
            strategy_name = type(strategy).__name__
            try:
                chunks = strategy.segment(span, self._config)
                # Tag every chunk with the Layer 2 strategy that produced it.
                # SemanticChunker.chunk() reads this to populate layer2_strategy
                # in MultilangChunkMeta (provenance / debugging field).
                tagged: list[Chunk] = []
                for ch in chunks:
                    meta = dict(ch.metadata) if ch.metadata else {}
                    meta["layer2_strategy"] = strategy_name
                    tagged.append(
                        Chunk(
                            text=ch.text,
                            start_char=ch.start_char,
                            end_char=ch.end_char,
                            metadata=meta,
                        )
                    )
                all_chunks.extend(tagged)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "WritingSystemAdapter: strategy %r failed on script %r (%s). "
                    "Using grapheme-cluster fallback for this span.",
                    strategy_name,
                    span.script.value,
                    exc,
                )
                fallback_chunks = self._fallback.segment(span, self._config)
                for ch in fallback_chunks:
                    meta = dict(ch.metadata) if ch.metadata else {}
                    meta["layer2_strategy"] = "GraphemeClusterStrategy"
                    all_chunks.append(
                        Chunk(
                            text=ch.text,
                            start_char=ch.start_char,
                            end_char=ch.end_char,
                            metadata=meta,
                        )
                    )

        if not all_chunks:
            return [Chunk(text="", start_char=0, end_char=0)]
        return all_chunks

    def adapt_text(self, text: str) -> list[Chunk]:
        """Use convenience method: segment plain *text* end-to-end.

        Runs :class:`ScriptSegmenter` internally, then calls :meth:`adapt`.

        Parameters
        ----------
        text : str
            NFC-normalised input text.

        Returns
        -------
        list[Chunk]
            Flat ordered chunk list.
        """
        try:
            segmenter = ScriptSegmenter()
            spans = segmenter.segment(text)
        except ImportError:
            # regex not installed — treat entire text as a single UNKNOWN span
            span = ScriptSpan(
                text=text,
                script=ScriptType.UNKNOWN,
                direction="ltr",
                start=0,
                end=len(text),
            )
            return self._fallback.segment(span, self._config)
        return self.adapt(spans)

    def _resolve_strategy(self, script: ScriptType) -> SegmentationStrategy:
        """Return the best strategy for *script*, checking all priority levels."""
        script_key = script.value

        # 1. Extra strategies (instance-level override)
        if script_key in self._extra:
            return self._extra[script_key]

        # 2. Global custom tokenizer registry
        try:
            from ._custom_tokenizer import _TOKENIZER_REGISTRY  # noqa: PLC0415

            if script_key in _TOKENIZER_REGISTRY:
                raw = _TOKENIZER_REGISTRY.get(script_key)
                # Wrap plain TokenizerProtocol in an adapter shim
                return _RegistryStrategyShim(raw)
        except Exception:  # noqa: BLE001
            pass

        # 3. Built-in strategy map
        if script_key in self._builtin:
            return self._builtin[script_key]

        # 4. Grapheme-cluster fallback + warning
        logger.warning(
            "WritingSystemAdapter: no strategy registered for script %r. "
            "Falling back to GraphemeClusterStrategy.",
            script_key,
        )
        return self._fallback


class _RegistryStrategyShim:
    """Adapt a plain :class:`TokenizerProtocol` from the registry as a strategy."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        self._fallback = GraphemeClusterStrategy()

    def segment(
        self,
        span: ScriptSpan,
        config: WritingSystemAdapterConfig,
    ) -> list[Chunk]:
        """Segment *span* using the wrapped tokenizer."""
        text = span.text
        if not text.strip():
            return [Chunk(text=text, start_char=span.start, end_char=span.end)]
        try:
            tokens: list[str] = self._tok.tokenize(text)
            return GraphemeClusterStrategy._words_to_chunks(
                tokens, text, span.start, config
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "_RegistryStrategyShim: tokenizer failed (%s). "
                "Falling back to GraphemeClusterStrategy.",
                exc,
            )
            return self._fallback.segment(span, config)
