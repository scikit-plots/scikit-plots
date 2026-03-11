r"""
scikitplot.corpus._chunkers._fixed_window
=========================================
Sliding-window chunking with configurable size, overlap, and unit.

Pure Python — no external dependencies. Suitable for RAG pipelines that
require fixed-size text chunks for embedding and retrieval, where sentence
or paragraph boundaries are either absent or not semantically meaningful.

Design
------
Two unit modes are supported:

``unit="words"``
    The window is measured in whitespace-delimited tokens. Character offsets
    are computed from the original text by recording the start position of
    each token via ``re.finditer(r"\\S+", text)``. Chunk text is sliced from
    the original string, preserving all internal whitespace and punctuation.

``unit="chars"``
    The window is measured in characters. Chunks are exact character slices
    of the input, without any token alignment.

Overlap
-------
``overlap`` controls how many units (words or chars) the next window
reuses from the current one. Overlap must be in ``[0, window_size)``.
Step size is computed as ``step = window_size - overlap``.

- ``overlap=0`` → non-overlapping windows
- ``overlap=window_size-1`` → maximum overlap (advances 1 unit per step)

Character offsets
-----------------
``char_start`` in each returned tuple is the absolute character offset of the
**first character** of the chunk within the input string. This is consistent
with spaCy's ``sent.start_char`` convention.

Python compatibility
--------------------
Python 3.8-3.15. No external dependencies.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import re
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple  # noqa: F401

from scikitplot.corpus._base import ChunkerBase
from scikitplot.corpus._schema import ChunkingStrategy

logger_import = __import__("logging").getLogger(__name__)

# Regex to locate whitespace-delimited token start positions
_TOKEN_RE: re.Pattern[str] = re.compile(r"\S+")

# Accepted unit values
_VALID_UNITS = ("words", "chars")


class FixedWindowChunker(ChunkerBase):
    """
    Sliding-window chunking with configurable size, overlap, and unit mode.

    Divides text into equal-sized windows of ``window_size`` words or
    characters, advancing by ``step = window_size - overlap`` units between
    windows. Trailing windows smaller than ``window_size`` are included unless
    they fall below ``min_chars`` after stripping.

    Parameters
    ----------
    window_size : int
        Number of units (words or characters) per chunk. Must be > 0.
    overlap : int, optional
        Number of units shared between adjacent windows. Must be in
        ``[0, window_size)``. Default: ``0`` (non-overlapping).
    unit : {"words", "chars"}, optional
        Whether the window is measured in whitespace-delimited tokens
        (``"words"``) or raw characters (``"chars"``). Default: ``"words"``.
    min_chars : int, optional
        Minimum number of characters (after stripping) a chunk must contain
        to be included. Shorter trailing chunks are discarded. Default: ``1``.

    Attributes
    ----------
    strategy : ChunkingStrategy
        Class variable. Always ``ChunkingStrategy.FIXED_WINDOW``.
    step : int
        Computed as ``window_size - overlap``. Read-only property.

    Raises
    ------
    ValueError
        On any invalid constructor argument (see parameter docstrings).

    See Also
    --------
    scikitplot.corpus._chunkers.SentenceChunker : Sentence-level segmentation.
    scikitplot.corpus._chunkers.ParagraphChunker : Paragraph-level segmentation.

    Notes
    -----
    **Word unit and internal whitespace:** In ``unit="words"`` mode, each
    chunk is sliced from the original string as
    ``text[word_start[i] : word_start[i + window_size]].rstrip()``.
    This preserves all internal whitespace (including multiple spaces and
    non-ASCII whitespace) exactly as it appeared in the input.

    **Char unit and multi-byte characters:** In ``unit="chars"`` mode, ``i``
    indexes Unicode code points (Python ``str`` indexing), not UTF-8 bytes.
    This is correct for NLP tasks; adjust if byte-level alignment is required.

    **Empty text:** Always returns ``[]`` for empty or whitespace-only input.

    Examples
    --------
    Word-based windows (default):

    >>> chunker = FixedWindowChunker(window_size=4, overlap=1)
    >>> chunks = chunker.chunk("one two three four five six")
    >>> for start, text in chunks:
    ...     print(f"[{start}] {text!r}")
    [0] 'one two three four'
    [8] 'three four five six'

    Character-based windows:

    >>> chunker = FixedWindowChunker(window_size=5, overlap=2, unit="chars")
    >>> chunks = chunker.chunk("abcdefghij")
    >>> for start, text in chunks:
    ...     print(f"[{start}] {text!r}")
    [0] 'abcde'
    [3] 'defgh'
    [6] 'ghij'

    Non-overlapping word windows:

    >>> chunker = FixedWindowChunker(window_size=3, overlap=0)
    >>> chunks = chunker.chunk("a b c d e f")
    >>> len(chunks)
    2
    """

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.FIXED_WINDOW

    def __init__(
        self,
        window_size: int,
        overlap: int = 0,
        unit: str = "words",
        min_chars: int = 1,
    ) -> None:
        # --- window_size ---
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(
                f"FixedWindowChunker: window_size must be a positive int;"
                f" got {window_size!r}."
            )
        # --- overlap ---
        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError(
                f"FixedWindowChunker: overlap must be a non-negative int;"
                f" got {overlap!r}."
            )
        if overlap >= window_size:
            raise ValueError(
                f"FixedWindowChunker: overlap ({overlap}) must be less than"
                f" window_size ({window_size})."
            )
        # --- unit ---
        if unit not in _VALID_UNITS:
            raise ValueError(
                f"FixedWindowChunker: unit must be one of {_VALID_UNITS}; got {unit!r}."
            )
        # --- min_chars ---
        if not isinstance(min_chars, int) or min_chars < 1:
            raise ValueError(
                f"FixedWindowChunker: min_chars must be a positive int;"
                f" got {min_chars!r}."
            )

        self.window_size: int = window_size
        self.overlap: int = overlap
        self.unit: str = unit
        self.min_chars: int = min_chars

    @property
    def step(self) -> int:
        """
        Number of units advanced between consecutive windows.

        Returns
        -------
        int
            ``window_size - overlap``. Always >= 1.
        """
        return self.window_size - self.overlap

    # ------------------------------------------------------------------
    # ChunkerBase contract
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, str]]:
        """
        Divide ``text`` into fixed-size sliding windows.

        Parameters
        ----------
        text : str
            Raw text to divide. Empty or whitespace-only input returns ``[]``.
        metadata : dict or None, optional
            Chunk-level metadata from the reader. Not used by
            ``FixedWindowChunker`` but accepted for base-class contract
            compatibility. Default: ``None``.

        Returns
        -------
        list of (int, str)
            Ordered list of ``(char_start, chunk_text)`` pairs.
            ``char_start`` is the absolute character offset of the first
            character of the chunk within the input ``text``.

        Raises
        ------
        ValueError
            If ``text`` is ``None``.

        Examples
        --------
        >>> chunker = FixedWindowChunker(window_size=3, overlap=1)
        >>> chunker.chunk("a b c d e f g")
        [(0, 'a b c'), (4, 'c d e'), (8, 'e f g')]

        >>> FixedWindowChunker(window_size=4).chunk("")
        []
        """
        if text is None:
            raise ValueError("FixedWindowChunker.chunk: text must not be None.")

        if not text.strip():
            return []

        if self.unit == "words":
            return self._chunk_by_words(text)
        else:  # noqa: RET505
            return self._chunk_by_chars(text)

    # ------------------------------------------------------------------
    # Private: word-unit chunking
    # ------------------------------------------------------------------

    def _chunk_by_words(self, text: str) -> list[tuple[int, str]]:
        """
        Produce fixed-size word-count windows from ``text``.

        Parameters
        ----------
        text : str
            Non-empty text to chunk.

        Returns
        -------
        list of (int, str)
            ``(char_start, chunk_text)`` pairs.
        """
        # Build parallel lists: token start positions and token end positions
        token_starts: list[int] = []
        token_ends: list[int] = []
        for m in _TOKEN_RE.finditer(text):
            token_starts.append(m.start())
            token_ends.append(m.end())

        n_tokens = len(token_starts)
        if n_tokens == 0:
            return []

        results: list[tuple[int, str]] = []
        i = 0

        while i < n_tokens:
            end_token_idx = min(i + self.window_size, n_tokens)
            char_start = token_starts[i]

            # Chunk text runs from start of first token to end of last token
            # in this window. Use token_ends to avoid capturing trailing
            # whitespace from the original text.
            char_end = token_ends[end_token_idx - 1]
            chunk_text = text[char_start:char_end]

            stripped = chunk_text.strip()
            if len(stripped) >= self.min_chars:
                results.append((char_start, stripped))
            else:
                logger_import.debug(
                    "FixedWindowChunker: discarding short word-window chunk"
                    " (%d chars < min_chars=%d).",
                    len(stripped),
                    self.min_chars,
                )

            # Advance by step; stop when the next window would start beyond
            # the last token (full-step non-overlapping) or when we have
            # already processed the final token (overlapping)
            next_i = i + self.step
            if next_i >= n_tokens:
                break
            i = next_i

        return results

    # ------------------------------------------------------------------
    # Private: char-unit chunking
    # ------------------------------------------------------------------

    def _chunk_by_chars(self, text: str) -> list[tuple[int, str]]:
        """
        Produce fixed-size character windows from ``text``.

        Parameters
        ----------
        text : str
            Non-empty text to chunk.

        Returns
        -------
        list of (int, str)
            ``(char_start, chunk_text)`` pairs.
        """
        results: list[tuple[int, str]] = []
        n = len(text)
        i = 0

        while i < n:
            chunk_text = text[i : i + self.window_size]
            stripped = chunk_text.strip()
            if len(stripped) >= self.min_chars:
                results.append((i, chunk_text))
            else:
                logger_import.debug(
                    "FixedWindowChunker: discarding short char-window chunk"
                    " (%d chars < min_chars=%d).",
                    len(stripped),
                    self.min_chars,
                )
            i += self.step

        return results

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"FixedWindowChunker("
            f"window_size={self.window_size},"
            f" overlap={self.overlap},"
            f" unit={self.unit!r},"
            f" step={self.step},"
            f" min_chars={self.min_chars})"
        )


__all__ = ["FixedWindowChunker"]
