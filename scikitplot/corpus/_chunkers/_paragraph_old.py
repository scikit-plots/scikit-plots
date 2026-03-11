r"""
scikitplot.corpus._chunkers._paragraph
======================================
Paragraph-boundary segmentation via blank-line splitting.

Pure Python — no external dependencies. Suitable for plain text, pre-processed
XML body text, and any format where paragraph boundaries are expressed by one
or more consecutive blank lines (``\\n\\n``).

Design
------
A "blank line" is any line that contains only whitespace characters.
The split regex ``\\n\\s*\\n+`` matches one newline, any amount of
whitespace (including none), followed by one or more additional newlines.
This correctly handles:

- Standard double-newline (``\\n\\n``)
- Windows-style line endings after normalisation (``\\n`` after ``\\r\\n`` strip)
- Extra blank lines (``\\n\\n\\n``) collapsed to one boundary
- Lines containing only spaces or tabs between newlines

Character offsets
-----------------
``char_start`` in each returned tuple is the absolute offset of the **first
non-whitespace character** of the paragraph within the input string. This
matches spaCy's ``sent.start_char`` convention used by
:class:`~scikitplot.corpus._chunkers.SentenceChunker`.

Python compatibility
--------------------
Python 3.8-3.15. No external dependencies.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple  # noqa: F401

from scikitplot.corpus._base import ChunkerBase
from scikitplot.corpus._schema import ChunkingStrategy

logger_import = __import__("logging").getLogger(__name__)

# Matches one or more blank lines: \n + optional whitespace + at least one \n
_BLANK_LINE_RE: re.Pattern[str] = re.compile(r"\n\s*\n+")


class ParagraphChunker(ChunkerBase):
    r"""
    Paragraph-boundary segmentation via consecutive blank-line splitting.

    Splits text on one or more consecutive blank lines, returning one chunk
    per paragraph. Optionally enforces a minimum character length to discard
    extremely short paragraphs (e.g. lone page numbers or section labels).

    Parameters
    ----------
    min_chars : int, optional
        Minimum number of non-whitespace characters a paragraph must contain
        to be included in the output. Paragraphs shorter than this are
        discarded silently. Default: ``1`` (include everything non-empty).
    strip_paragraph : bool, optional
        When ``True``, each paragraph text is stripped of leading and trailing
        whitespace before being returned. The ``char_start`` offset always
        points to the first non-whitespace character regardless of this flag.
        Default: ``True``.

    Attributes
    ----------
    strategy : ChunkingStrategy
        Class variable. Always ``ChunkingStrategy.PARAGRAPH``.

    See Also
    --------
    scikitplot.corpus._chunkers.SentenceChunker : spaCy sentence segmentation.
    scikitplot.corpus._chunkers.FixedWindowChunker : Sliding window chunking.

    Notes
    -----
    **Windows line endings:** This chunker expects Unix line endings (``\\n``).
    Input with ``\\r\\n`` endings will still work because ``\\r`` is treated as
    a non-newline whitespace character and the paragraph boundary regex
    ``\\n\\s*\\n+`` matches across it. For strict Windows-line-ending input,
    pre-process with ``text.replace("\\r\\n", "\\n")``.

    **Single-line input:** A document with no blank lines is returned as a
    single chunk spanning the entire input (char_start=0).

    Examples
    --------
    Standard use:

    >>> chunker = ParagraphChunker()
    >>> chunks = chunker.chunk("First paragraph.\\n\\nSecond paragraph.")
    >>> chunks
    [(0, 'First paragraph.'), (18, 'Second paragraph.')]

    With minimum character filter:

    >>> chunker = ParagraphChunker(min_chars=20)
    >>> chunks = chunker.chunk("Short.\\n\\nLong enough paragraph text here.")
    >>> len(chunks)
    1

    Empty and whitespace-only input:

    >>> ParagraphChunker().chunk("")
    []
    >>> ParagraphChunker().chunk("   \\n\\n   ")
    []
    """

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.PARAGRAPH

    def __init__(
        self,
        min_chars: int = 1,
        strip_paragraph: bool = True,
    ) -> None:
        if min_chars < 1:
            raise ValueError(
                f"ParagraphChunker: min_chars must be >= 1; got {min_chars!r}."
            )

        self.min_chars: int = min_chars
        self.strip_paragraph: bool = strip_paragraph

    # ------------------------------------------------------------------
    # ChunkerBase contract
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, str]]:
        r"""
        Split ``text`` into paragraphs on consecutive blank lines.

        Parameters
        ----------
        text : str
            Raw text to split. Empty or whitespace-only input returns ``[]``
            immediately.
        metadata : dict or None, optional
            Chunk-level metadata from the reader. Not used by
            ``ParagraphChunker`` but accepted to satisfy the base-class
            contract. Default: ``None``.

        Returns
        -------
        list of (int, str)
            Ordered list of ``(char_start, paragraph_text)`` pairs.
            ``char_start`` is the character offset of the first
            non-whitespace character of the paragraph within the input
            ``text``. Paragraphs shorter than ``min_chars`` are excluded.

        Raises
        ------
        ValueError
            If ``text`` is ``None`` (not just empty).

        Notes
        -----
        The algorithm scans for all blank-line boundary matches, slices
        between them to extract raw paragraph text, computes the
        ``char_start`` offset as ``slice_start + len(leading_whitespace)``,
        and strips the paragraph if ``strip_paragraph=True``.

        The implementation is O(n) in the length of ``text``.

        Examples
        --------
        >>> chunker = ParagraphChunker()
        >>> chunker.chunk("Para one.\\n\\nPara two.")
        [(0, 'Para one.'), (11, 'Para two.')]

        >>> chunker.chunk("")
        []

        >>> chunker.chunk("No blank lines here.")
        [(0, 'No blank lines here.')]
        """
        if text is None:
            raise ValueError("ParagraphChunker.chunk: text must not be None.")

        if not text.strip():
            return []

        results: list[tuple[int, str]] = []
        pos: int = 0

        for match in _BLANK_LINE_RE.finditer(text):
            raw_chunk = text[pos : match.start()]
            self._append_chunk(results, raw_chunk, pos)
            pos = match.end()

        # Handle trailing text after the last blank-line boundary (or the
        # entire text when there are no blank lines)
        raw_chunk = text[pos:]
        self._append_chunk(results, raw_chunk, pos)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _append_chunk(
        self,
        results: list[tuple[int, str]],
        raw_chunk: str,
        slice_start: int,
    ) -> None:
        """
        Strip ``raw_chunk``, compute absolute ``char_start``, apply
        ``min_chars`` filter, and append to ``results`` in-place.

        Parameters
        ----------
        results : list
            Output list to append to.
        raw_chunk : str
            Unstripped paragraph text slice from the input.
        slice_start : int
            Absolute character position where ``raw_chunk`` begins within
            the original input text.
        """  # noqa: D205
        stripped = raw_chunk.strip()
        if not stripped:
            return
        if len(stripped) < self.min_chars:
            logger_import.debug(
                "ParagraphChunker: discarding short paragraph (%d chars < min_chars=%d): %r",
                len(stripped),
                self.min_chars,
                stripped[:40],
            )
            return

        # Compute absolute char_start: position of first non-whitespace
        # character within the original text
        leading_ws = len(raw_chunk) - len(raw_chunk.lstrip())
        char_start = slice_start + leading_ws

        paragraph_text = stripped if self.strip_paragraph else raw_chunk
        results.append((char_start, paragraph_text))

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"ParagraphChunker("
            f"min_chars={self.min_chars},"
            f" strip_paragraph={self.strip_paragraph})"
        )


__all__ = ["ParagraphChunker"]
