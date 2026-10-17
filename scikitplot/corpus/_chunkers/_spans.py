# scikitplot/corpus/_chunkers/_spans.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Source spans for chunkers (R3-3).

A chunker computes *spans* in the source and takes each chunk's text as a
slice of it, so ``chunk.text == source[context_start:context_end]`` holds by
construction. Re-joining pieces with ``" "`` or ``"\\n"`` produced text the
source never contained (a space inserted into Japanese, a collapsed blank
line), and searching for such text afterwards fell back silently to a wrong
position.

Notes
-----
**Developer.** These helpers are deliberately small and pure. Every chunker
that splits by strings (a sentence backend, a tokenizer) maps the strings back
to spans with :func:`locate`, which refuses a piece that is not the next
substring of the source instead of guessing.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__: list[str] = []

Span = tuple[int, int]


def locate(source: str, pieces: Sequence[str], *, where: str) -> list[Span]:
    """Return the span of each piece, searching forward in order.

    Parameters
    ----------
    source : str
        The text that was split.
    pieces : sequence of str
        Substrings of ``source``, in order (a splitter's output).
    where : str
        Who is asking, for the error message.

    Returns
    -------
    list of (int, int)
        ``(start, end)`` with ``source[start:end] == piece``.

    Raises
    ------
    ValueError
        If a piece is not found after the previous one: the splitter changed
        the text (normalised it, re-joined it), so no span can be exact.
    """
    spans: list[Span] = []
    cursor = 0
    for index, piece in enumerate(pieces):
        start = source.find(piece, cursor)
        if start == -1:
            raise ValueError(
                f"{where}: piece {index} ({piece[:40]!r}) is not a substring of "
                "the source after the previous piece, so its position cannot be "
                "known; the splitter must return the source's own text (or set "
                "include_offsets=False)."
            )
        spans.append((start, start + len(piece)))
        cursor = start + len(piece)
    return spans


def strip_span(source: str, start: int, end: int) -> Span:
    """Return ``(start, end)`` narrowed to exclude surrounding whitespace.

    Parameters
    ----------
    source : str
        The text the span indexes.
    start, end : int
        The span.

    Returns
    -------
    (int, int)
        The narrowed span; ``(start, start)`` when it holds only whitespace.
        ``source[result] == source[start:end].strip()``.
    """
    first = start
    while start < end and source[start].isspace():
        start += 1
    if start == end:
        return (first, first)
    while source[end - 1].isspace():
        end -= 1
    return (start, end)


def cover(spans: Sequence[Span]) -> Span:
    """Return the smallest span covering ``spans`` (in source order).

    Parameters
    ----------
    spans : sequence of (int, int)
        Non-empty.

    Returns
    -------
    (int, int)
        ``(first start, last end)``.
    """
    return (spans[0][0], spans[-1][1])
