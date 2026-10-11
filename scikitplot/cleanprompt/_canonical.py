"""
One definition of "the same value, however it is written".

A value hidden once must be recognised when it recurs in capitals, broken
across a line, with a non-breaking space, in full-width letters or with a
typographic apostrophe (``CP-070``), and a surrogate stand-in must never
contain a value this conversation holds (``CP-071``). Both questions are
answered here, so they cannot drift apart.

Notes
-----
**User notes.** Nothing to configure. Only *equivalences* are matched —
case, any run of whitespace, Unicode compatibility forms that are one
character, apostrophe and dash variants. Rewordings are not: ``Holt,
Marion``, ``M. Holt`` or a surname alone are different strings, and hiding
them is a job for ``hide=`` or the entity-recognition tier.

**Developer notes.** :func:`canonical` never changes a string's length, so a
match found in canonical text has the original's offsets. Matching ignores
case through :data:`re.IGNORECASE`, which is length-preserving, rather than
:meth:`str.casefold`, which is not (``ß`` -> ``ss``).

See Also
--------
scikitplot.cleanprompt._runtime : Remembered values and the leak check.
scikitplot.cleanprompt._engine : Surrogate stand-ins, and the detection view.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "DetectionView",
    "canonical",
    "detection_view",
    "normal_form",
    "value_pattern",
]

#: Characters written for an apostrophe, and for a hyphen or dash. Each set is
#: read as one character when a remembered value is looked for (``CP-070``).
_APOSTROPHES = "'\u2018\u2019\u02bc\u02b9\u0060\u00b4\uff07"
_DASHES = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"


@lru_cache(maxsize=1)
def _canonical_table() -> dict[int, str]:
    """
    Return the translation that gives text its canonical, same-length form.

    Notes
    -----
    **Developer notes.** Built once from :mod:`unicodedata`. A character maps
    only to a *single* character, so the canonical text has exactly the
    original's length and a match's offsets are the original's offsets. That
    excludes compatibility forms that expand (``ﬁ`` -> ``fi``); those are
    left as they are, which can only miss a variant, never move a span.
    """
    table: dict[int, str] = {}
    ranges = (range(0x80, 0x10000), range(0x1D400, 0x1D800))
    for block in ranges:
        for point in block:
            char = chr(point)
            if char.isspace():
                table[point] = " "
                continue
            folded = unicodedata.normalize("NFKC", char)
            if len(folded) == 1 and folded != char:
                table[point] = " " if folded.isspace() else folded
    for char in "\t\n\r\x0b\x0c":
        table[ord(char)] = " "
    for char in _APOSTROPHES:
        table[ord(char)] = "'"
    for char in _DASHES:
        table[ord(char)] = "-"
    return table


def canonical(text: str) -> str:
    r"""
    Return ``text`` in the form remembered values are matched in.

    Parameters
    ----------
    text : str
        Any text.

    Returns
    -------
    str
        The same length as ``text``: every whitespace character a space, every
        apostrophe ``'``, every dash ``-``, and every character with a
        single-character compatibility form (full-width, mathematical
        letters) in that form. Case is left alone; matching ignores it.

    Examples
    --------
    >>> canonical("Marion\u00a0O\u2019Brien\nＨolt")
    "Marion O'Brien Holt"
    """  # ruff: ignore[ambiguous-unicode-character-docstring]
    return text.translate(_canonical_table())


def normal_form(text: str) -> str:
    r"""
    Return the key two writings of one value share.

    Parameters
    ----------
    text : str
        A value, or a writing of one.

    Returns
    -------
    str
        :func:`canonical` text with whitespace runs collapsed to one space,
        trimmed, and case-folded. Not length-preserving: use it to compare,
        never to locate.

    Examples
    --------
    >>> normal_form("  MARION\n\u00a0Holt ") == normal_form("Marion Holt")
    True
    """
    return " ".join(canonical(text).split()).casefold()


def value_pattern(values: Iterable[str], max_gap: int | None = None) -> re.Pattern:
    r"""
    Compile a matcher for ``values`` as whole tokens of canonical text.

    Parameters
    ----------
    values : iterable of str
        Values to find. Empty or whitespace-only values are ignored.
    max_gap : int, optional
        Longest whitespace run allowed between two words of a value. ``None``
        allows any; restoration passes a bound so a streamed reply knows how
        far back a stand-in could begin (``CP-072``).

    Returns
    -------
    re.Pattern
        Search :func:`canonical` text with it; a match's offsets are the
        original text's. Longest value first, so one containing another is
        found whole. Matches nothing when no value is given.

    Notes
    -----
    **Developer notes.** Bounded by "no word character on either side"
    rather than by ``\b``: ``\b`` needs a word character *inside* the edge,
    so it never bounds ``+1 555 010 4477`` or a value ending in ``.``.

    Examples
    --------
    >>> pattern = value_pattern(["Marion Holt"])
    >>> bool(pattern.search(canonical("call MARION\nHOLT now")))
    True
    >>> bool(pattern.search(canonical("Marionette Holtz")))
    False
    """
    if max_gap is not None and (
        isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap < 1
    ):
        raise ValueError(f"max_gap must be a positive integer or None, got {max_gap!r}")
    gap = " +" if max_gap is None else f" {{1,{max_gap}}}"
    forms = {
        gap.join(re.escape(word) for word in canonical(value).split(" ") if word)
        for value in values
        if isinstance(value, str)
    }
    forms.discard("")
    ordered = sorted(forms, key=lambda one: (-len(one), one))
    body = "|".join(ordered) if ordered else "(?!)"
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


#: The first code point outside ASCII; the view leaves ASCII exactly as written.
_FIRST_NON_ASCII = 0x80

#: Code-point ranges searched for format characters (Unicode category ``Cf``).
#: Every ``Cf`` character in the Unicode versions CPython ships lies in the
#: Basic and Supplementary Multilingual Planes or in plane 14 (the tag
#: characters, ``U+E0000``-``U+E007F``); scanning these is complete for them
#: and keeps the one-time table build fast.
_FORMAT_PLANES = (range(0x20000), range(0xE0000, 0xE1000))


@lru_cache(maxsize=1)
def _view_tables() -> tuple[dict[int, str | None], re.Pattern]:
    """
    Return the detection view's translation table and its deletion matcher.

    Notes
    -----
    **Developer notes.** Two rules, both fixed and both deterministic for a
    given :mod:`unicodedata` version:

    - every **format character** (category ``Cf``: zero-width space and
      joiners, word joiner, byte-order mark, soft hyphen, bidirectional
      controls, tag characters) is deleted. None of them is visible, and each
      can be put inside a value to break a pattern while the value still reads
      the same to a person — and to a language model;
    - every other **non-ASCII** character is folded by the same table
      :func:`canonical` uses: compatibility forms that are one character
      (full-width, mathematical letters), Unicode whitespace, apostrophes and
      dashes. ASCII is left exactly as it is, so a line break stays a line
      break and code stays code.

    The translation deletes or replaces one character with one character, so
    the only offset change is a deletion, which :class:`DetectionView` maps
    back.
    """
    table: dict[int, str | None] = {
        point: char
        for point, char in _canonical_table().items()
        if point >= _FIRST_NON_ASCII
    }
    deleted = []
    for block in _FORMAT_PLANES:
        for point in block:
            if unicodedata.category(chr(point)) == "Cf":
                table[point] = None
                deleted.append(point)
    ranges = []
    for point in deleted:
        if ranges and ranges[-1][1] == point - 1:
            ranges[-1][1] = point
        else:
            ranges.append([point, point])
    body = "".join(
        (
            re.escape(chr(low))
            if low == high
            else f"{re.escape(chr(low))}-{re.escape(chr(high))}"
        )
        for low, high in ranges
    )
    return table, re.compile(f"[{body}]")


@dataclass(frozen=True)
class DetectionView:
    """
    A second reading of a text, for detection only, with the way back.

    Parameters
    ----------
    text : str
        The view: format characters removed, non-ASCII compatibility forms
        folded.
    shifts : tuple of int
        For the ``j``-th removed character, the number of view characters in
        front of it (non-decreasing). Empty when nothing was removed.

    Notes
    -----
    **Developer notes.** A view index ``v`` belongs to the original index
    ``v + k``, where ``k`` counts the removed characters that precede it —
    the ones whose ``shifts`` entry is at most ``v``. :func:`bisect.bisect_right`
    finds ``k`` in logarithmic time, so a text salted with a zero-width
    character between every letter costs no more per span than a clean one.
    """

    text: str
    shifts: tuple[int, ...] = ()

    def source_index(self, index: int) -> int:
        """
        Return the original offset of view character ``index``.

        Parameters
        ----------
        index : int
            Offset into :attr:`text`.

        Returns
        -------
        int
            Offset into the original text.
        """
        return index + bisect_right(self.shifts, index)

    def source_span(self, start: int, end: int) -> tuple[int, int]:
        """
        Map a non-empty view span ``[start, end)`` onto the original text.

        Parameters
        ----------
        start, end : int
            Offsets into :attr:`text`, ``start < end``.

        Returns
        -------
        tuple of int
            ``(start, end)`` in the original. Removed characters *inside* the
            span are included, so the original slice is the value exactly as
            it was written; removed characters at its edges are not.
        """
        return self.source_index(start), self.source_index(end - 1) + 1


def detection_view(text: str) -> DetectionView | None:
    r"""
    Return the text as detectors should also read it, or ``None`` if identical.

    Parameters
    ----------
    text : str
        The original text.

    Returns
    -------
    DetectionView or None
        ``None`` when the view would equal ``text`` — always for ASCII — so
        the common case costs one check.

    Notes
    -----
    **User notes.** Nothing to configure. A value written with an invisible
    character inside it (``ada\u200b@example.com``), in full-width letters, or
    with a non-breaking space or hyphen inside a telephone number is found as
    if it were written plainly, and the original — invisible characters
    included — is what the vault keeps and what restoration puts back.

    **Developer notes — the source is never rewritten (``CP-098``).** Detectors
    read the original text; structural and literal detectors *also* read this
    view, and their spans are mapped back onto the original before overlap
    resolution. Nothing downstream sees the view: the vault holds original
    surfaces, the rewrite indexes the original, and restoration is exact. A
    view span can only add to what the original-text pass found, and overlaps
    merge, so the view can widen redaction but never narrow it.

    Examples
    --------
    >>> detection_view("plain ascii") is None
    True
    >>> view = detection_view("ada\u200b@example.com")
    >>> view.text
    'ada@example.com'
    >>> view.source_span(0, len(view.text))
    (0, 16)
    """
    if text.isascii():
        return None
    table, deleted = _view_tables()
    folded = text.translate(table)
    if folded == text:
        return None
    if len(folded) == len(text):
        return DetectionView(folded)
    shifts = tuple(
        match.start() - position
        for position, match in enumerate(deleted.finditer(text))
    )
    return DetectionView(folded, shifts)
