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
scikitplot.cleanprompt._engine : Surrogate stand-ins.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache

__all__ = [
    "canonical",
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
