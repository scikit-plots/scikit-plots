#!/usr/bin/env python3
"""Generate ``scikitplot/cleanprompt/_confusables.py`` from UTS #39 data.

Usage::

    python maintenances/cleanprompt/_maintenance/tools/generate_confusables.py \
        path/to/confusables.txt [--check]

``confusables.txt`` is the Unicode Security Mechanisms data file (UTS #39,
https://www.unicode.org/Public/security/latest/confusables.txt). The tool
writes the runtime table, or with ``--check`` exits 1 when the committed
table differs from what the file produces.

Notes
-----
**Developer notes - what the table holds.** One entry per non-ASCII letter or
digit that looks like one ASCII letter or digit, mapping it to that
character. UTS #39 maps every character to a *skeleton prototype*, and
prototypes ignore case: ``I``, ``l`` and ``1`` share the prototype ``l``, so
the Cyrillic capital ``U+0406`` maps to ``l``. Detection needs the ASCII
character the letter impersonates, so the tool builds each prototype's class
of ASCII members (the ASCII characters whose own mapping is that prototype,
and the prototype itself when it is ASCII) and picks the member with the same
general category as the source: an upper-case letter takes the upper-case
member, a lower-case letter the lower-case one, a digit the digit of the
same numeric value (Bengali ``U+09EA``, four, looks like ``8`` and is left
out: a number written in that script must not be misread). When no
single member fits, the character is left out; a guess would change what a
pattern reads. Only letters (``L*``) and decimal digits (``Nd``) are mapped:
a symbol folded into a letter could join two tokens. Every entry is one code
point to one code point, so the detection view's offsets are unchanged.

The output is deterministic for a given input file and a given
:mod:`unicodedata` version (letter case, category and digit value come from
the running Python): entries are sorted by code point, and the data file's
``Version`` and ``Date`` and the ``unicodedata`` version are recorded, so a
``--check`` run on another Python names the version it differs in.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

#: Where the generated module lives, relative to the repository root.
TARGET = Path("scikitplot/cleanprompt/_confusables.py")

#: ASCII characters a look-alike may stand for.
_ASCII_ALNUM = frozenset(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


def _parse(lines: list[str]) -> tuple[dict[str, str], str, str]:
    """
    Return ``{source: prototype}`` for single-code-point sources, and the header.

    Parameters
    ----------
    lines : list of str
        The data file.

    Returns
    -------
    mapping : dict of str to str
        Each single-code-point source and its prototype sequence.
    version, date : str
        From the header; empty when absent.

    Raises
    ------
    ValueError
        If a data line does not have the documented ``source ; target ; type``
        shape.
    """
    mapping: dict[str, str] = {}
    version = date = ""
    for number, raw in enumerate(lines, 1):
        line = raw.lstrip("﻿")
        if line.startswith("#"):
            text = line[1:].strip()
            if text.startswith("Version:"):
                version = text.split(":", 1)[1].strip()
            elif text.startswith("Date:"):
                date = text.split(":", 1)[1].strip()
            continue
        body = line.split("#", 1)[0].strip()
        if not body:
            continue
        fields = [field.strip() for field in body.split(";")]
        if len(fields) < 2:  # noqa: PLR2004 - the documented record shape
            msg = f"line {number}: expected 'source ; target ; type', got {raw!r}"
            raise ValueError(msg)
        source = fields[0].split()
        if len(source) != 1:
            continue
        mapping[chr(int(source[0], 16))] = "".join(
            chr(int(point, 16)) for point in fields[1].split()
        )
    return mapping, version, date


def build(lines: list[str]) -> tuple[dict[int, str], str, str]:
    """
    Return the look-alike table ``{code point: ASCII character}`` and the header.

    Parameters
    ----------
    lines : list of str
        The data file.

    Returns
    -------
    table : dict of int to str
        Sorted by code point.
    version, date : str
        From the data file's header.
    """
    mapping, version, date = _parse(lines)
    members: dict[str, set[str]] = {}
    for char in _ASCII_ALNUM:
        members.setdefault(mapping.get(char, char), set()).add(char)
    table: dict[int, str] = {}
    for source, prototype in mapping.items():
        if source.isascii():
            continue
        category = unicodedata.category(source)
        if not (category.startswith("L") or category == "Nd"):
            continue
        candidates = members.get(prototype, set())
        if category == "Lu":
            fitting = {c for c in candidates if c.isupper()}
        elif category == "Ll":
            fitting = {c for c in candidates if c.islower()}
        elif category == "Nd":
            # A digit folds only where its shape and its value agree: Bengali
            # four looks like 8, and folding it so would misread a number
            # genuinely written in that script.
            value = str(unicodedata.digit(source))
            fitting = {c for c in candidates if c == value}
        else:
            fitting = candidates
        if len(fitting) == 1:
            table[ord(source)] = next(iter(fitting))
    return dict(sorted(table.items())), version, date


def render(table: dict[int, str], version: str, date: str) -> str:
    """Return the generated module's text."""
    rows = "\n".join(
        f'    0x{point:05X}: "{char}",  # {unicodedata.name(chr(point), "?")}'
        for point, char in table.items()
    )
    return f'''"""
Letters and digits of other scripts that look like one ASCII letter or digit.

Generated by ``maintenances/cleanprompt/_maintenance/tools/generate_confusables.py``
from the Unicode Security Mechanisms data file ``confusables.txt`` (UTS #39),
Version {version or "unknown"}, Date {date or "unknown"}. Do not edit by hand:
regenerate with a newer data file instead.

Notes
-----
**Developer notes.** Read only by the detection view
(:func:`~._canonical.detection_view`), which folds each character to the
ASCII character it impersonates so that structural patterns see a value
spelled with one (``CP-114``). One code point to one code point, so offsets
are unchanged; the source text is never rewritten. How each entry was chosen
is described in the generator's module notes.

The data is Copyright (C) 1991-2020 Unicode, Inc., distributed under the
Unicode License Agreement - Data Files and Software (SPDX
``Unicode-DFS-2016``), whose full text and permission notice are in
``LICENSES/LICENSE_unicode``.
"""

from __future__ import annotations

__all__ = ["CONFUSABLES", "UNICODEDATA_VERSION", "UNICODE_VERSION"]

#: UTS #39 data version this table was generated from.
UNICODE_VERSION = "{version}"

#: :mod:`unicodedata` version the generator classified characters with.
UNICODEDATA_VERSION = "{unicodedata.unidata_version}"

#: Code point to the ASCII letter or digit it looks like.
CONFUSABLES: dict[int, str] = {{
{rows}
}}
'''


def main(argv: list[str] | None = None) -> int:
    """Write or check the generated table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="path to confusables.txt")
    parser.add_argument("--check", action="store_true", help="exit 1 if stale")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    lines = args.data.read_text(encoding="utf-8").splitlines()
    text = render(*build(lines))
    target = args.repo / TARGET
    if args.check:
        current = target.read_text(encoding="utf-8") if target.is_file() else ""
        if current != text:
            sys.stderr.write(
                f"{TARGET} is stale; regenerate it (this Python's unicodedata "
                f"is {unicodedata.unidata_version})\n"
            )
            return 1
        return 0
    target.write_text(text, encoding="utf-8")
    sys.stdout.write(f"wrote {TARGET}: {len(text.splitlines())} lines\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
