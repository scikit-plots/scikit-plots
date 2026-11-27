# scikitplot/lean/_shapes.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Statement shapes: a theorem's statement with bound names and numerals abstracted.

Shared by the index, which fingerprints each result while it reads a file,
and by discovery, which groups results of one shape. Standard library only.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ._syntax import _NAME, _Lexer

#: Kinds whose statement is a result.
_RESULTS = frozenset({"theorem", "lemma"})
#: Shape tokens: names, numerals, single symbols.
_SHAPE_TOKEN = re.compile(rf"{_NAME}|\d+(?:\.\d+)?|[^\s\w]")
_NUMERAL = re.compile(r"\d+(?:\.\d+)?\Z")
_PROP_DEFINITION = re.compile(r":\s*Prop\s*:=")
#: Where a result's signature starts: after its keyword.
_KEYWORD = re.compile(r"(?<![\w'!?.])(?:theorem|lemma)(?![\w'!?])")
_OPEN = "([{⦃⟨"
_CLOSE = ")]}⦄⟩"
#: What the signature scan stops at: brackets, and a ``:`` that is not ``:=``.
_STRUCTURE = re.compile(r"[(\[{⦃⟨)\]}⦄⟩]|:(?!=)")
#: What a string or char literal becomes in a shape, as a numeral becomes ``#``.
_LITERAL = '"'


def _mask_lines(statement: str) -> list[tuple[str, str]]:
    """Return each line of *statement* with its masked form, masked once."""
    lexer = _Lexer()
    return [(raw, lexer.mask(raw)) for raw in statement.split("\n")]


def _masked(lines: list[tuple[str, str]]) -> str:
    """
    Return masked *lines* as code: comments dropped, each literal one ``"`` token.

    The lexer blanks comments, strings and char literals alike. A blanked
    run whose text opens a string (``"``, ``r#"``) or a char literal (``'``)
    keeps one placeholder, so ``f "a" = 1`` and ``f = 1`` differ in shape
    and ``s = "a"`` and ``s = "b"`` do not.
    """
    out = []
    for raw, masked in lines:
        if masked == raw:  # no comment or literal on the line: most lines
            out.append(masked)
            continue
        chars = []
        hidden_from = None  # where the current blanked run began in raw
        for i, (code, text) in enumerate(zip(masked, raw)):
            if code == " " and text != " ":  # blanked
                if hidden_from is None:
                    hidden_from = i
                continue
            if hidden_from is not None and code == " ":  # blank between blanked
                continue
            if hidden_from is not None:
                chars.append(_run(raw, hidden_from))
                hidden_from = None
            chars.append(code)
        if hidden_from is not None:
            chars.append(_run(raw, hidden_from))
        chars.append(masked[len(raw) :])
        out.append("".join(chars))
    return " ".join(out)


def _run(raw: str, start: int) -> str:
    """Return the stand-in for a blanked run of *raw* from *start*: a literal or nothing."""
    opens = raw[start]
    if opens in "\"'" or (opens == "r" and raw[start + 1 : start + 2] in ('"', "#")):
        return f" {_LITERAL} "
    return " "


_BINDER = None  # compiled on first use (keeps import cheap)
_NAME_ONLY = r"[^\W\d][\w'!?]*"
#: A binder without a type: ``(x y)``, ``{x y}``, ``⦃x y⦄``.
_TYPELESS = re.compile(rf"[({{⦃]\s*(?:{_NAME_ONLY}\s+)*{_NAME_ONLY}\s*[)}}⦄]")
_QUANTIFIED = None


def _binders() -> tuple[Any, Any]:
    global _BINDER, _QUANTIFIED  # noqa: PLW0603
    if _BINDER is None:
        import re  # noqa: PLC0415

        name = r"[^\W\d][\w'!?]*"
        _BINDER = re.compile(rf"[(\[{{⦃⟨]\s*((?:{name}\s+)*{name})\s*:")
        _QUANTIFIED = re.compile(rf"(?:∀|∃!?|λ|\bfun\b|Π|Σ)\s*((?:{name}\s*)+)")
    return _BINDER, _QUANTIFIED


def _bound_names(statement: str) -> set[str]:
    """
    Return the names a statement binds: ``(s : Real)``, ``{x y}``, ``∀ n``, ``fun x``.

    A binder without a type, ``{x y}`` or ``(x)``, binds only in the
    signature (before the result's type): after it, ``{x}`` is a set and
    ``(f x)`` an application. ``[Group G]`` binds no name.
    """
    return _bound_in("\n".join(masked for _raw, masked in _mask_lines(statement)))


def _bound_in(masked: str) -> set[str]:
    """:func:`_bound_names` of a statement already masked."""
    binder, quantified = _binders()
    names: set[str] = set()
    for found in binder.finditer(masked):
        names.update(found.group(1).split())
    for found in quantified.finditer(masked):
        names.update(found.group(1).split())
    for group in _signature_groups(masked):
        if _TYPELESS.fullmatch(group):
            names.update(group[1:-1].split())
    return names


def _signature_groups(masked: str) -> list[str]:
    """
    Return the top-level bracket groups of a result's signature, brackets kept.

    The signature runs from the keyword to the ``:`` of the result's type;
    a group nested in a binder's type (``(h : (f x) = y)``) is not listed.
    """
    keyword = _KEYWORD.search(masked)
    if keyword is None:
        return []
    groups: list[str] = []
    depth = 0
    start = 0
    for found in _STRUCTURE.finditer(masked, keyword.end()):
        i, char = found.start(), found.group()
        if char in _OPEN:
            if not depth:
                start = i
            depth += 1
        elif char in _CLOSE:
            if depth == 1:
                groups.append(masked[start : i + 1])
            depth = max(0, depth - 1)
        elif not depth:  # the type's `:` (`:=` is not matched)
            break
    return groups


def _shape(statement: str, written: str | None) -> str | None:
    """
    Return a statement's shape: its code tokens, bound names and numerals abstracted.

    The keyword, everything before it (doc comment, attributes, modifiers)
    and the declaration's own name are dropped; each name the statement
    binds becomes ``x0``, ``x1``, ... in order of first appearance; each
    numeral becomes ``#``. Two statements of one shape differ only there.
    """
    return _shape_of(_mask_lines(statement), written)


def _shape_of(lines: list[tuple[str, str]], written: str | None) -> str | None:
    """:func:`_shape` of a statement already masked."""
    tokens = _SHAPE_TOKEN.findall(_masked(lines))
    start = next((i for i, t in enumerate(tokens) if t in _RESULTS), None)
    if start is None:
        return None
    tokens = tokens[start + 1 :]
    if tokens and written is not None and tokens[0] == written:
        tokens = tokens[1:]
    if not tokens:
        return None
    bound = _bound_in("\n".join(masked for _raw, masked in lines))
    renamed: dict[str, str] = {}
    out = []
    for token in tokens:
        if token in bound:
            out.append(renamed.setdefault(token, f"x{len(renamed)}"))
        elif _NUMERAL.match(token):
            out.append("#")
        else:
            out.append(token)
    return " ".join(out)


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _written(lines: list[tuple[str, str]], kind: str) -> str | None:
    """Return the name as written after *kind* in a masked statement, or ``None``."""
    tokens = _SHAPE_TOKEN.findall(" ".join(masked for _raw, masked in lines))
    for i, token in enumerate(tokens[:-1]):
        if token == kind:
            return tokens[i + 1]
    return None


def fingerprint(statement: str, kind: str, name: str | None) -> bytes | None:
    """
    Return the 16-byte digest of a result's shape, or ``None`` when it has none.

    *statement* is the result's text before its proof; *name* its full name
    (the written name is tried first, then the last component of *name*).
    """
    lines = _mask_lines(statement)  # masked once for every step
    written = (name or "").rsplit(".", 1)[-1] or None
    shape = _shape_of(lines, _written(lines, kind) or written)
    return None if shape is None else _digest(shape)


def states_a_claim(text: str) -> bool:
    """Return whether a definition's text gives it type ``Prop`` (``: Prop :=``)."""
    lexer = _Lexer()
    return (
        _PROP_DEFINITION.search(" ".join(lexer.mask(line) for line in text.split("\n")))
        is not None
    )
