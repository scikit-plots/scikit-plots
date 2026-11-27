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


_BINDER = None  # compiled on first use (keeps import cheap)
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
    """Return the names a statement binds: ``(s : Real)``, ``{x y}``, ``∀ n``, ``fun x``."""
    binder, quantified = _binders()
    lexer = _Lexer()
    masked = "\n".join(lexer.mask(line) for line in statement.split("\n"))
    names: set[str] = set()
    for found in binder.finditer(masked):
        names.update(found.group(1).split())
    for found in quantified.finditer(masked):
        names.update(found.group(1).split())
    return names


def _shape(statement: str, written: str | None) -> str | None:
    """
    Return a statement's shape: its code tokens, bound names and numerals abstracted.

    The keyword, everything before it (doc comment, attributes, modifiers)
    and the declaration's own name are dropped; each name the statement
    binds becomes ``x0``, ``x1``, ... in order of first appearance; each
    numeral becomes ``#``. Two statements of one shape differ only there.
    """
    lexer = _Lexer()
    masked = " ".join(lexer.mask(line) for line in statement.split("\n"))
    tokens = _SHAPE_TOKEN.findall(masked)
    start = next((i for i, t in enumerate(tokens) if t in _RESULTS), None)
    if start is None:
        return None
    tokens = tokens[start + 1 :]
    if tokens and written is not None and tokens[0] == written:
        tokens = tokens[1:]
    if not tokens:
        return None
    bound = _bound_names(statement)
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


def _written(statement: str, kind: str) -> str | None:
    """Return the name as written after *kind* in a statement, or ``None``."""
    lexer = _Lexer()
    masked = " ".join(lexer.mask(line) for line in statement.split("\n"))
    tokens = _SHAPE_TOKEN.findall(masked)
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
    written = (name or "").rsplit(".", 1)[-1] or None
    shape = _shape(statement, _written(statement, kind) or written)
    return None if shape is None else _digest(shape)


def states_a_claim(text: str) -> bool:
    """Return whether a definition's text gives it type ``Prop`` (``: Prop :=``)."""
    lexer = _Lexer()
    return (
        _PROP_DEFINITION.search(" ".join(lexer.mask(line) for line in text.split("\n")))
        is not None
    )
