# scikitplot/agent/_canonical.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Strict JSON values, their canonical text and their digests.

Everything an agent run keeps (its state, its audit trail, a tool's result)
is a *plain* JSON value: ``None``, ``bool``, ``int``, a finite ``float``,
``str``, a ``list`` of plain values, or a ``dict`` with ``str`` keys and plain
values. :func:`plain` converts a value to that form or refuses it, so a value
that would not survive a round trip through a saved state is refused when it
is produced, not when the state is read back (invariant I8).

Notes
-----
**Developer.** :func:`json.dumps` alone is not strict enough: it writes
``NaN`` unless told otherwise, turns ``{1: "a"}`` into ``{"1": "a"}`` (two
different dicts then have one text) and serialises tuples as lists (a value
read back is then not the value written). :func:`plain` refuses the first two
and converts the third explicitly. The canonical text sorts keys and has no
insignificant whitespace, so equal values have equal digests on every Python
version.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

__all__ = ["canonical_json", "digest", "plain"]

#: Nesting deeper than this is refused rather than left to the recursion limit.
MAX_DEPTH = 64
#: Integers of more digits are refused: Python refuses to write an integer of
#: more than 4300 digits as text (``sys.int_info.default_max_str_digits``),
#: so a larger one would fail when the state is saved, not when it is made.
MAX_INT_DIGITS = 4000
_INT_BOUND = 10**MAX_INT_DIGITS


def plain(value: Any, *, what: str = "value") -> Any:
    """
    Return ``value`` as a plain JSON value, or raise.

    Parameters
    ----------
    value : object
        ``None``, ``bool``, ``int``, ``float``, ``str``, or a ``list``,
        ``tuple``, or ``dict`` of them (``dict`` keys must be ``str``).
    what : str, default="value"
        Names the value in an error message.

    Returns
    -------
    object
        A new value of plain JSON types: tuples become lists, ``dict``
        subclasses become ``dict``. Scalars are returned as they are.

    Raises
    ------
    TypeError
        If the value, or anything inside it, is of another type, or a key is
        not a string.
    ValueError
        If a float is not finite, a string holds a lone surrogate, an integer
        has more than :data:`MAX_INT_DIGITS` digits, or the nesting is deeper
        than :data:`MAX_DEPTH`.

    Examples
    --------
    >>> from scikitplot.agent._canonical import plain
    >>> plain({"a": (1, 2.5, None)})
    {'a': [1, 2.5, None]}
    """
    return _plain(value, what, 0)


def _plain(value: Any, path: str, depth: int) -> Any:
    if depth > MAX_DEPTH:
        raise ValueError(f"{path}: nested deeper than {MAX_DEPTH} levels")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            # A lone surrogate ("\\ud800" decoded from JSON) has no UTF-8 form.
            raise ValueError(
                f"{path}: text with a lone surrogate is not valid JSON"
            ) from None
        # A str subclass (a str enum member) becomes the plain text it holds.
        return value if type(value) is str else str.__str__(value)
    if isinstance(value, int):
        if abs(value) >= _INT_BOUND:
            raise ValueError(f"{path}: an integer of more than {MAX_INT_DIGITS} digits")
        return int(value)  # an int subclass (an IntEnum) becomes its value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: {value!r} is not valid JSON")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item, f"{path}[{i}]", depth + 1) for i, item in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path}: a key must be a string, got {type(key).__name__} {key!r}"
                )
            # A key is text too: the same checks (a lone surrogate) apply.
            name = _plain(key, f"{path} key {key!r}", depth + 1)
            out[name] = _plain(item, f"{path}[{key!r}]", depth + 1)
        return out
    raise TypeError(f"{path}: {type(value).__name__} is not a JSON value")


def canonical_json(value: Any) -> str:
    """
    Return the canonical JSON text of a plain value.

    Keys are sorted, separators carry no whitespace, non-ASCII text is kept
    as is, and ``NaN`` or infinity raise.

    Parameters
    ----------
    value : object
        A value :func:`plain` accepts.

    Returns
    -------
    str

    Examples
    --------
    >>> from scikitplot.agent._canonical import canonical_json
    >>> canonical_json({"b": 1, "a": [True, None]})
    '{"a":[true,null],"b":1}'
    """
    return json.dumps(
        plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    """
    Return the SHA-256 of a plain value's canonical JSON, as 64 hex digits.

    Parameters
    ----------
    value : object
        A value :func:`plain` accepts.

    Returns
    -------
    str

    Examples
    --------
    >>> from scikitplot.agent._canonical import digest
    >>> digest({"a": 1}) == digest({"a": 1})
    True
    >>> len(digest([]))
    64
    """
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
