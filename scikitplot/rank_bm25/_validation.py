"""
Row-index and count contract shared by every public boundary.

A position or a count that reaches a public entry point unchecked was recorded
four times in three submodules: an artifact ordinal that resolved ``-1`` and
``True`` to real rows, batch scoring positions that reached NumPy fancy
indexing as a boolean mask, a result count that truncated a ranking from the
end when negative, and a tool result size accepted as ``-1``. Fixing each
locally would have left four error shapes for one question, so the question is
answered here and imported.

Notes
-----
**User.** These raise the exception Python already uses for the mistake: an
out-of-range position raises :class:`IndexError`, a count outside its range
raises :class:`ValueError`, and a value of the wrong type raises
:class:`TypeError`. Messages name the offending value, the bound, and the
parameter as the calling function spells it.

**Developer.** ``bool`` is rejected before the integral check because it is an
``int`` subclass, so ``True`` would otherwise pass as position ``1``. Negative
positions are rejected rather than wrapped: Python's own sequences wrap, but a
row identifier is not a sequence offset, and silently resolving ``-1`` to the
last row turns a caller's off-by-one into a plausible wrong answer.

Each submodule that needs this contract carries its own byte-identical copy, so
no submodule depends on another to validate its own arguments. The copies are
kept identical by ``scikitplot/tests/test_submodule_independence.py``, which
compares their digests: duplication is only safe while it is enforced, and the
failure this contract exists to prevent -- one concept answered two different
ways -- is exactly what silent drift between copies would reintroduce.

Only the standard library is imported, so this module can live anywhere.
"""

from __future__ import annotations

from typing import Any

__all__ = ["require_count", "require_index"]


def _as_integral(value: Any, name: str) -> int:
    """
    Return ``value`` as an ``int``, refusing anything that is not integral.

    Parameters
    ----------
    value : object
        Candidate value. Objects exposing ``__index__`` — NumPy integers among
        them — are converted.
    name : str
        Parameter name as the calling function spells it, used in the message.

    Returns
    -------
    int
        The integral value.

    Raises
    ------
    TypeError
        If the value is a ``bool``, or is not integral. ``bool`` is checked
        first because it is an ``int`` subclass.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be an integer, got bool {value!r}; True is not a "
            "position or a count."
        )
    if isinstance(value, int):
        return value
    index = getattr(value, "__index__", None)
    if index is None:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__} {value!r}."
        )
    return index()


def require_index(value: Any, count: int, *, name: str = "index") -> int:
    """
    Return ``value`` as a position valid in ``[0, count)``.

    Parameters
    ----------
    value : object
        Candidate position.
    count : int
        Number of rows available.
    name : str, optional
        Parameter name as the calling function spells it.

    Returns
    -------
    int
        The validated position.

    Raises
    ------
    TypeError
        If the value is not an integer, or is a ``bool``.
    IndexError
        If the position is negative or not below ``count``. Negative values are
        refused rather than counted from the end.

    Examples
    --------
    >>> require_index(2, 5)
    2
    >>> require_index(-1, 5)
    Traceback (most recent call last):
        ...
    IndexError: index -1 is outside [0, 5); negative positions are not counted from the end.
    """
    position = _as_integral(value, name)
    if position < 0:
        raise IndexError(
            f"{name} {position} is outside [0, {count}); negative positions are "
            "not counted from the end."
        )
    if position >= count:
        raise IndexError(f"{name} {position} is outside [0, {count}).")
    return position


def require_count(
    value: Any, *, maximum: int | None = None, name: str = "count"
) -> int:
    """
    Return ``value`` as a non-negative count within its declared maximum.

    Parameters
    ----------
    value : object
        Candidate count.
    maximum : int or None, optional
        Declared ceiling. ``None`` means the caller declares no ceiling, which
        is not the same as a ceiling of zero.
    name : str, optional
        Parameter name as the calling function spells it.

    Returns
    -------
    int
        The validated count. Zero is valid: it is a request for nothing.

    Raises
    ------
    TypeError
        If the value is not an integer, or is a ``bool``.
    ValueError
        If the count is negative, or above ``maximum``.

    Examples
    --------
    >>> require_count(5, maximum=20)
    5
    >>> require_count(-1)
    Traceback (most recent call last):
        ...
    ValueError: count -1 must be zero or greater.
    """
    number = _as_integral(value, name)
    if number < 0:
        raise ValueError(f"{name} {number} must be zero or greater.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} {number} must be at most {maximum}.")
    return number
