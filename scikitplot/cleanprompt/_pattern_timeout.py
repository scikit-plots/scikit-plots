"""
A per-document time limit for the patterns of custom packs.

Python's :mod:`re` has no time limit: a pattern with a backtracking shape can
take seconds, then minutes, on one near-match, and nothing interrupts it. The
pattern-risk check (:mod:`._pattern_risk`) reports the shapes it recognises
when a pack loads, but a report is not a bound. When a time limit is set and
the optional ``regex`` package is installed (the ``timeout`` tier), the
patterns of custom packs run on that engine instead, with a deadline per
document; running past it fails the document with
:class:`~._exceptions.PatternTimeoutError`.

Notes
-----
**User notes.** Set the limit with ``--pattern-timeout SECONDS``,
``FluentCleanPrompt().pattern_timeout(seconds)``, ``pattern_timeout`` in a
plan file, or ``CLEANPROMPT_PATTERN_TIMEOUT`` for a machine or a CI job. It
applies to custom packs only: the built-in patterns keep :mod:`re`, and the
test suite pins their behaviour. Without a custom pack the limit has nothing
to bound and is not checked; with one, a limit and no ``regex`` is an error,
never a silent run without the bound. ``doctor`` reports the ``timeout`` tier.

The limit is one budget for the whole document: every custom pattern, every
chunk of a long file. A document that runs past it is not cleaned at all, so
nothing half-redacted is returned. In ``batch`` that file fails and the others
continue; the MCP server and the web app fail that request and keep serving.

**Developer notes.** ``regex`` in its ``V0`` mode follows :mod:`re` syntax, but
not everywhere, so every custom pattern is compiled again on it when the
cleaner is built, and one it rejects stops the build naming its pack and kind.
Flags are mapped by name, not by value: :data:`re.ASCII` is 256, which in
``regex`` is ``V1``, a different syntax; passing ``re`` flag bits through would
silently change what a pattern means.

The deadline lives in a :class:`contextvars.ContextVar` set around each
document (:func:`pattern_deadline`), so threads and concurrent requests each
have their own. A detector used outside a document gets a fresh deadline for
each scan. ``regex`` raises the built-in :class:`TimeoutError`, which is an
:class:`OSError`; it is converted at once, so no caller can mistake it for an
unreadable file, and the exception chain is dropped so the scanned text cannot
appear in a traceback.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable

from ._capabilities import require
from ._exceptions import PatternError, PatternTimeoutError

__all__ = [
    "PATTERN_TIMEOUT_ENV",
    "BoundedPattern",
    "active_deadline",
    "bounded_compile",
    "check_seconds",
    "pattern_deadline",
    "resolve_pattern_timeout",
]

#: Environment variable that sets the limit for a machine or a CI job.
PATTERN_TIMEOUT_ENV = "CLEANPROMPT_PATTERN_TIMEOUT"

#: The deadline of the document being cleaned (``time.monotonic`` seconds).
_DEADLINE: ContextVar[float | None] = ContextVar(
    "cleanprompt_pattern_deadline", default=None
)

#: ``re`` flags a custom pattern may carry, by name (see the module notes).
_FLAG_NAMES = ("IGNORECASE", "MULTILINE", "DOTALL", "VERBOSE", "ASCII", "UNICODE")


def check_seconds(seconds: Any, name: str = "pattern_timeout") -> float:
    """
    Return ``seconds`` as a float, refusing anything but a positive finite number.

    Parameters
    ----------
    seconds : int or float
        The limit.
    name : str, default='pattern_timeout'
        Named in the message.

    Returns
    -------
    float
        The limit.

    Raises
    ------
    ValueError
        For a bool, a non-number, zero, a negative number, NaN or infinity.
    """
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        msg = f"{name} must be a positive number of seconds, got {seconds!r}"
        raise ValueError(msg)
    return float(seconds)


def resolve_pattern_timeout(seconds: float | None = None) -> float | None:
    """
    Return the limit in force: ``seconds``, else the environment, else none.

    Parameters
    ----------
    seconds : float or None, default=None
        An explicit limit. ``None`` defers to ``CLEANPROMPT_PATTERN_TIMEOUT``.

    Returns
    -------
    float or None
        The limit in seconds, or ``None`` when no limit is set (the variable
        unset or empty).

    Raises
    ------
    ValueError
        If the limit, given or from the environment, is not a positive number
        of seconds.
    """
    if seconds is not None:
        return check_seconds(seconds)
    raw = os.environ.get(PATTERN_TIMEOUT_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        msg = f"{PATTERN_TIMEOUT_ENV}={raw!r} is not a number of seconds"
        raise ValueError(msg) from None
    return check_seconds(value, PATTERN_TIMEOUT_ENV)


def bounded_compile(spec: Any, pack: str) -> Any:
    """
    Compile one custom pattern on the time-limited engine.

    Parameters
    ----------
    spec : PatternSpec
        The pattern, already valid under :mod:`re`.
    pack : str
        Its pack, named in messages.

    Returns
    -------
    regex.Pattern
        The pattern, in ``V0`` (``re``-compatible) mode, with the same flags.

    Raises
    ------
    CapabilityError
        If the ``timeout`` tier (the ``regex`` package) is not usable.
    PatternError
        If the pattern carries a flag that has no counterpart, or the engine
        rejects its syntax.
    """
    require("timeout")
    import regex  # ruff: ignore[import-outside-top-level]  # optional tier

    flags = regex.V0
    remaining = spec.flags
    for name in _FLAG_NAMES:
        bit = int(getattr(re, name))
        if remaining & bit:
            flags |= getattr(regex, name)
            remaining &= ~bit
    where = f"pattern {spec.kind} of pack {pack!r}"
    if remaining:
        msg = (
            f"{where} cannot run under the pattern time limit: flag value "
            f"{remaining} has no counterpart in the time-limited engine; remove "
            "the flag or the limit"
        )
        raise PatternError(msg, name=spec.kind)
    try:
        return regex.compile(spec.pattern, flags=flags)
    except (regex.error, ValueError, OverflowError, MemoryError) as exc:
        msg = (
            f"{where} cannot run under the pattern time limit: the time-limited "
            f"engine rejects it ({exc}); rewrite it so both engines accept it, "
            "or remove the limit"
        )
        raise PatternError(msg, name=spec.kind) from None


@contextmanager
def pattern_deadline(seconds: float | None) -> Iterator[None]:
    """
    Give the document cleaned inside this block one time budget.

    Parameters
    ----------
    seconds : float or None
        The limit; ``None`` sets no deadline.

    Yields
    ------
    None

    Notes
    -----
    **Developer notes.** The outermost block wins: a document whose parts are
    cleaned through nested calls (an extracted artifact, a chunked file) keeps
    the one budget it started with.
    """
    if seconds is None or _DEADLINE.get() is not None:
        yield
        return
    token = _DEADLINE.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def active_deadline(seconds: float) -> float:
    """Return the current document's deadline, or one starting now."""
    deadline = _DEADLINE.get()
    return deadline if deadline is not None else time.monotonic() + seconds


class BoundedPattern:
    """
    The scanning calls of one compiled pattern, each limited by one deadline.

    Parameters
    ----------
    pattern : regex.Pattern
        From :func:`bounded_compile`.
    deadline : float
        ``time.monotonic`` seconds.
    expired : callable
        Returns the :class:`~._exceptions.PatternTimeoutError` to raise.

    Notes
    -----
    **Developer notes.** Offers exactly what
    :meth:`~._detectors.RegexDetector.detect` calls: ``search``, ``match`` and
    ``finditer``. Each call gets the time left; none is made once it is gone.
    """

    __slots__ = ("_deadline", "_expired", "_pattern")

    def __init__(
        self,
        pattern: Any,
        deadline: float,
        expired: Callable[[], PatternTimeoutError],
    ) -> None:
        self._pattern = pattern
        self._deadline = deadline
        self._expired = expired

    def _left(self) -> float:
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise self._expired()
        return left

    def search(self, text: str, pos: int = 0, endpos: int | None = None) -> Any:
        """Search from ``pos``; see :meth:`re.Pattern.search`."""
        try:
            return self._pattern.search(text, pos, endpos, timeout=self._left())
        except TimeoutError:
            raise self._expired() from None

    def match(self, text: str, pos: int = 0, endpos: int | None = None) -> Any:
        """Match at ``pos``; see :meth:`re.Pattern.match`."""
        try:
            return self._pattern.match(text, pos, endpos, timeout=self._left())
        except TimeoutError:
            raise self._expired() from None

    def finditer(self, text: str) -> Iterator[Any]:
        """Yield every match; see :meth:`re.Pattern.finditer`."""
        matches = self._pattern.finditer(text, timeout=self._left())
        while True:
            try:
                found = next(matches)
            except StopIteration:
                return
            except TimeoutError:
                raise self._expired() from None
            self._left()
            yield found
