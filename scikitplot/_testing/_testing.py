# scikitplot/_testing/_testing.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Core testing utilities: warning suppression and assertion helpers.

This module is **private** (underscore-prefixed) and is intended to be
embedded as a sub-package inside a larger library.  Public symbols are
re-exported through the parent package ``__init__.py``.

Notes
-----
Developer note: Keep all imports at module scope minimal.  This file is
imported early during test collection; heavy dependencies slow discovery and
create hard-to-debug circular import chains.
"""

from __future__ import annotations

import functools
import unittest
import warnings

__all__ = [
    "SkipTest",
    "assert_no_warnings",
    "ignore_warnings",
]

# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

#: Alias for :class:`unittest.case.SkipTest` so callers can import it from
#: one consistent location without depending on ``unittest`` directly.
SkipTest = unittest.case.SkipTest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _IgnoreWarnings:
    """Context manager and decorator that suppresses warnings of a given category.

    Uses :func:`warnings.catch_warnings` internally so that filter state is
    restored correctly on exit — even when an exception propagates — and to
    remain safe under threading.

    Parameters
    ----------
    category : type, default=Warning
        :class:`Warning` subclass to suppress.  Defaults to the root
        :class:`Warning` class, which silences **all** warning categories.

    Raises
    ------
    TypeError
        If *category* is not a :class:`Warning` subclass.

    Notes
    -----
    User note: Prefer the :func:`ignore_warnings` factory to construct
    instances; it validates arguments and provides ergonomic shorthand forms.

    Developer note: Earlier versions manually mutated ``warnings.filters``
    and ``warnings.showwarning`` directly.  That approach is not thread-safe
    and does not compose under nesting.  This implementation delegates to
    :func:`warnings.catch_warnings` for both the context-manager and
    decorator code paths, guaranteeing identical, correct behaviour.

    Sequential reuse (exit then re-enter the *same* instance) is supported.
    Concurrent or nested reuse of the *same* instance is rejected with
    :class:`RuntimeError`.
    """

    def __init__(self, category: type = Warning) -> None:
        if not (isinstance(category, type) and issubclass(category, Warning)):
            raise TypeError(f"'category' must be a Warning subclass, got {category!r}.")
        self.category = category
        self._entered: bool = False
        self._catch_warnings: warnings.catch_warnings | None = None

    # ------------------------------------------------------------------
    # Decorator protocol
    # ------------------------------------------------------------------

    def __call__(self, fn: object) -> object:
        """Wrap *fn* so that matching warnings are suppressed on every call.

        Parameters
        ----------
        fn : callable
            The function to wrap.

        Returns
        -------
        wrapper : callable
            A new callable with the same signature and metadata as *fn*.

        Raises
        ------
        TypeError
            If *fn* is not callable.
        """
        if not callable(fn):
            raise TypeError(f"'fn' must be callable, got {type(fn).__name__!r}.")

        @functools.wraps(fn)
        def wrapper(*args: object, **kwargs: object) -> object:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", self.category)
                return fn(*args, **kwargs)

        return wrapper

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "_IgnoreWarnings":
        """Activate warning suppression.

        Returns
        -------
        self : _IgnoreWarnings
            Returns *self* so ``with iw as ctx`` binds correctly.

        Raises
        ------
        RuntimeError
            If this instance is already active (re-entrant use on the same
            instance).
        """
        if self._entered:
            raise RuntimeError(
                f"Cannot enter {self!r} twice. "
                "Create a new instance for nested suppression."
            )
        self._entered = True
        self._catch_warnings = warnings.catch_warnings()
        self._catch_warnings.__enter__()
        warnings.simplefilter("ignore", self.category)
        return self  # BUG FIX: was missing — caused `with iw as ctx` to set ctx=None

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool | None:
        """Restore filter state and deactivate the context.

        Raises
        ------
        RuntimeError
            If ``__exit__`` is called without a prior ``__enter__``.
        """
        if not self._entered:
            raise RuntimeError(f"Cannot exit {self!r} without entering first.")
        try:
            return self._catch_warnings.__exit__(exc_type, exc_val, exc_tb)
        finally:
            # BUG FIX: reset flag so sequential reuse of the same instance works.
            self._entered = False
            self._catch_warnings = None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{type(self).__name__}(category={self.category.__name__})"


class _AssertNoWarningsContext:
    """Context manager that raises :class:`AssertionError` if warnings are emitted.

    Parameters
    ----------
    warning_class : type, default=Warning
        :class:`Warning` subclass to watch.  Any warning whose category is a
        subclass of *warning_class* triggers an :class:`AssertionError` on
        context exit.

    Raises
    ------
    TypeError
        If *warning_class* is not a :class:`Warning` subclass.

    Notes
    -----
    Developer note: ``record=True`` is passed to :func:`warnings.catch_warnings`
    so warnings are captured, not emitted.  ``simplefilter("always")`` ensures
    every occurrence is recorded, including repeats that ``"once"`` would hide.
    """

    def __init__(self, warning_class: type = Warning) -> None:
        if not (isinstance(warning_class, type) and issubclass(warning_class, Warning)):
            raise TypeError(
                f"'warning_class' must be a Warning subclass, "
                f"got {warning_class!r}."
            )
        self.warning_class = warning_class
        self._entered: bool = False
        self._manager: warnings.catch_warnings | None = None
        self._recorded: list[warnings.WarningMessage] | None = None

    def __enter__(self) -> "_AssertNoWarningsContext":
        """Activate warning capture.

        Returns
        -------
        self : _AssertNoWarningsContext

        Raises
        ------
        RuntimeError
            On re-entrant use of the same instance.
        """
        if self._entered:
            raise RuntimeError(f"Cannot enter {self!r} twice.")
        self._entered = True
        self._manager = warnings.catch_warnings(record=True)
        self._recorded = self._manager.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        """Restore state and assert no matching warnings were emitted.

        Raises
        ------
        RuntimeError
            If ``__exit__`` is called without ``__enter__``.
        AssertionError
            If one or more warnings matching :attr:`warning_class` were raised.
        """
        if not self._entered:
            raise RuntimeError(f"Cannot exit {self!r} without entering first.")
        try:
            self._manager.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._entered = False

        # Do not suppress exceptions raised inside the block.
        if exc_type is not None:
            return False

        relevant = [
            w
            for w in (self._recorded or [])
            if issubclass(w.category, self.warning_class)
        ]
        if relevant:
            messages = "\n  ".join(
                f"{w.category.__name__}: {w.message}" for w in relevant
            )
            raise AssertionError(
                f"Expected no {self.warning_class.__name__} warnings, "
                f"but got {len(relevant)}:\n  {messages}"
            )
        return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(warning_class={self.warning_class.__name__})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ignore_warnings(
    obj: object = None,
    *,
    category: type = Warning,
) -> object:
    """Context manager and decorator to suppress warnings.

    Three usage modes are supported:

    1. **Context manager** — suppress warnings for a ``with`` block.
    2. **Decorator factory** — ``@ignore_warnings(category=…)`` wraps a
       function so warnings are suppressed on every call.
    3. **Immediate wrap** — ``ignore_warnings(fn)`` is shorthand for
       ``ignore_warnings()(fn)``.

    Parameters
    ----------
    obj : callable or None, default=None
        When ``None`` (default), returns a :class:`_IgnoreWarnings` instance
        usable as both a context manager and a decorator.  When *obj* is a
        callable, wraps it immediately and returns the wrapper.
    category : type, default=Warning
        The :class:`Warning` subclass to suppress.  Keyword-only to prevent
        the common mistake of passing a warning class positionally.

    Returns
    -------
    wrapped : callable
        When *obj* is callable: the wrapped function.
    ctx : _IgnoreWarnings
        When *obj* is ``None``: a reusable context-manager / decorator.

    Raises
    ------
    ValueError
        If *obj* is a :class:`Warning` subclass (common positional mistake).
    TypeError
        If *obj* is neither ``None`` nor a callable.

    See Also
    --------
    assert_no_warnings : Assert that a block emits *no* warnings.

    Notes
    -----
    User note: Suppression affects the entire process for the duration of the
    block or call.  Tests verifying cross-module warning behaviour should not
    use this utility.

    Developer note: ``category`` is keyword-only (the ``*`` separator) to
    prevent silent test no-ops caused by positional misuse.

    Examples
    --------
    Context manager::

        >>> import warnings
        >>> from _testing import ignore_warnings
        >>> with ignore_warnings():
        ...     warnings.warn("suppressed")

    Decorator factory::

        >>> @ignore_warnings(category=DeprecationWarning)
        ... def noisy():
        ...     warnings.warn("gone", DeprecationWarning)
        >>> noisy()  # No warning emitted.

    Immediate wrap::

        >>> def noisy():
        ...     warnings.warn("gone")
        >>> ignore_warnings(noisy)()  # No warning emitted.
    """
    if isinstance(obj, type) and issubclass(obj, Warning):
        raise ValueError(
            f"'obj' must be a callable, not a warning class.  "
            f"You passed {obj.__name__!r}.  "
            f"To filter by category use: "
            f"ignore_warnings(category={obj.__name__})"
        )
    if obj is not None and not callable(obj):
        raise TypeError(
            f"'obj' must be a callable or None, got {type(obj).__name__!r}."
        )

    instance = _IgnoreWarnings(category=category)
    if callable(obj):
        return instance(obj)
    return instance


def assert_no_warnings(warning_class: type = Warning) -> _AssertNoWarningsContext:
    """Return a context manager that raises if any matching warnings are emitted.

    Parameters
    ----------
    warning_class : type, default=Warning
        :class:`Warning` subclass to watch for.  Defaults to :class:`Warning`,
        which catches *all* warning categories.

    Returns
    -------
    ctx : _AssertNoWarningsContext
        A context manager that raises :class:`AssertionError` on exit if any
        warning matching *warning_class* was raised inside the block.

    Raises
    ------
    TypeError
        If *warning_class* is not a :class:`Warning` subclass.

    See Also
    --------
    ignore_warnings : Suppress warnings without asserting.

    Examples
    --------
    >>> import warnings
    >>> from _testing import assert_no_warnings
    >>> with assert_no_warnings():
    ...     pass  # No warning → passes silently.

    >>> with assert_no_warnings(DeprecationWarning):
    ...     warnings.warn("oops", DeprecationWarning)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    AssertionError: Expected no DeprecationWarning warnings, but got 1: ...
    """
    return _AssertNoWarningsContext(warning_class=warning_class)
