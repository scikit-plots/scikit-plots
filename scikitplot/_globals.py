# scikitplot/_globals.py
#
# flake8: noqa: D213
# pylint: disable=import-outside-toplevel
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# This module was copied from the numpy project.
# https://github.com/numpy/numpy/blob/main/numpy/_globals.py

"""
Module defining global singleton classes.

This module raises a :exc:`RuntimeError` if an attempt to reload it is made.
In that way the identities of the classes defined here are fixed and will
remain so even if scikit-plots itself is reloaded.  In particular, a function
like the following will still work correctly after a reload::

    def foo(arg=skplt._NoValue):
        if arg is skplt._NoValue:
            ...

That was not the case when the singleton classes were defined in the package
``__init__.py`` file.  See numpy/numpy#7844 for a discussion of the reload
problem that motivated this design.

Notes
-----
**Developer notes**

* ``SingletonBase.__init_subclass__`` stamps ``_instance = None`` directly
  onto every subclass at class-creation time.  This makes each subclass's own
  ``_instance`` slot unconditionally authoritative, eliminating the latent bug
  where ``SingletonBase._instance`` (shared via MRO) could be mistaken for a
  subclass's own value if ``SingletonBase`` itself were ever instantiated.

* ``SingletonBase.__new__`` accepts no construction arguments beyond ``cls``.
  Singleton objects carry no state and must not be parameterised at
  instantiation time.  Forwarding ``*args`` / ``**kwargs`` to
  ``object.__new__`` is incorrect: ``object.__new__`` rejects extra arguments
  in Python 3 when ``__init__`` is not customised, and doing so would also
  silently accept nonsensical arguments without error.

* ``_CopyMode.__bool__`` deliberately raises :exc:`ValueError` for
  ``IF_NEEDED`` to preserve backwards-compatible behaviour inherited from
  NumPy.
"""

from __future__ import annotations

import enum
from typing import Self

from ._utils import set_module as _set_module

__all__ = [
    "_CopyMode",
    "_Default",
    "_Deprecated",
    "_NoValue",
]

######################################################################
## Disallow Reloading Module
######################################################################

# Disallow reloading this module so as to preserve the identities of the
# classes defined here.
if "_is_loaded" in globals():
    raise RuntimeError("Reloading scikitplot._globals is not allowed")
_is_loaded = True

######################################################################
## Copy modes supported
######################################################################


@_set_module("scikitplot")
class _CopyMode(enum.Enum):
    """An enumeration for the copy modes supported by ``numpy.copy()`` and ``numpy.array()``.

    The following three modes are supported:

    - ``ALWAYS``:    A deep copy of the input array is always taken.
    - ``IF_NEEDED``: A deep copy is taken only if necessary.
    - ``NEVER``:     A deep copy is never taken.  If a copy cannot be avoided
                     a :exc:`ValueError` is raised.

    Note that the buffer-protocol could in theory perform copies.  NumPy
    currently assumes an object exporting the buffer protocol will never do
    this.
    """

    ALWAYS = True
    NEVER = False
    IF_NEEDED = 2

    def __bool__(self) -> bool:
        """Return the boolean interpretation of this copy mode.

        Returns
        -------
        bool
            ``True`` for ``ALWAYS``, ``False`` for ``NEVER``.

        Raises
        ------
        ValueError
            If the mode is ``IF_NEEDED``, which has no unambiguous boolean
            interpretation.

        Notes
        -----
        The ``IF_NEEDED`` branch raises :exc:`ValueError` intentionally for
        backwards compatibility with NumPy.
        """
        if self == _CopyMode.ALWAYS:
            return True
        if self == _CopyMode.NEVER:
            return False
        raise ValueError(f"{self} is neither True nor False.")


######################################################################
## SingletonBase
######################################################################


class SingletonBase:
    """Base class implementing the singleton pattern.

    Ensures that at most one instance of each concrete subclass exists.
    Subsequent instantiation calls return the already-created instance.

    Attributes
    ----------
    _instance : SingletonBase or None
        Holds the single instance of each concrete subclass.  ``None`` before
        first instantiation.  Each subclass receives its *own* ``_instance``
        attribute (stamped by :meth:`__init_subclass__`) so that subclasses
        never accidentally share state through MRO lookup.

    Notes
    -----
    **User note** — do not subclass this in application code; it is an
    internal scikit-plots implementation detail.

    **Developer note — instance isolation** — ``_instance = None`` is
    declared on ``SingletonBase`` as a sentinel, but :meth:`__init_subclass__`
    stamps a fresh ``_instance = None`` directly onto every subclass at
    *class-creation time*.  This guarantees that ``cls._instance`` in
    :meth:`__new__` always resolves to the subclass's own attribute, never
    to ``SingletonBase._instance`` via MRO fallback.  Without this,
    calling ``SingletonBase()`` before any subclass would leave
    ``SingletonBase._instance`` non-``None``, causing all uninitialised
    subclasses to return the wrong instance on their first call.

    **Developer note — no construction arguments** — ``__new__`` intentionally
    accepts no ``*args`` / ``**kwargs``.  Singletons carry no per-instance
    state and must not be parameterised.  Forwarding extra arguments to
    ``object.__new__`` raises :exc:`TypeError` in Python 3 when ``__init__``
    is not customised, and would also silently swallow nonsensical call sites.
    """

    _instance: SingletonBase | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        """Stamp a fresh ``_instance = None`` onto every subclass.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`super().__init_subclass__` to support
            cooperative multiple inheritance.

        Notes
        -----
        Called automatically by Python when a class statement that inherits
        from ``SingletonBase`` is executed.  The stamp makes
        ``cls._instance`` an attribute of the *subclass itself*, preventing
        MRO-based fallback to ``SingletonBase._instance`` in
        :meth:`__new__`.
        """
        super().__init_subclass__(**kwargs)
        cls._instance = None

    def __new__(cls) -> Self:
        """Return the sole instance of *cls*, creating it on first call.

        Parameters
        ----------
        cls : type[SingletonBase]
            The class being instantiated.

        Returns
        -------
        SingletonBase
            The single instance of *cls*.

        Notes
        -----
        No construction arguments are accepted.  Singletons carry no
        per-instance state; parameterisation would violate the pattern.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self) -> tuple:
        """Support pickling while preserving singleton identity.

        Returns
        -------
        tuple
            ``(type(self), ())`` — instructs :mod:`pickle` to call
            ``type(self)()`` during unpickling, which returns the existing
            singleton instance rather than creating a new object.
        """
        return (self.__class__, ())


######################################################################
## _DefaultType
######################################################################


class _DefaultType(SingletonBase):
    """Singleton sentinel representing "use the default value".

    Used to distinguish between a caller explicitly passing a value and
    relying on the function's built-in default behaviour.

    Examples
    --------
    >>> default = _DefaultType()
    >>> print(default)
    <default>

    >>> another_default = _DefaultType()
    >>> default is another_default  # Singleton: always the same instance.
    True
    """

    def __repr__(self) -> str:
        """Return ``'<default>'``.

        Returns
        -------
        str
            The human-readable sentinel label.
        """
        return "<default>"


# Module-level singleton instance for direct use.
_Default = _DefaultType()


######################################################################
## _DeprecatedType
######################################################################


class _DeprecatedType(SingletonBase):
    """Singleton sentinel indicating a deprecated value or feature.

    Use this to signal that a parameter or feature is no longer recommended
    and may be removed in a future release.

    Examples
    --------
    >>> deprecated = _DeprecatedType()
    >>> print(deprecated)
    <deprecated>

    >>> another_deprecated = _DeprecatedType()
    >>> deprecated is another_deprecated  # Singleton: always the same instance.
    True
    """

    def __repr__(self) -> str:
        """Return ``'<deprecated>'``.

        Returns
        -------
        str
            The human-readable sentinel label.
        """
        return "<deprecated>"


# Module-level singleton instance for direct use.
_Deprecated = _DeprecatedType()


######################################################################
## _NoValueType
######################################################################


class _NoValueType(SingletonBase):
    """Singleton sentinel indicating no user-supplied value.

    Provides a unique marker to detect whether a caller provided a value or
    whether default behaviour should apply.  Prefer this over ``None`` when
    ``None`` is itself a meaningful argument value.

    Common use cases:

    - A new keyword is added to a function that forwards its inputs to another
      function or method defined outside scikit-plots.  Downstream libraries
      may not yet support the keyword, so forwarding it unconditionally could
      break previously working code.
    - A keyword is being deprecated and a deprecation warning should only be
      emitted when the keyword is explicitly used.

    Examples
    --------
    >>> no_value = _NoValueType()
    >>> print(no_value)
    <no value>

    >>> another_instance = _NoValueType()
    >>> no_value is another_instance  # Singleton: always the same instance.
    True
    """

    def __repr__(self) -> str:
        """Return ``'<no value>'``.

        Returns
        -------
        str
            The human-readable sentinel label.
        """
        return "<no value>"


# Module-level singleton instance for direct use.
_NoValue = _NoValueType()
