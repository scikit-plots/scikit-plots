# pylint: disable=import-outside-toplevel

# This module was copied from the numpy project.
# https://github.com/numpy/numpy/blob/main/numpy/_globals.py

"""
Module defining global singleton classes.

This module raises a RuntimeError if an attempt to reload it is made. In that
way the identities of the classes defined here are fixed and will remain so
even if numpy itself is reloaded. In particular, a function like the following
will still work correctly after numpy is reloaded::

    def foo(arg=np._NoValue):
        if arg is np._NoValue:
            ...

That was not the case when the singleton classes were defined in the numpy
``__init__.py`` file. See gh-7844 for a discussion of the reload problem that
motivated this module.

"""

import enum

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from typing import Optional
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
## copy modes supported
######################################################################


@_set_module("scikitplot")
class _CopyMode(enum.Enum):
    """
    An enumeration for the copy modes supported by numpy.copy() and numpy.array().

    The following three modes are supported,

    - ALWAYS: This means that a deep copy of the input
              array will always be taken.
    - IF_NEEDED: This means that a deep copy of the input
                 array will be taken only if necessary.
    - NEVER: This means that the deep copy will never be taken.
             If a copy cannot be avoided then a `ValueError` will be
             raised.

    Note that the buffer-protocol could in theory do copies.  NumPy currently
    assumes an object exporting the buffer protocol will never do this.

    """

    ALWAYS = True
    NEVER = False
    IF_NEEDED = 2

    def __bool__(self):
        # For backwards compatibility
        if self == _CopyMode.ALWAYS:
            return True

        if self == _CopyMode.NEVER:
            return False

        raise ValueError(f"{self} is neither True nor False.")


######################################################################
## SingletonBase class
# Inherits from object (or implicitly from object).
# Supports pickling by implementing __reduce__.
# Purpose: A base class that implements the Singleton pattern,
# ensuring that only one instance of the class exists.
# It overrides the __new__ method to control instance creation.
# Intended Use: Used for general singleton objects—i.e.,
# classes that need to ensure that only one instance of the class is created.
# Use: when you need a singleton class with general behavior,
# such as managing single resources, configurations, or services.
# SingletonBase is ideal for simple singleton patterns
# where you don't require predefined values or enum-like behavior.
# It is simple, flexible, and easy to implement.
######################################################################


class SingletonBase:
    """
    A base class for implementing singleton pattern objects.

    This class ensures that only one instance of a class exists. It overrides
    the `__new__` method to ensure that only a single instance is created. Any
    subsequent attempts to create an instance of the class return the same instance.

    Attributes
    ----------
    _instance : Optional[SingletonBase]
        The single instance of the class, initialized to `None`.

    Methods
    -------
    __new__(cls) -> SingletonBase
        Ensures that only one instance of the class is created, implementing the singleton pattern.
    __reduce__(self) -> tuple
        Ensures that the singleton instance is preserved during pickling and unpickling.

    """

    # Class attribute to hold the single instance of the class.
    # _instance: Union["SingletonBase", None] = None
    _instance: "SingletonBase | None" = None

    # magic method to get called in an objects instantiation.
    def __new__(cls: "type[SingletonBase]", *args, **kwargs) -> "SingletonBase":
        """
        Override the default object creation method to implement the singleton pattern.

        Ensures that only one instance of the class is created. If an instance already exists,
        that instance is returned instead of creating a new one.

        Parameters
        ----------
        cls : type[SingletonBase]
            The class being instantiated.
        *args : tuple
            Positional arguments to be passed to the class constructor.
        **kwargs : dict
            Keyword arguments to be passed to the class constructor.

        Returns
        -------
        SingletonBase
            The single instance of the class.

        """
        # ensure that only one instance exists
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __reduce__(self) -> tuple:
        """
        Ensure the singleton instance is correctly restored during object serialization.

        (i.e., pickling and unpickling). This method ensures that the singleton behavior
        is maintained during serialization.

        Returns
        -------
        tuple
            A tuple of (class, args) used to restore the singleton instance during unpickling.

        """
        return (self.__class__, ())


######################################################################
## Singleton special keyword types
## _DefaultType class
######################################################################


class _DefaultType(SingletonBase):
    """
    A special keyword value representing the use of a default value.

    This class is used to indicate that a parameter is set to its default value.
    It helps to distinguish between cases where the user intentionally provided
    a value and cases where the default behavior is used.

    Examples
    --------
    >>> default = _DefaultType()
    >>> print(default)
    <default>

    >>> another_default = _DefaultType()
    >>> default is another_default  # Singleton behavior ensures one instance
    True

    """

    # print our string object
    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Printing <default> makes it easy to spot when the default value is used.

        Returns
        -------
        str
            The string representation of the default object.

        """
        return "<default>"


# Create class instance to direct use
_Default = _DefaultType()

######################################################################
## Singleton Marker Types
## _DeprecatedType class
######################################################################


class _DeprecatedType(SingletonBase):
    """
    A special keyword value indicating that a value or feature is deprecated.

    This class is useful to signal that a parameter or feature is no longer recommended
    for use and may be removed in future versions of code or APIs.

    Examples
    --------
    >>> deprecated = _DeprecatedType()
    >>> print(deprecated)
    <deprecated>

    >>> another_deprecated = _DeprecatedType()
    >>> deprecated is another_deprecated  # Ensures singleton behavior
    True

    """

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        Printing <deprecated> makes it easy to identify deprecated values or features.

        Returns
        -------
        str
            The string representation of the deprecated object.

        """
        return "<deprecated>"


# Create class instance to direct use
_Deprecated = _DeprecatedType()

######################################################################
## Singleton Marker Types
## _NoValueType class
######################################################################


class _NoValueType(SingletonBase):
    """
    A special keyword value indicating no user-defined input.

    This class provides a unique marker to detect whether a user has provided
    a value to a function or if a default behavior should be applied.

    The instance of this class may be used as the default value assigned to a
    keyword if no other obvious default (e.g., `None`) is suitable,

    Common reasons for using this keyword are:

    - A new keyword is added to a function, and that function forwards its
      inputs to another function or method which can be defined outside of
      NumPy. For example, ``np.std(x)`` calls ``x.std``, so when a ``keepdims``
      keyword was added that could only be forwarded if the user explicitly
      specified ``keepdims``; downstream array libraries may not have added
      the same keyword, so adding ``x.std(..., keepdims=keepdims)``
      unconditionally could have broken previously working code.
    - A keyword is being deprecated, and a deprecation warning must only be
      emitted when the keyword is used.

    Examples
    --------
    >>> no_value = _NoValueType()
    >>> print(no_value)
    <no value>

    >>> another_instance = _NoValueType()
    >>> no_value is another_instance  # All instances are the same
    True

    """

    def __repr__(self) -> str:
        """
        Return a string representation of the object.

        This helps in debugging and makes it clear when <no value> is printed.

        Returns
        -------
        str
            The string representation of the no value object.

        """
        return "<no value>"


# Create class instance to direct use
_NoValue = _NoValueType()

######################################################################
##
######################################################################
