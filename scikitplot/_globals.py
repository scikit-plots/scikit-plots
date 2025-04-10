# This module was copied from the numpy project.
"""
Module defining global singleton classes.

This module raises a RuntimeError if an attempt to reload it is made. In that
way the identities of the classes defined here are fixed and will remain so
even if scikitplot itself is reloaded. In particular, a function like the following
will still work correctly after scikitplot is reloaded::

    def foo(arg=np._NoValue):
        if arg is np._NoValue:
            ...

That was not the case when the singleton classes were defined in the scikitplot
``__init__.py`` file. See gh-7844 for a discussion of the reload problem that
motivated this module.
"""

from typing import Optional, Type

# from typing import TYPE_CHECKING

__all__ = [
    "ModuleDeprecationWarning",
    "VisibleDeprecationWarning",
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
## ModuleDeprecationWarning class
######################################################################


class ModuleDeprecationWarning(DeprecationWarning):
    """
    Module deprecation warning class.

    This custom warning class is used to signal the deprecation of an entire module.
    The `nose` testing framework treats ordinary `DeprecationWarning` as test failures,
    which makes it challenging to deprecate whole modules. To address this, this special
    `ModuleDeprecationWarning` is defined, which the `nose` tester will allow without
    causing test failures.

    This is especially useful when deprecating entire modules or submodules without
    breaking existing tests.

    Attributes
    ----------
    __module__ : str
        The module in which this warning is defined, set to 'scikitplot'.

    Methods
    -------
    __module__
        A string representing the module that contains this warning.

    """

    # Set the module for the warning to 'scikitplot'
    __module__: str = "scikitplot"


# ModuleDeprecationWarning.__module__ = 'scikitplot'

######################################################################
## VisibleDeprecationWarning class
######################################################################


class VisibleDeprecationWarning(UserWarning):
    """
    Visible deprecation warning class.

    In Python, deprecation warnings are usually suppressed by default. This custom warning
    class is designed to make deprecation warnings more visible, which is useful when
    the usage is likely a user mistake or bug. This class ensures that the warning is shown
    to the user more prominently, alerting them about deprecated functionality.

    It is useful in situations where deprecation indicates potential issues with the
    user's code and immediate attention is required.

    Attributes
    ----------
    __module__ : str
        The module in which this warning is defined, set to 'scikitplot'.

    Methods
    -------
    __module__
        A string representing the module that contains this warning.

    """

    # Set the module for the warning to 'scikitplot'
    __module__: str = "scikitplot"


# VisibleDeprecationWarning.__module__ = 'scikitplot'

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
    _instance: Optional["SingletonBase"] = None

    # magic method to get called in an objects instantiation.
    def __new__(cls: Type["SingletonBase"], *args, **kwargs) -> "SingletonBase":
        """
        Override the default object creation method to implement the singleton pattern.

        Ensures that only one instance of the class is created. If an instance already exists,
        that instance is returned instead of creating a new one.

        Parameters
        ----------
        cls : Type[SingletonBase]
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
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __reduce__(self) -> tuple:
        """
        Ensures the singleton instance is correctly restored during object serialization
        (i.e., pickling and unpickling).

        This method ensures that the singleton behavior is maintained during
        serialization.

        Returns
        -------
        tuple
            A tuple of (class, args) used to restore the singleton instance during unpickling.

        """
        return (self.__class__, ())


######################################################################
## SingletonBaseEnum class
# Inherits from both SingletonBase and enum.Enum.
# Inherits serialization logic from SingletonBase (supports pickling).
# Purpose: A specialized subclass of SingletonBase that also inherits from enum.Enum.
# It combines the Singleton pattern with the ability
# to create enumerated values where each enum value is a singleton instance.
# Intended Use: Used for cases where you want to define singleton instances
# that also have enum-like behavior (e.g., unique, immutable constants).
# Use: when you need both singleton behavior and enum functionality
# (e.g., predefined constant values like states or configurations).
# SingletonBaseEnum is best when you need singleton instances
# tied to an enumeration of predefined values.
# It is perfect for cases where you want both singleton behavior
# and enum features, but it comes with more complexity and constraints.
######################################################################

# class SingletonBaseEnum(SingletonBase, enum.Enum):
#     """
#     A base class for singleton pattern objects that also uses `enum.Enum`.

#     This class combines the singleton pattern with enumeration features. Each value
#     of the `enum.Enum` is a singleton instance of the class, ensuring that the same instance
#     is returned whenever the same enum value is referenced.

#     Attributes
#     ----------
#     _instance : Optional[SingletonBaseEnum]
#         The single instance of the enum value, initially set to `None`.

#     Methods
#     -------
#     __new__(cls) -> SingletonBaseEnum
#         Overrides the default object creation to implement the singleton pattern.
#         Ensures that only one instance of the enum value is created (singleton behavior).
#     """
#     _instance: Optional["SingletonBaseEnum"] = None

#     def __new__(cls: Type["SingletonBaseEnum"], value: Any) -> "SingletonBaseEnum":
#         """
#         Override the object creation method to implement the singleton pattern for enum values.

#         This method ensures that each enum value has only one instance. If the instance
#         does not exist, it creates it; otherwise, it returns the existing instance.

#         Parameters
#         ----------
#         cls : Type[SingletonBaseEnum]
#             The class being instantiated.
#         value : Any
#             The value of the enum member.

#         Returns
#         -------
#         SingletonBaseEnum
#             The single instance of the enum value.
#         """
#         # all enum instances are actually created during class construction
#         # without calling this method; this method is called by the metaclass'
#         # __call__ (i.e. Color(3) ), and by pickle
#         if cls._instance is None:
#             # Create the singleton instance only once
#             cls._instance = super().__new__(cls, value)
#         return cls._instance

######################################################################
## Singleton Marker Types
## _DefaultType class
######################################################################


class _DefaultType(SingletonBase):
    """
    A marker representing the use of a default value.

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
        Returns a string representation of the object.

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
    A marker indicating that a value or feature is deprecated.

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
        Returns a string representation of the object.

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
    A special value indicating no user-defined input.

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
        Returns a string representation of the object.

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
## Singleton for Resource Management
######################################################################


class ThreadPool(SingletonBase):
    """Singleton for managing a thread pool."""

    def __init__(self):
        import concurrent.futures

        if not hasattr(self, "executor"):
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def submit_task(self, fn, *args, **kwargs):
        """Submit a task to the thread pool."""
        return self.executor.submit(fn, *args, **kwargs)


# # Usage:
# thread_pool = ThreadPool()

# def task(x):
#     return x * x

# future = thread_pool.submit_task(task, 5)
# print(future.result())  # Output: 25

######################################################################
## Singleton for DatabaseConnection
######################################################################


class DatabaseConnection(SingletonBase):
    """Singleton class for managing a database connection."""

    def __init__(self):
        """Initialize the database connection."""
        if not hasattr(self, "connection"):
            self.connection = self.connect_to_database()

    def connect_to_database(self):
        """Simulate a database connection."""
        return "Database Connection Established"

    def get_connection(self):
        """Return the single database connection."""
        return self.connection


# Usage:
# db1 = DatabaseConnection()

######################################################################
##
######################################################################
