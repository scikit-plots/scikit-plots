import abc
from typing import TypeVar

# Compatibility shim for `TypeAlias` in Python < 3.10
try:
    # Python 3.10+ — native support
    from typing import TypeAlias
except ImportError:
    try:
        # Fallback for older Python using typing_extensions (must be installed)
        from typing_extensions import TypeAlias
    except ImportError:
        # Final fallback: dummy placeholder (used only for type hints)
        TypeAlias = object


# Define an abstract base class
class Quantity(abc.ABC):
    pass


# Create a concrete class that subclasses Quantity
class IntegerQuantity(Quantity):
    pass


class FloatQuantity(Quantity):
    pass


# Define the TypeVar with the bound to Quantity
Q = TypeVar("Q", bound=Quantity)
