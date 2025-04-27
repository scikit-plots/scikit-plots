import abc
from typing import TypeVar


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
