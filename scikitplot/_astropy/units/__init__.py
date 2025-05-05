# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines the `Quantity` object, which represents a number with some
associated units. `Quantity` objects support operations like ordinary numbers,
but will deal with unit conversions internally.
"""

# pylint: disable=import-error

# Compatibility shim for `TypeAlias` in Python < 3.10
try:
    # Python 3.10+ — native support
    from typing import TypeAlias
except ImportError:
    try:
        # Fallback for older Python using typing_extensions (must be installed)
        from typing_extensions import (  # type: ignore[reportMissingModuleSource]
            TypeAlias,
        )
    except ImportError:
        # Final fallback: dummy placeholder (used only for type hints)
        TypeAlias = object

from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np  # type: ignore[reportMissingModuleSource]


class DummyQuantity(np.ndarray, ABC):
    """A dummy placeholder for astropy's Quantity class that represents a number with an associated unit.

    Parameters
    ----------
    value : number, np.ndarray, or str
        The numerical value of this quantity.
    unit : str, optional
        The unit associated with the quantity (default is None).
    dtype : np.dtype, optional
        The dtype for the result, default is `np.float64`.
    copy : bool, optional
        Whether to copy the value data, default is True.
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  As in `~numpy.array`.
        This parameter is ignored if the input is a `Quantity` and ``copy=False``.
    subok : bool, optional
        If `False` (default), the returned array will be forced to be a `Quantity`.
        Otherwise, `Quantity` subclasses will be passed through,
        or a subclass appropriate for the unit will be used
        (such as `~astropy.units.Dex` for ``u.dex(u.AA)``).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array should have.
        Ones will be prepended to the shape as needed to meet
        this requirement.  This parameter is ignored if the input is a `Quantity` and ``copy=False``.

    Example Usage
    -------------
    Here is a mock example showing how to create a `DummyQuantity` and interact with it:

    ```python
    # Creating a DummyQuantity instance
    q1 = DummyQuantity([1, 2, 3], unit="m/s")

    # Displaying the value and unit of the quantity
    print(q1)  # Output: "<DummyQuantity value"=[1 2 3] unit=m/s>

    # Accessing the raw values as ndarray
    print(q1.value)  # Output: [1 2 3]

    # Accessing the unit
    print(q1.unit)  # Output: m/s

    # Mock conversion to another unit
    q2 = q1.to("km/h")  # Output: Converting from m/s to km/h
    print(q2)  # Output: "<DummyQuantity value"=[1 2 3] unit=m/s>
    ```
    In this example, a `DummyQuantity` object is created with a value array `[1, 2, 3]` and a unit `"m/s"`.
    The value can be accessed directly, and the unit is also accessible through the `.unit` property.
    The `.to()` method simulates a unit conversion (though it doesn't actually convert the values in this mock).

    Notes
    -----
    This `DummyQuantity` class mimics the behavior of the `Quantity` class
    from `astropy` but doesn't implement full unit conversion or other advanced functionality.
    It is a simplified version meant for demonstration purposes.
    """

    def __new__(
        cls,
        value,
        unit=None,
        dtype=np.float64,
        copy=True,
        order=None,
        subok=False,
        ndmin=0,
    ):
        # Create a numpy ndarray from the input values
        obj = np.asarray(value, dtype=dtype)

        # Mimicking ndarray behavior
        if copy:
            obj = obj.copy()

        # Store the unit in a private attribute _unit
        obj = obj.view(cls)
        obj._unit = unit  # Store the unit here

        # Handle dimensions if necessary
        if ndmin > obj.ndim:
            obj = np.expand_dims(obj, axis=tuple(range(ndmin - obj.ndim)))

        return obj

    @property
    def value(self):
        """Return the numeric value of the quantity."""
        return self

    @property
    def unit(self):
        """Return the unit associated with the quantity."""
        return getattr(self, "_unit", None)

    def __repr__(self):
        """Custom string representation to show value and unit."""
        return f"<DummyQuantity value={self} unit={self.unit}>"

    def to(self, unit):
        """Mock conversion to another unit (doesn't actually convert here)."""
        print(f"Converting from {self.unit} to {unit}")
        # In reality, you'd handle unit conversion here.
        return self

    def __array__(self):
        """Return the raw array representation."""
        return np.array(self)

    @abstractmethod
    def some_abstract_method(self):
        """Abstract method to be implemented by subclasses."""
        pass

    pass


# from astropy.units import Quantity  # This will restore the real class
# Mocking for test or placeholder purposes
Quantity = DummyQuantity

# Define the TypeVar with the bound to Quantity
Q = TypeVar("Q", bound=Quantity)
