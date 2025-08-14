# Licensed under a 3-clause BSD style license - see LICENSE.rst

# pylint: disable=import-error

from copy import deepcopy

import numpy as np  # type: ignore[reportMissingModuleSource]
import numpy.ma as ma  # type: ignore[reportMissingModuleSource]


class DummyMaskedColumn:
    """Define a dummy masked data column for use in a Table object.

    Parameters
    ----------
    data : list, ndarray, or None
        Column data values
    name : str
        Column name and key for reference within Table
    mask : list, ndarray or None
        Boolean mask for which True indicates missing or invalid data
    fill_value : float, int, str, or None
        Value used when filling masked column elements
    dtype : `~numpy.dtype`-like
        Data type for column
    shape : tuple or ()
        Dimensions of a single row element in the column data
    length : int or 0
        Number of row elements in column data
    description : str or None
        Full description of column
    unit : str or None
        Physical unit
    format : str, None, or callable
        Format string for outputting column values. This can be an
        "old-style" (`format % value`) or "new-style" (`str.format`)
        format specification string or a function or any callable object that
        accepts a single value and returns a string.
    meta : dict-like or None
        Meta-data associated with the column

    Examples
    --------
    A DummyMaskedColumn is similar to a MaskedColumn but doesn't implement all the functionality.

    >>> col = DummyMaskedColumn(data=[1, 2], name='name')
    >>> col = DummyMaskedColumn(data=[1, 2], name='name', mask=[True, False])
    >>> col = DummyMaskedColumn(data=[1, 2], name='name', dtype=float, fill_value=99)
    """

    def __init__(
        self,
        data=None,
        name=None,
        mask=None,
        fill_value=None,
        dtype=np.float64,
        shape=(),
        length=0,
        description=None,
        unit=None,
        format=None,
        meta=None,
        copy=True,
        copy_indices=True,
    ):
        self.data = np.array(data) if data is not None else np.array([])
        self.name = name
        self.mask = np.array(mask) if mask is not None else np.ma.nomask
        self.fill_value = fill_value
        self.dtype = dtype
        self.shape = shape
        self.length = length
        self.description = description
        self.unit = unit
        self.format = format
        self.meta = meta
        self.copy = copy

        # If no mask provided, assume no mask (this is a simplified approach)
        if self.mask is np.ma.nomask:
            self.mask = np.zeros_like(self.data, dtype=bool)

    @property
    def value(self):
        """Return the underlying data with the mask applied."""
        return np.ma.array(self.data, mask=self.mask, fill_value=self.fill_value)

    def __repr__(self):
        return (
            f"<DummyMaskedColumn name={self.name}, data={self.data}, mask={self.mask}>"
        )

    def get_masked_data(self):
        """Retrieve the data with the mask applied."""
        return np.ma.masked_array(self.data, mask=self.mask)

    def __getitem__(self, index):
        """Get a specific entry or slice of the column."""
        return self.data[index]


# from astropy.table import MaskedColumn  # This will restore the real class
# Mocking for test or placeholder purposes
MaskedColumn = DummyMaskedColumn
