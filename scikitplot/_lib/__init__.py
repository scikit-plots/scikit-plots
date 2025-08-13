# scikitplot/_lib/__init__.py

"""
Module containing private utility functions
===========================================

This module serves as the core API for the `xp` array namespace,
providing essential functionalities for array operations.
::

    from scikitplot import _lib
    _lib.test()

See Also
--------
array_api_compat :
    Python array API standard https://data-apis.org/array-api/latest/index.html

Core Features
--------------
- Create and manipulate arrays.
- Perform mathematical and statistical operations.
- Provide utility functions for array handling and processing.

Behind `xp` functionality
----------------------------

The `_lib` module offers a simple and intuitive interface for users to interact
with array data. Below are some key functionalities:

1. **Array Creation**
2. **Mathematical Operations**
3. **Statistical Functions**
4. **Array Manipulation**
5. **Utilities**

Notes
-----
- Ensure that the `xp` namespace is correctly imported before using the
  functionalities within this module.
- For more advanced usage, refer to the documentation of the individual
  functions available in this module.
"""

from . import _ccallback_c
from ._testutils import PytestTester

test = PytestTester(__name__)
del PytestTester
