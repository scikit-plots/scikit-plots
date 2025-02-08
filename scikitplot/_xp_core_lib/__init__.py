"""
Module containing private utility functions
===========================================

This module serves as the core API for the `xp` array namespace,
providing essential functionalities for array operations.
::

    from scikitplot import _xp_core_lib
    _xp_core_lib.test()

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

The `_xp_core_api` module offers a simple and intuitive interface for users to interact
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

# scikitplot/_xp_core_lib/__init__.py

# https://data-apis.org/array-api/latest/index.html
# https://github.com/data-apis/array-api-compat/blob/main/array_api_compat/numpy/__init__.py
try:
    from . import _ccallback_c as _ccallback_c
    from .array_api_compat import __version__ as __array_api_compat_version__  # xpc
    from .array_api_compat.numpy import __array_api_version__
    from .array_api_extra import __version__ as __array_api_extra_version__  # xpx
except:
    __array_api_version__ = "2023.12"
    __array_api_compat_version__ = "1.10.1.dev0"
    __array_api_extra_version__ = "0.5.1.dev0"

from ._testutils import PytestTester

test = PytestTester(__name__)
del PytestTester
