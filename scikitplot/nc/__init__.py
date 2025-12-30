# scikitplot/nc/__init__.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
NumCpp: A Templatized Header Only C++ Library with Implementation of the Python NumPy-Compatible API.

* Author: `David Pilger <dpilger26@gmail.com>`_
* License: MIT

Compilers:

* C++ Standards: C++17 C++20 C++23
* MSVC Visual Studio: 2022
* GCC GNU: 13.3, 14.2
* Clang LLVM: 18, 19
* Boost Versions: 1.73+

.. seealso::
   * https://github.com/dpilger26/NumCpp
"""

# PUBLIC, user-facing API: Python wrapper that accepts array_like (lists, tuples, ndarrays)
from ._linalg import *  # noqa: F403
from ._version import *  # noqa: F403

__author__ = "David Pilger"
__author_email__ = "dpilger26@gmail.com"
__git_hash__ = "7d390df4ae94268e58222278529b22ebae2ee663"


def get_include() -> str:
    """
    Return the absolute path to the NumCpp C++ headers include directory.

    Returns
    -------
    str
        Path to the directory containing C and C++ header files.

    Notes
    -----
    When using ``setuptools``, for example in ``setup.py``::

        import scikitplot.nc as nc
        ...
        Extension('extension_name', ...
                  include_dirs=nc.[get_include()])
        ...

    Examples
    --------
    >>> import scikitplot.nc as nc
    >>> nc.get_include()
    '/path/to/scikitplot/cexternals/_numcpp/include'  # may vary

    >>> import importlib.resources
    >>> import pathlib
    >>> include_dir = (
    ...     pathlib.Path(importlib.resources.files("scikitplot.cexternals._numcpp"))
    ...     / "include"
    ... )
    """
    import os  # noqa: I001, PLC0415
    import scikitplot  # noqa: I001, PLC0415

    if getattr(scikitplot, "show_config", None) is None:
        # running from lightnumpy source directory
        d = os.path.join(
            os.path.dirname(scikitplot.__file__), "cexternals/_numcpp", "include"
        )
    else:
        # using installed lightnumpy core headers
        # import scikitplot.cexternals._numcpp as nc
        # __name__ is a special attribute in Python that defines the name of the current module.
        # __file__ is a special attribute that contains the path to the current module file.
        # dirname = nc.__path__  # nc.__file__
        d = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../cexternals/_numcpp/include")
        )
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"'scikitplot.cexternals._numcpp' C and C++ headers directory not found: {d}"
        )
    return d
