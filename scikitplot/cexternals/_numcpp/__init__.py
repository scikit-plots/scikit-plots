
# Authors: David Pilger
# SPDX-License-Identifier: MIT

"""
NumCpp: A Header-Only C++ Library with a NumPy-Compatible API.

**Author:** David Pilger <dpilger26@gmail.com>
**License:** MIT
**Language Standard:** C++17-C++23
**Dependencies:** Boost ≥ 1.73
**Supported Compilers:** MSVC 2022, GCC 13-14, Clang 18-19

NumCpp is a templatized, header-only C++ library that provides a NumPy-style
interface for numerical computing. It features an `NdArray` class with full
support for slicing, broadcasting, random generation, vectorization, and
linear algebra, closely mirroring Python's NumPy API.

Notes
-----
This library is header-only and requires no separate compilation. It is
designed for high-performance numerical computing in C++ with a familiar
NumPy-like syntax.

Core Features:

- Array creation: `arange`, `linspace`, `zeros`, `ones`, `eye`
- Broadcasting and slicing
- Mathematical and statistical functions
- Random number generation
- Linear algebra (`linalg` module)
- Comparison, logical, and reduction operations
- File I/O, printing, endian utilities

Example Equivalents:

=======================  ===========================
NumPy (Python)           NumCpp (C++)
=======================  ===========================
np.arange(3, 7)          nc::arange<int>(3, 7)
np.sum(a)               nc::sum(a)
np.linalg.inv(a)        nc::linalg::inv(a)
=======================  ===========================

See Also
--------
Project Repository: https://github.com/dpilger26/NumCpp
"""

# __all__ = [
#     "get_include",
#     "nc",
#     "nc_develop",
# ]


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

        import scikitplot.cexternals._numcpp as nc
        ...
        Extension('extension_name', ...
                  include_dirs=nc.[get_include()])
        ...

    Examples
    --------
    >>> import scikitplot.cexternals._numcpp as nc
    >>> nc.get_include()
    '/path/to/scikitplot/cexternals/_numcpp/include'  # may vary

    >>> import importlib.resources
    >>> import pathlib
    >>> include_dir = pathlib.Path(importlib.resources.files('scikitplot.cexternals._numcpp')) / 'include'
    """
    import os
    import scikitplot
    if scikitplot.show_config is None:
        # running from lightnumpy source directory
        d = os.path.join(os.path.dirname(scikitplot.__file__), "cexternals/_numcpp", "include")
    else:
        # using installed lightnumpy core headers
        # import scikitplot.cexternals._numcpp as nc
        # __name__ is a special attribute in Python that defines the name of the current module.
        # __file__ is a special attribute that contains the path to the current module file.
        # dirname = nc.__path__  # nc.__file__
        d = os.path.join(os.path.dirname(__file__), 'include')

    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"'scikitplot.cexternals._numcpp' C and C++ headers directory not found: {d}"
        )
    return d
