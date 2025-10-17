
# Authors: David Pilger
# SPDX-License-Identifier: MIT

"""
NumCpp: A Templatized Header-Only C++ Implementation of NumPy
================================================================
Author: David Pilger <dpilger26@gmail.com>
License: MIT | Language: C++17-23 | Boost ≥ 1.73
Supported Compilers: MSVC 2022, GCC 13-14, Clang 18-19

NumCpp provides a NumPy-like API for C++, offering `NdArray` containers,
slicing, broadcasting, random generation, and mathematical operations.
It closely mirrors Python's NumPy syntax:
  np.arange(3,7)  ↔  nc::arange<int>(3,7)
  np.sum(a)       ↔  nc::sum(a)
  np.linalg.inv(a)↔  nc::linalg::inv(a)

Includes modules for:
- Initialization (linspace, zeros, ones, eye)
- Random number generation
- Logical, comparison, reduction, and linear algebra ops
- File I/O and printing utilities

See Also
--------
Docs & source: https://github.com/dpilger26/NumCpp
"""
import contextlib as _contextlib

__all__ = [
    "nc",
    "nc_develop",
]

for _m in __all__:
    with _contextlib.suppress(ImportError):
        # import importlib
        # importlib.import_module(f"scikitplot.externals.NumCpp.{_m}")
        __import__(f"scikitplot.externals.NumCpp.{_m}", globals(), locals())


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

        import scikitplot.cexternals.NumCpp as NumCpp
        ...
        Extension('extension_name', ...
                  include_dirs=NumCpp.[get_include()])
        ...

    Examples
    --------
    >>> import scikitplot.cexternals.NumCpp as NumCpp
    >>> NumCpp.get_include()
    '/path/to/scikitplot/cexternals/NumCpp/include'  # may vary

    >>> import importlib.resources
    >>> import pathlib
    >>> include_dir = pathlib.Path(importlib.resources.files('scikitplot.cexternals.NumCpp')) / 'include'
    """
    import os
    import scikitplot
    if scikitplot.show_config is None:
        # running from lightnumpy source directory
        d = os.path.join(os.path.dirname(scikitplot.__file__), "cexternals/NumCpp", "include")
    else:
        # using installed lightnumpy core headers
        # import scikitplot.cexternals.NumCpp as NumCpp
        # dirname = NumCpp.__path__  # NumCpp.__file__
        d = os.path.join(os.path.dirname(__file__), 'include')

    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"'scikitplot.cexternals.NumCpp' C and C++ headers directory not found: {d}"
        )
    return d
