# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the numpy project.
# https://github.com/numpy/numpy/tree/main/numpy/f2py

"""
Fortran to Python Interface Generator.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the terms
of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

References
----------
.. [1] `https://stdlib.fortran-lang.org/lists/modules.html`_
.. [2] `https://numpy.org/doc/stable/f2py/index.html`_
.. [3] `https://scipy.github.io/old-wiki/pages/Cookbook/f2py_and_NumPy.html`_
.. [4] `https://scipy-cookbook.readthedocs.io/items/F2Py.html`_
.. [5] `https://scipy-cookbook.readthedocs.io/items/idx_interfacing_with_other_languages.html`_
"""

__all__ = ["get_include"]

import os

# from ..exceptions import VisibleDeprecationWarning


def get_include():
    """
    Return the directory that contains the ``fortranobject.c`` and ``.h`` files.

    .. note::

        This function is not needed when building an extension with
        `numpy.distutils` directly from ``.f`` and/or ``.pyf`` files
        in one go.

    Python extension modules built with f2py-generated code need to use
    ``fortranobject.c`` as a source file, and include the ``fortranobject.h``
    header. This function can be used to obtain the directory containing
    both of these files.

    Returns
    -------
    include_path : str
        Absolute path to the directory containing ``fortranobject.c`` and
        ``fortranobject.h``.

    Notes
    -----
    .. versionadded:: 1.21.1

    Unless the build system you are using has specific support for f2py,
    building a Python extension using a ``.pyf`` signature file is a two-step
    process. For a module ``mymod``:

    * Step 1: run ``python -m numpy.f2py mymod.pyf --quiet``. This
      generates ``mymodmodule.c`` and (if needed)
      ``mymod-f2pywrappers.f`` files next to ``mymod.pyf``.
    * Step 2: build your Python extension module. This requires the
      following source files:

      * ``mymodmodule.c``
      * ``mymod-f2pywrappers.f`` (if it was generated in Step 1)
      * ``fortranobject.c``

    See Also
    --------
    numpy.get_include : function that returns the numpy include directory

    References
    ----------
    .. [1] `https://stdlib.fortran-lang.org/lists/modules.html`_
    .. [2] `https://numpy.org/doc/stable/f2py/index.html`_
    .. [3] `https://scipy.github.io/old-wiki/pages/Cookbook/f2py_and_NumPy.html`_
    .. [4] `https://scipy-cookbook.readthedocs.io/items/F2Py.html`_
    .. [5] `https://scipy-cookbook.readthedocs.io/items/idx_interfacing_with_other_languages.html`_
    """
    return os.path.join(os.path.dirname(__file__), "src")
