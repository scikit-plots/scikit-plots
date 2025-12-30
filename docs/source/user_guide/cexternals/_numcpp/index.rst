
.. _cexternals-numcpp-index:

======================================================================
NumCpp Header Only C++ (experimental)
======================================================================

This module contains some functions related to :py:mod:`~.nc`, :py:mod:`~._numcpp` under :py:mod:`~.cexternals`.

NumCpp: A Templatized Header Only C++ Library with Implementation of the Python NumPy-Compatible API.

.. seealso::

   * https://github.com/dpilger26/NumCpp
   * https://numcpp.readthedocs.io/en/latest/
   * https://numcpp.readthedocs.io/en/latest/commandTable.html

* Author: `David Pilger <dpilger26@gmail.com>`_
* License: MIT

Compilers:

* C++ Standards: C++17 C++20 C++23
* MSVC Visual Studio: 2022
* GCC GNU: 13.3, 14.2
* Clang LLVM: 18, 19
* Boost Versions: 1.73+

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

* Array creation: `arange`, `linspace`, `zeros`, `ones`, `eye`
* Broadcasting and slicing
* Mathematical and statistical functions
* Random number generation
* Linear algebra (`linalg` module)
* Comparison, logical, and reduction operations
* File I/O, printing, endian utilities

Example Equivalents:

=======================  ===========================
NumPy (Python)           NumCpp (C++)
=======================  ===========================
np.arange(3, 7)          nc::arange<int>(3, 7)
np.sum(a)                nc::sum(a)
np.linalg.inv(a)         nc::linalg::inv(a)
=======================  ===========================

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

   from scikitplot import nc
   nc.get_include()

.. .. code-block:: python
.. prompt:: python >>>

   from scikitplot import nc
   print(nc.__doc__)

   print(nc.dot.__doc__)
