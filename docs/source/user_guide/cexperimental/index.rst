.. _cexperimental-index:

..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. # https://rsted.info.ucl.ac.be/
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

======================================================================
C-Experimental (experimental)
======================================================================

This module contains functions related to :py:mod:`~.cexperimental`.
Placeholder for c-experimental features for developers...

Cython Bindings
----------------------------------------------------------------------

The Cython language is a superset of the Python language that additionally
supports calling C functions and declaring C types on variables
and class attributes.

.. seealso::

   * https://cython.org/

Pybind11 Bindings
----------------------------------------------------------------------

pybind11 is a lightweight header-only library that exposes C++ types in Python
and vice versa, mainly to create Python bindings of existing C++ code.

.. seealso::

   * https://github.com/pybind/pybind11


C/CPP Headers Source
----------------------------------------------------------------------

NumPy C-API Headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NumPy provides a C-API to enable users to extend the system
and get access to the array object for use in other routines.
The best way to truly understand ...

.. seealso::

   * https://numpy.org/devdocs/user/c-info.html
   * https://numpy.org/devdocs/reference/c-api/index.html

.. jupyter-execute::

    >>> import numpy as np
    >>> # Return the directory that contains the NumPy *.h header files.
    >>> np.get_include()



LightNumPy C/Cpp-API Headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A lightweight version of NumPy (or similar functionality).

.. seealso::

   * https://github.com/scikit-plots/lightnumpy
   * https://github.com/dpilger26/NumCpp

.. jupyter-execute::

    >>> try:
    >>>   # pip install git+https://github.com/scikit-plots/lightnumpy.git@main
    >>>   import lightnumpy as lp
    >>>   # Return the directory that contains the NumCpp *.h header files.
    >>>   inc_dir_lightnumpy = lp.get_include()
    >>> except: pass
    >>> else:
    >>>   !ls $inc_dir_lightnumpy
