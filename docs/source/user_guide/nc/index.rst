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
.. # https://documatt.com/restructuredtext-reference/element/admonition.html
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. currentmodule:: scikitplot.nc

.. _nc-index:

nc (NumCpp)
======================================================================

Examples relevant to the :py:mod:`~.nc` module with  :py:mod:`~._numcpp`.

:py:mod:`~.nc` is a Python module that provides high-performance numerical functions by exposing the C++ header-only library NumCpp, allowing seamless use of NumPy arrays in Python.

.. seealso::

   * https://github.com/dpilger26/NumCpp
   * https://numcpp.readthedocs.io/en/latest/
   * https://numcpp.readthedocs.io/en/latest/commandTable.html

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
