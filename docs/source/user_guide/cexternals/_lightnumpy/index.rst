.. _lightnumpy-index:

======================================================================
LightNumPy (experimental)
======================================================================

A lightweight version of NumPy or Cupy (or similar functionality).

.. # https://rsted.info.ucl.ac.be/
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist
.. seealso::

   * https://github.com/scikit-plots/lightnumpy
   * https://github.com/dpilger26/NumCpp

.. .. rubric:: References
References
----------
.. [1] `https://stdlib.fortran-lang.org/lists/modules.html`_
.. [2] `https://numpy.org/doc/stable/f2py/index.html`_
.. [3] `https://scipy.github.io/old-wiki/pages/Cookbook/f2py_and_NumPy.html`_
.. [4] `https://scipy-cookbook.readthedocs.io/items/F2Py.html`_
.. [5] `https://scipy-cookbook.readthedocs.io/items/idx_interfacing_with_other_languages.html`_

Why LightNumPy?
----------------------------------------------------------------------
Performance-Driven: Optimized for both CPU and hardware accelerators (GPU/TPU).
Focused: Includes essential features without unnecessary overhead.
Adaptable: Modular structure for customized extensions.
Scalable: Ideal for IoT, embedded systems, and resource-limited devices.

Get Header files location:

.. jupyter-execute::

    >>> try:
    >>>   # pip install git+https://github.com/scikit-plots/lightnumpy.git@main
    >>>   import lightnumpy as ln
    >>>   # Return the directory that contains the lightnumpy and NumCpp *.h header files.
    >>>   inc_dir_lightnumpy = ln.get_include()
    >>> except: pass
    >>> else:
    >>>   !ls $inc_dir_lightnumpy

::

   >>> import lightnumpy as ln
