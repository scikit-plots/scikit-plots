.. _lightnumpy-index:

======================================================================
LightNumPy (experimental)
======================================================================

A lightweight version of NumPy (or similar functionality).

.. seealso::

   * https://github.com/scikit-plots/lightnumpy
   * https://github.com/dpilger26/NumCpp

.. jupyter-execute::

    >>> try:
    >>>   import lightnumpy as lp
    >>>   # Return the directory that contains the NumCpp *.h header files.
    >>>   inc_dir_lightnumpy = lp.get_include()
    >>> except: pass
    >>> else:
    >>>   !ls $inc_dir_lightnumpy

Why LightNumPy?
----------------------------------------------------------------------
Performance-Driven: Optimized for both CPU and hardware accelerators (GPU/TPU).
Focused: Includes essential features without unnecessary overhead.
Adaptable: Modular structure for customized extensions.
Scalable: Ideal for IoT, embedded systems, and resource-limited devices.