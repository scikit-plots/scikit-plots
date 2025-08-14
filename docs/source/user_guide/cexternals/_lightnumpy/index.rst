.. _lightnumpy-index:

======================================================================
LightNumPy (experimental)
======================================================================

A lightweight version of NumPy or Cupy (or similar functionality).

.. seealso::

   * https://github.com/scikit-plots/lightnumpy
   * https://github.com/dpilger26/NumCpp

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
