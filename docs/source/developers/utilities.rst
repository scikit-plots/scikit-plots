.. _developers-utils:

.. currentmodule:: scikitplot.utils

.. warning::

    Not Implemented...

========================
Utilities for Developers
========================

**Reserved.**

scikit-plots contains a number of utilities to help with development.  These are
located in :mod:`scikitplot.utils`, and include tools in a number of categories.
All the following functions and classes are in the module :mod:`scikitplot.utils`.

.. warning ::

   These utilities are meant to be used internally within the scikit-plots
   package.  They are not guaranteed to be stable between versions of
   scikit-plots.  Backports, in particular, will be removed as the scikit-plots
   dependencies evolve.


Validation Tools
================

These are tools used to check and validate input.  When you write a function
which accepts arrays, matrices, or sparse matrices as arguments, the following
should be used when applicable.

- :func:`assert_all_finite`: Throw an error if array contains NaNs or Infs.

- :func:`as_float_array`: convert input to an array of floats.  If a sparse
  matrix is passed, a sparse matrix will be returned.

- :func:`check_array`: check that input is a 2D array, raise error on sparse
  matrices. Allowed sparse matrix formats can be given optionally, as well as
  allowing 1D or N-dimensional arrays. Calls :func:`assert_all_finite` by
  default.

- :func:`check_X_y`: check that X and y have consistent length, calls
  check_array on X, and column_or_1d on y. For multilabel classification or
  multitarget regression, specify multi_output=True, in which case check_array
  will be called on y.

- :func:`indexable`: check that all input arrays have consistent length and can
  be sliced or indexed using safe_index.  This is used to validate input for
  cross-validation.

- :func:`validation.check_memory` checks that input is ``joblib.Memory``-like,
  which means that it can be converted into a
  ``sklearn.utils.Memory`` instance (typically a str denoting
  the ``cachedir``) or has the same interface.

If your code relies on a random number generator, it should never use
functions like ``numpy.random.random`` or ``numpy.random.normal``.  This
approach can lead to repeatability issues in unit tests.  Instead, a
``numpy.random.RandomState`` object should be used, which is built from
a ``random_state`` argument passed to the class or function.  The function
:func:`check_random_state`, below, can then be used to create a random
number generator object.

- :func:`check_random_state`: create a ``np.random.RandomState`` object from
  a parameter ``random_state``.

  - If ``random_state`` is ``None`` or ``np.random``, then a
    randomly-initialized ``RandomState`` object is returned.
  - If ``random_state`` is an integer, then it is used to seed a new
    ``RandomState`` object.
  - If ``random_state`` is a ``RandomState`` object, then it is passed through.

For example::

    >>> from scikitplot.utils import check_random_state
    >>> random_state = 0
    >>> random_state = check_random_state(random_state)
    >>> random_state.rand(4)
    array([0.5488135 , 0.71518937, 0.60276338, 0.54488318])

When developing your own scikit-plots compatible estimator, the following
helpers are available.

- :func:`validation.check_is_fitted`: check that the estimator has been fitted
  before calling ``transform``, ``predict``, or similar methods. This helper
  allows to raise a standardized error message across estimator.

- :func:`validation.has_fit_parameter`: check that a given parameter is
  supported in the ``fit`` method of a given estimator.