"""
.. highlight:: cython

C-Experimental API Functions by Cython
=======================================

This module provides Cython implementations of several mathematical functions
often used in statistical and machine learning contexts,
such as the expit (sigmoid) function, its logarithm, and the logit function.
These functions are scalar and typed versions of functions commonly
found in libraries like :py:mod:`scipy.special`.

The module leverages Cython's fused types to handle different numeric types
(`double`, `float`, `long double`) in a single function definition,
making the code both efficient and flexible.
"""

# cimport cython  # noqa: F401
# from cython cimport nogil  # noqa: F401

from libc.math cimport (
    # NAN,
    exp,
    log
)  # noqa: F401
# from numpy cimport (
#     npy_float,
#     npy_double,
#     npy_longdouble,
#     npy_cdouble,
#     npy_int,
#     npy_long
# )  # noqa: F401

# Implement functions

cpdef dfg_number_t expit(dfg_number_t x0) noexcept nogil:
    """
    Compute the sigmoid (expit) of a scalar input.

    The sigmoid function maps a real-valued input to the interval (0, 1) and is
    defined as:

    .. math::

        \\operatorname{expit}(x) = \\frac{1}{1 + e^{-x}}

    This function is widely used in logistic regression and other models in
    machine learning and statistics.

    Parameters
    ----------
    x0 : dfg_number_t
        Scalar input value. Can be of type `float`, `double`, or `long double`.

    Returns
    -------
    dfg_number_t
        The sigmoid of the input value, with the same type as `x0`.

    Examples
    --------
    >>> expit(0.0)
    0.5

    >>> expit(0.5)
    0.6224593312018546

    See Also
    --------
    logit : Inverse of the sigmoid function.
    log_expit : Logarithm of the sigmoid function.

    Notes
    -----
    This is a Cython-accelerated implementation using the `dfg_number_t` typedef,
    which controls the precision (e.g., `float`, `double`, or `long double`).

    .. versionadded:: 0.3.9
    """
    return 1.0 / (1.0 + exp(-x0))


cpdef dfg_number_t log_expit(dfg_number_t x0) noexcept nogil:
    """
    Compute the natural logarithm of the sigmoid (expit) of a scalar input.

    The function is defined as:

    .. math::

        \\log\\_\\operatorname{expit}(x) = -\\log\\left(1 + e^{-x}\\right)

    This transformation is commonly used when working with log-probabilities
    or in numerically stable implementations of logistic functions.

    Parameters
    ----------
    x0 : dfg_number_t
        Scalar input value. Can be of type `float`, `double`, or `long double`.

    Returns
    -------
    dfg_number_t
        The log-sigmoid of the input value, returned with the same type as `x0`.

    Examples
    --------
    >>> log_expit(0.5)
    -0.4740769841801067

    See Also
    --------
    expit : The sigmoid function.
    logit : The inverse sigmoid (log-odds) function.

    Notes
    -----
    This is a Cython-accelerated implementation using `dfg_number_t`
    as the floating-point type.

    .. versionadded:: 0.3.9
    """
    return -log(1.0 + exp(-x0))


cpdef dfg_number_t logit(dfg_number_t x0) noexcept nogil:
    """
    Compute the logit (inverse sigmoid) of a scalar input.

    The logit function transforms a probability in the interval (0, 1) into
    a real-valued number. It is defined as:

    .. math::

        \\operatorname{logit}(x) = \\log\\left(\\frac{x}{1 - x}\\right)

    This function is commonly used in logistic regression and other
    probabilistic models.

    Parameters
    ----------
    x0 : dfg_number_t
        Scalar input value in the range (0, 1). Typically a `float`, `double`,
        or `long double`.

    Returns
    -------
    dfg_number_t
        The logit of the input value, returned with the same type as `x0`.

    Examples
    --------
    >>> logit(0.5)
    0.0

    >>> logit(0.8)
    1.3862943611198906

    See Also
    --------
    expit : The sigmoid (logistic) function.
    log_expit : Logarithm of the sigmoid function.

    Notes
    -----
    The input must be strictly within the interval (0, 1) to avoid
    mathematical domain errors (e.g., division by zero or log of zero).

    .. versionadded:: 0.3.9
    """
    return log(x0 / (1.0 - x0))
