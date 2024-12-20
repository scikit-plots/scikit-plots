import numpy as np

from scikitplot._xp_core_lib._array_api import (
    array_namespace,
    xp_size,
    xp_broadcast_promote,
)

from scikitplot._xp_core_lib.validation import (
    _asarray_validated,
)

__all__ = [
    'sigmoid',
    'logsumexp',
    'softmax',
    'log_softmax',
]

def sigmoid(x, axis=None):
    r"""
    Compute the sigmoid function for the input array `x`.

    The sigmoid function is defined as::

        sigmoid(x) = 1 / (1 + exp(-x))
        
    .. math:: \text{sigmoid}(x) = \frac{1}{1 + e^{-x}}
    
    .. versionadded:: 0.3.9

    Parameters
    ----------
    x : array-like
        Input array for which to compute the sigmoid. This can be a list, 
        numpy array, or any array-like structure.

    axis : int or None, optional
        Axis or axes along which to compute the sigmoid. If None, the sigmoid
        will be computed over the entire array. The default is None.

    Returns
    -------
    numpy.ndarray
        The sigmoid of each element in `x`, with the same shape as `x`.

    Examples
    --------
    >>> import numpy as np
    >>> from scikitplot.experimental._logsumexp import sigmoid
    >>> x = np.array([0, 1, 2])
    >>> sigmoid(x)
    array([0.5       , 0.7310586 , 0.88079708])
    """
    return 1 / (1 + np.exp(-x))


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    a : array_like
        Input array.

    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
        
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
        
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned. If ``return_sign`` is True, ``res`` contains the log of
        the absolute value of the argument.
    sgn : ndarray
        If ``return_sign`` is True, this will be an array of floating-point
        numbers matching res containing +1, 0, -1 (for real-valued inputs)
        or a complex phase (for complex inputs). This gives the sign of the
        argument of the logarithm in ``res``.
        If ``return_sign`` is False, only one result is returned.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2

    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.

    Examples
    --------
    >>> import numpy as np
    >>> from scikitplot._experimental import logsumexp
    >>> a = np.arange(10)
    >>> logsumexp(a)
    9.4586297444267107
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107

    With weights

    >>> a = np.arange(10)
    >>> b = np.arange(10, 0, -1)
    >>> logsumexp(a, b=b)
    9.9170178533034665
    >>> np.log(np.sum(b*np.exp(a)))
    9.9170178533034647

    Returning a sign flag

    >>> logsumexp([1,2],b=[1,-1],return_sign=True)
    (1.5413248546129181, -1.0)

    Notice that `logsumexp` does not directly support masked arrays. To use it
    on a masked array, convert the mask into zero weights:

    >>> a = np.ma.array([np.log(2), 2, np.log(3)],
    ...                  mask=[False, True, False])
    >>> b = (~a.mask).astype(int)
    >>> logsumexp(a.data, b=b), np.log(5)
    1.6094379124341005, 1.6094379124341005

    """
    xp = array_namespace(a, b)

    a, b = xp_broadcast_promote(a, b, ensure_writeable=True, force_floating=True, xp=xp)
    axis = tuple(range(a.ndim)) if axis is None else axis

    if b is not None:
        a[b == 0] = -xp.inf

    # Scale by real part for complex inputs, because this affects
    # the magnitude of the exponential.
    if xp_size(a) == 0:
        # because `xp.max` doesn't have `initial` argument...
        shape = np.asarray(a.shape)  # NumPy is concise for scalar or tuple `axis`
        shape[axis] = 1
        a_max = xp.full(tuple(shape), -xp.inf, dtype=a.dtype)
    else:
        real = xp.real(a) if xp.isdtype(a.dtype, "complex floating") else a
        a_max = xp.max(real, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~xp.isfinite(a_max)] = 0
    elif not xp.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = xp.asarray(b)
        tmp = b * xp.exp(a - a_max)
    else:
        tmp = xp.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = xp.sum(tmp, axis=axis, keepdims=keepdims, dtype=tmp.dtype)
        if return_sign:
            # For complex, use the numpy>=2.0 convention for sign.
            if xp.isdtype(s.dtype, "complex floating"):
                sgn = s / xp.where(s == 0, xp.asarray(1, dtype=s.dtype), xp.abs(s))
            else:
                sgn = xp.sign(s)
            s = xp.abs(s)
        out = xp.log(s)

    if not keepdims:
        a_max = xp.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def softmax(x, axis=None):
    r"""Compute the softmax function.

    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::

        softmax(x) = np.exp(x)/sum(np.exp(x))

    Softmax function is defined as:
    
    .. math:: \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
        
    where :math:`( x_i )` is the i-th element of the input array.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : numpy.ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.

    Notes
    -----
    The formula for the softmax function :math:`\sigma(x)` for a vector
    :math:`x = \{x_0, x_1, ..., x_{n-1}\}` is

    .. math:: \sigma(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}

    The `softmax` function is the gradient of `logsumexp`.

    The implementation uses shifting to avoid overflow. See [1]_ for more
    details.

    References
    ----------
    .. [1] P. Blanchard, D.J. Higham, N.J. Higham, "Accurately computing the
       log-sum-exp and softmax functions", IMA Journal of Numerical Analysis,
       Vol.41(4), :doi:`10.1093/imanum/draa038`.

    Examples
    --------
    >>> import numpy as np
    >>> from scikitplot._experimental import softmax
    >>> np.set_printoptions(precision=5)

    >>> x = np.array([[1, 0.5, 0.2, 3],
    ...               [1,  -1,   7, 3],
    ...               [2,  12,  13, 3]])
    ...

    Compute the softmax transformation over the entire array.

    >>> m = softmax(x)
    >>> m
    array([[  4.48309e-06,   2.71913e-06,   2.01438e-06,   3.31258e-05],
           [  4.48309e-06,   6.06720e-07,   1.80861e-03,   3.31258e-05],
           [  1.21863e-05,   2.68421e-01,   7.29644e-01,   3.31258e-05]])

    >>> m.sum()
    1.0

    Compute the softmax transformation along the first axis (i.e., the
    columns).

    >>> m = softmax(x, axis=0)

    >>> m
    array([[  2.11942e-01,   1.01300e-05,   2.75394e-06,   3.33333e-01],
           [  2.11942e-01,   2.26030e-06,   2.47262e-03,   3.33333e-01],
           [  5.76117e-01,   9.99988e-01,   9.97525e-01,   3.33333e-01]])

    >>> m.sum(axis=0)
    array([ 1.,  1.,  1.,  1.])

    Compute the softmax transformation along the second axis (i.e., the rows).

    >>> m = softmax(x, axis=1)
    >>> m
    array([[  1.05877e-01,   6.42177e-02,   4.75736e-02,   7.82332e-01],
           [  2.42746e-03,   3.28521e-04,   9.79307e-01,   1.79366e-02],
           [  1.22094e-05,   2.68929e-01,   7.31025e-01,   3.31885e-05]])

    >>> m.sum(axis=1)
    array([ 1.,  1.,  1.])

    """
    x = _asarray_validated(x, check_finite=False)
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def log_softmax(x, axis=None):
    r"""Compute the logarithm of the softmax function.

    In principle::

        log_softmax(x) = log(softmax(x))

    but using a more accurate implementation.

    .. versionadded:: 0.3.9

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : ndarray or scalar
        An array with the same shape as `x`. Exponential of the result will
        sum to 1 along the specified axis. If `x` is a scalar, a scalar is
        returned.

    Notes
    -----
    `log_softmax` is more accurate than ``np.log(softmax(x))`` with inputs that
    make `softmax` saturate (see examples below).

    Examples
    --------
    >>> import numpy as np
    >>> from scikitplot._experimental import log_softmax
    >>> from scikitplot._experimental import softmax
    >>> np.set_printoptions(precision=5)

    >>> x = np.array([1000.0, 1.0])

    >>> y = log_softmax(x)
    >>> y
    array([   0., -999.])

    >>> with np.errstate(divide='ignore'):
    ...   y = np.log(softmax(x))
    ...
    >>> y
    array([  0., -inf])

    """

    x = _asarray_validated(x, check_finite=False)

    x_max = np.amax(x, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0

    tmp = x - x_max
    exp_tmp = np.exp(tmp)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out = np.log(s)

    out = tmp - out
    return out