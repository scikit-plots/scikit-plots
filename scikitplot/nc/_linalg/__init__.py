# scikitplot/nc/_linalg/__init__.py

"""
The scikitplot.nc._linalg linear algebra functions.

See Also
--------
numpy.linalg
"""

from __future__ import annotations

from .._wrappers import _binary_arraylike
from . import _linalg

__all__ = ["dot"]

# High-level, NumPy-style dot:
# - accepts array_like (lists, tuples, ndarrays)
# - reuses C++ docstring from _linalg.dot
dot = _binary_arraylike(_linalg.dot, name="dot")


# """
# Functions present in numpy.linalg are listed below.


# Matrix and vector products
# --------------------------

#    cross
#    multi_dot
#    matrix_power
#    tensordot
#    matmul

# Decompositions
# --------------

#    cholesky
#    outer
#    qr
#    svd
#    svdvals

# Matrix eigenvalues
# ------------------

#    eig
#    eigh
#    eigvals
#    eigvalsh

# Norms and other numbers
# -----------------------

#    norm
#    matrix_norm
#    vector_norm
#    cond
#    det
#    matrix_rank
#    slogdet
#    trace (Array API compatible)

# Solving equations and inverting matrices
# ----------------------------------------

#    solve
#    tensorsolve
#    lstsq
#    inv
#    pinv
#    tensorinv

# Other matrix operations
# -----------------------

#    diagonal (Array API compatible)
#    matrix_transpose (Array API compatible)

# Exceptions
# ----------

#    LinAlgError
# """
