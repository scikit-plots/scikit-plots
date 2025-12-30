# scikitplot/nc/_wrappers.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


def _promote_to_supported_dtype(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    func_name: str = "dot",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Promote two array-like inputs to a common dtype supported by the C++ backend.

    This helper is used by high-level wrappers (e.g. :func:`scikitplot.nc.dot`)
    to emulate NumPy-style mixed-dtype behaviour while targeting a small set of
    dtypes that the underlying C++ kernels actually implement efficiently.

    The promotion logic is:

    1. Convert ``a`` and ``b`` to C-contiguous :class:`numpy.ndarray` instances
       with their original dtypes.
    2. Use :func:`numpy.result_type` on these arrays to determine the "natural"
       common dtype according to NumPy's rules.
    3. Map that common dtype to one of the backend dtypes:

       * floating types -> ``numpy.float64``
       * integer and boolean types -> ``numpy.int64``

       If the common dtype is ``object``, we make a best-effort attempt to
       interpret the data as numeric (for example, lists containing ``None`` for
       missing values) by casting to ``float64``.
    4. Cast both arrays to the chosen target dtype (avoiding copies when
       possible).

    Parameters
    ----------
    a, b : array_like
        Input data. These can be Python scalars, lists, tuples, or
        :class:`numpy.ndarray` instances.

    func_name : str, default="dot"
        Name of the function using this helper. Only used to make error
        messages more informative.

    Returns
    -------
    a_arr, b_arr : numpy.ndarray
        Views or arrays with a common dtype (``float64`` or ``int64``) and
        C-contiguous memory layout, suitable for passing directly to the C++
        backend.

    Raises
    ------
    TypeError
        If the inferred common dtype is neither integer, boolean, floating,
        nor an object dtype that can be safely converted to ``float64``.

    Notes
    -----
    * We intentionally convert to arrays *before* calling
      :func:`numpy.result_type` to avoid ambiguous dtype interpretations for
      Python lists (e.g. nested lists being misinterpreted as structured dtype
      specifications).
    * For object dtypes (``dtype('O')``), we try to coerce both arrays to
      ``float64``. This allows simple patterns such as lists containing
      ``None`` to be treated as numeric with missing values (``None`` becomes
      ``nan``). If coercion fails, a clear :class:`TypeError` is raised.
    """
    # Step 1: convert to arrays once, with original dtypes, in C order.
    a_arr = np.asarray(a, order="C")
    b_arr = np.asarray(b, order="C")

    # Step 2: let NumPy infer the "natural" result type.
    common = np.result_type(a_arr, b_arr)

    # Step 3: map NumPy's common dtype to one of our supported backend dtypes.
    if np.issubdtype(common, np.floating):
        # For now we always promote floats to float64.
        target = np.float64
    elif np.issubdtype(common, np.integer) or np.issubdtype(common, np.bool_):
        # Integers / bools are promoted to int64.
        target = np.int64
    elif common == np.dtype("O"):
        # Object dtype: try to interpret as numeric, treating things like
        # ``None`` as missing values that can map to ``nan``.
        try:
            a_arr = np.asarray(a_arr, dtype=np.float64, order="C")
            b_arr = np.asarray(b_arr, dtype=np.float64, order="C")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"scikitplot.nc.{func_name} received object-dtype inputs that "
                "could not be converted to numeric values. "
                "Please convert your inputs to a numeric dtype explicitly or "
                "clean non-numeric entries before calling this function."
            ) from exc
        return a_arr, b_arr
    else:
        # (Optional) later you could add complex support here.
        raise TypeError(
            f"scikitplot.nc.{func_name} does not yet support dtype {common!r}; "
            "supported inputs are integer, boolean, or floating types."
        )

    # Step 4: cast to the target dtype, avoiding copies when possible.
    if a_arr.dtype != target:
        a_arr = a_arr.astype(target, copy=False)
    if b_arr.dtype != target:
        b_arr = b_arr.astype(target, copy=False)

    return a_arr, b_arr


def _binary_arraylike(
    core: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    name: str | None = None,
    require_same_dtype: bool = False,
    promote_dtypes: bool = True,
) -> Callable[[npt.ArrayLike, npt.ArrayLike], np.ndarray]:
    """
    Wrap a binary C++ kernel so it behaves like a NumPy-style array_like function.

    This helper is intended for internal use when exposing low-level kernels
    such as :mod:`~scikitplot.nc._linalg._linalg` to the public Python API
    in :mod:`~scikitplot.nc.linalg` and :mod:`~scikitplot.nc`.

    It takes a C++-backed function that operates on two
    :class:`numpy.ndarray` objects and returns a new Python function that:

    * accepts generic ``array_like`` inputs (lists, tuples, ndarrays, scalars),
    * optionally promotes mixed dtypes to a common backend dtype using
      :func:`_promote_to_supported_dtype`,
    * ensures C-contiguous layout, and
    * forwards the resulting arrays to the core kernel.

    Parameters
    ----------
    core : callable
        Low-level kernel that expects two ``numpy.ndarray`` inputs and returns
        a ``numpy.ndarray``. For example, a function bound from C++ via
        pybind11 that wraps ``nc::dot`` or another NumCpp routine.

    name : str, optional
        Human-readable name used in error messages and assigned to the wrapper
        as ``__name__``. If omitted, ``core.__name__`` is used.

    require_same_dtype : bool, default=False
        If ``True`` and ``promote_dtypes`` is ``False``, a :class:`TypeError`
        is raised when the input dtypes differ. This is useful for kernels that
        do not implement any mixed-dtype logic and expect identical dtypes.

    promote_dtypes : bool, default=True
        If ``True``, mixed dtypes are automatically promoted to a common dtype
        before calling ``core`` using :func:`numpy.result_type` and
        :func:`_promote_to_supported_dtype`. This makes the wrapper behave
        more like :mod:`numpy` and :mod:`scipy` functions that accept a wide
        variety of input combinations.

        If ``False``, inputs are converted to C-contiguous arrays with
        :func:`numpy.asarray`, and the optional ``require_same_dtype`` check
        is applied.

    Returns
    -------
    wrapped : callable
        High-level function that mirrors the signature

        ``wrapped(a, b) -> numpy.ndarray``

        and can be used directly in the public API. The wrapper inherits the
        docstring from ``core`` so that user-facing documentation stays
        centralized on the underlying implementation.

    Notes
    -----
    * This utility does **not** change the semantics of the C++ kernel
      itself; it only standardizes input handling on the Python side.
    * For most public-facing functions (e.g. ``scikitplot.nc.dot``), you
      will want ``promote_dtypes=True`` so that Python users can freely mix
      ints, floats, and booleans in the same call.
    """
    func_name = name or core.__name__

    def wrapped(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
        if promote_dtypes:
            a_arr, b_arr = _promote_to_supported_dtype(a, b, func_name=func_name)
        else:
            # Simpler path: just ensure we have C-contiguous ndarrays.
            a_arr = np.asarray(a, order="C")
            b_arr = np.asarray(b, order="C")

            if require_same_dtype and a_arr.dtype != b_arr.dtype:
                raise TypeError(
                    f"{func_name} requires `a` and `b` to have the same "
                    f"dtype; got {a_arr.dtype!r} and {b_arr.dtype!r}."
                )

        return core(a_arr, b_arr)

    wrapped.__name__ = func_name
    # Reuse the docstring from the low-level C++ binding to avoid duplication
    wrapped.__doc__ = core.__doc__
    return wrapped
