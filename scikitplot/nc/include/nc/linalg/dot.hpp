// include/nc/linalg/dot.hpp
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "NumCpp.hpp"

namespace py = pybind11;

namespace scikitplot_nc::linalg
{

// ---------------------------------------------------------------------------
// Numpydoc-style docstring for scikitplot.nc._linalg._linalg.dot
// (will be reused by the Python wrapper via __doc__)
// ---------------------------------------------------------------------------
// In C++17+, "inline" allows the same variable to be defined in multiple translation units,
// but the linker will merge them into a single symbol.
// Before C++17, you had two options: "static" (gives internal linkage per translation unit),
// "extern" in header + definition in .cpp
// R"pbdoc(...)"pbdoc" is a multi-line raw string literal.
inline constexpr const char* dot_doc = R"pbdoc(
dot(a, b)

Dot product of two arrays using the C++ NumCpp backend.

This function behaves similarly to :func:`numpy.dot` for 1-D and 2-D
arrays but executes the computation in C++ via the NumCpp library.

Parameters
----------
a, b : array_like
    Input vectors or matrices. They are converted to
    :class:`numpy.ndarray` without copying whenever possible.
    Arrays must

    * have the same dtype, and
    * be 1-D or 2-D.

    For 1-D inputs, the result is a scalar 0-D array.
    For 2-D inputs, standard matrix multiplication rules apply.

Returns
-------
out : numpy.ndarray
    Dot product of `a` and `b`. The result dtype matches the input
    dtype for the supported dtypes.

Raises
------
ValueError
    If either input has more than 2 dimensions.
TypeError
    If the dtypes of `a` and `b` do not match, or if the dtype is
    not supported.

Notes
-----
Use both NumCpp and Numpy. https://github.com/dpilger26/NumCpp/issues/16

The computation is performed by the C++ NumCpp implementation
:cpp:`nc::dot` on :cpp:`nc::NdArray` containers obtained via the
NumCpp pybind interface.

This is a low-level C++ binding. It expects NumPy arrays and is
typically called via :func:`~scikitplot.nc.dot`, which accepts
generic array-like inputs and calls :func:`numpy.asarray` on them.

At the moment the following NumPy dtypes are supported:

* ``float64``
* ``float32``
* ``int64``

Other dtypes will raise a :class:`TypeError`.

See Also
--------
numpy.dot
scikitplot.nc.dot

Examples
--------
>>> import numpy as np
>>> a = np.array([[1,2],[3,4]])
>>> b = np.array([[5,6],[7,8]])
>>> np.dot(a, b)

>>> import scikitplot.nc as nc
>>> nc.dot(a, b)
>>> nc.dot([1,2], [3,4])
)pbdoc";


// Convenience alias for NumCpp's pybind array type.
// This is defined in NumCpp's PythonInterface when
// NUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE is enabled.
template <typename T>
using PbArray = nc::pybindInterface::pbArray<T>;


/**
 * Internal templated implementation:
 * - Assumes both arrays are already checked to have the same dtype.
 * - Assumes ndim <= 2.
 *
 * Important: we use PbArray<T> to match
 * nc::pybindInterface::pybind2nc(pbArray<dtype>&).
 */
template <typename T>
inline py::array_t<T> dot_impl(PbArray<T>& a, PbArray<T>& b)
{
    if (a.ndim() > 2 || b.ndim() > 2) {
        throw py::value_error(
            "scikitplot.nc.dot only supports 1D or 2D arrays."
        );
    }

    // Convert NumPy arrays -> NumCpp NdArray<T>
    auto a_nc = nc::pybindInterface::pybind2nc(a);
    auto b_nc = nc::pybindInterface::pybind2nc(b);

    // Delegate to NumCpp's dot implementation (dtype-preserving)
    auto result_nc = nc::dot<T>(a_nc, b_nc);

    // Convert NumCpp NdArray<T> back -> NumPy array
    return nc::pybindInterface::nc2pybind(result_nc);
}


/**
 * Front-end dispatcher:
 * - Takes untyped py::array inputs
 * - Checks dtype equality
 * - Dispatches to the appropriate templated dot_impl<T>
 * - Returns a generic py::array (dtype-preserving)
 */
// inline py::array_t<double> dot(
//     py::array_t<T, py::array::c_style> inArray1,
//     py::array_t<T, py::array::c_style> inArray2
// ){
//     // Check shapes: must be 1D or 2D arrays
//     // if (inArray1.ndim() > 2 || inArray2.ndim() > 2)
//     //     throw std::runtime_error("nc_linalg_dot only supports 1D or 2D arrays.");
//     // Convert from Python NumPy array to NumCpp NdArray
//     auto a = nc::pybindInterface::pybind2nc(inArray1);
//     auto b = nc::pybindInterface::pybind2nc(inArray2);
//     // Perform dot product computation using NumCpp
//     auto result = nc::dot<double>(a, b);
//     // Convert back to NumCpp NdArray -> Python NumPy array to return to Python
//     return nc::pybindInterface::nc2pybind(result);
// }
inline py::array dot(py::array a, py::array b)
{
    auto dtype_a = a.dtype();
    auto dtype_b = b.dtype();

    if (!dtype_a.is(dtype_b)) {
        throw py::type_error(
            "`a` and `b` must have the same dtype for scikitplot.nc.dot."
        );
    }

    // float64
    if (dtype_a.is(py::dtype::of<double>())) {
        PbArray<double> a_t(a);
        PbArray<double> b_t(b);
        auto out = dot_impl<double>(a_t, b_t);
        return out;
    }

    // float32
    if (dtype_a.is(py::dtype::of<float>())) {
        PbArray<float> a_t(a);
        PbArray<float> b_t(b);
        auto out = dot_impl<float>(a_t, b_t);
        return out;
    }

    // int64 (NumPy default integer on most platforms)
    if (dtype_a.is(py::dtype::of<long long>())) {
        using i64 = long long;
        PbArray<i64> a_t(a);
        PbArray<i64> b_t(b);
        auto out = dot_impl<i64>(a_t, b_t);
        return out;
    }

    throw py::type_error(
        "Unsupported dtype for scikitplot.nc.dot. "
        "Supported dtypes are float64, float32 and int64."
    );
}

} // namespace scikitplot_nc::linalg
