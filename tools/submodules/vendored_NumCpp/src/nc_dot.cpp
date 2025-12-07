// nc_dot.cpp
#pragma once  // Ensures the file is included only once by the compiler

// #define NUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE
// #define NUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE

// Standard C++ library headers
// #include <string>     // For std::string
// #include <sstream>    // For std::ostringstream (optional, if needed for other operations)
// #include <iostream>   // For std::cout (optional, if needed for debugging)
// #include <stdexcept>  // For runtime error handling
// #include <vector>     // For STL vector

#include <pybind11/pybind11.h>     // Main pybind11 header
#include <pybind11/numpy.h>        // For Pybind11 NumPy support
#include "NumCpp.hpp"              // NumCpp library header (header-only)

namespace py = pybind11;
// Creates a namespace alias 'py' for the pybind11 library.
// Allows using 'py::' instead of 'pybind11::' to shorten code.

// namespace nc_py = nc::pybindInterface;
// Creates a namespace alias 'nc_py' for 'nc::pybindInterface'.
// Makes it easier to access functions/classes in the pybindInterface sub-namespace.

// using namespace nc;
// Imports all symbols from the 'nc' namespace into the current scope.
// You can use classes and functions like NdArray or dot without 'nc::' prefix.
// ⚠️ Be cautious: may cause name conflicts if other namespaces have the same symbols.

// using nc::NdArray;
// Imports only the 'NdArray' class from the 'nc' namespace into the current scope.
// Safer than 'using namespace nc' because it only brings in one symbol.

// using nc::dot;
// Imports only the 'dot' function from the 'nc' namespace into the current scope.
// Allows calling 'dot(a, b)' directly without 'nc::dot'.

// using nc_array = nc::NdArray<dtype>;
// Defines a type alias 'nc_array' for 'nc::NdArray<dtype>'.
// Simplifies code by allowing you to write 'nc_array' instead of the full templated type.

// #ifdef __cplusplus
// // Provide C-Compatible Interface (explicitly defined for each type)
// // Ensures compatibility with C, Python, or other languages expecting C-style linkage.
// extern "C" {
// #endif
// // pass here
// #ifdef __cplusplus
// }  // End of `extern "C"` block
// #endif

// =======================================
// Example Function: Dot Product "nc_dot"
// =======================================
//
// This function:
//  • Accepts two NumPy arrays from Python
//  • Converts them into NumCpp NdArray types
//  • Computes the dot product using NumCpp
//  • Returns the result as a NumPy array back to Python
//
// Template allows data types (float, double, int...)
// Here we explicitly instantiate double version below.
//
// ---------------------------------------------------
// Function: nc_dot
// Purpose: Compute dot product using NumCpp
// Parameters:
//   inArray1, inArray2 - NumPy arrays provided by Python
// Returns:
//   A NumPy array that represents the dot product
// ---------------------------------------------------
template <typename T>


// // Macro to register dot function for any type
// #define GEN_NC_DOT(T)
//     nc_dot(){};
// // Register all desired numeric types
// GEN_NC_DOT(int)
// GEN_NC_DOT(unsigned int)
// GEN_NC_DOT(float)  // float32 loses digits after the 7th decimal place.
// GEN_NC_DOT(double)  // float64 preserves about 15–16 decimal digits.
// GEN_NC_DOT(long)
// GEN_NC_DOT(unsigned long)
// inline nc::NdArray<double> nc_dot(const nc::NdArray<double>& a, const nc::NdArray<double>& b)
inline py::array_t<double> nc_dot(
    py::array_t<T, py::array::c_style> inArray1,
    py::array_t<T, py::array::c_style> inArray2
){
    // Check shapes: must be 1D or 2D arrays
    // if (inArray1.ndim() > 2 || inArray2.ndim() > 2)
    //     throw std::runtime_error("nc_dot only supports 1D or 2D arrays.");

    // Convert from Python NumPy array to NumCpp NdArray
    auto array1 = nc::pybindInterface::pybind2nc(inArray1);
    auto array2 = nc::pybindInterface::pybind2nc(inArray2);

    // Perform dot product computation using NumCpp
    auto result = nc::dot<double>(array1, array2);

    // Convert back to NumCpp NdArray -> Python NumPy array to return to Python
    return nc::pybindInterface::nc2pybind(result);
}


// R"pbdoc(...)"pbdoc" is a raw string literal. You can put multi-line text inside without worrying about escaping quotes or newlines.
// Define a common docstring once
// In C++17+, "inline" allows the same variable to be defined in multiple translation units, but the linker will merge them into a single symbol.
// Before C++17, you had two options: "static" (gives internal linkage per translation unit), "extern" in header + definition in .cpp
inline const char* doc_nc_dot = R"pbdoc(
Compute the dot product between two arrays (1d, 2d) using NumCpp and return NumPy array.

Parameters
----------
array1 : numpy.ndarray
    First input array.
array2 : numpy.ndarray
    Second input array.

Returns
-------
numpy.ndarray
    Dot product of the two arrays as a NumPy array.

Notes
-----
- Use both NumCpp and Numpy. https://github.com/dpilger26/NumCpp/issues/16
- Supports 1D or 2D numeric arrays.
- Supports types: int, unsigned int, float, double, long, unsigned long.
- Automatically converts NumPy arrays to NumCpp NdArray under the hood.

Examples
--------
>>> import scikitplot.nc as nc
>>> nc.dot([1,2], [3,4])

>>> import numpy as np
>>> a = np.array([[1,2],[3,4]])
>>> b = np.array([[5,6],[7,8]])
>>> nc.dot(a, b)
)pbdoc";
