/********************************************************************
 *  NumCpp + Pybind11 Python Extension (src/nc.cpp)
 *
 * Robust and extended Python bindings for NumCpp dot product.
 * Supports all standard numeric types: int, float, double, etc.
 * Automatically converts NumPy arrays to NumCpp NdArray types.
 *
 *  ✅ Designed to work with Meson `py.extension_module`
 *  ✅ Compatible with NumPy arrays (automatically converted)
 *  ✅ Uses Pybind11 & NumCpp integration layer
 *  ----------------------------------------------------------------
 *  BUILD REQUIREMENTS:
 *    You must compile with the following defines:
 *      -DNUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE
 *      -DNUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE
 *
 *    These are already passed in your Meson config:
 *      extra_compile_args = [
 *        "-DNUMCPP_INCLUDE_PYBIND_PYTHON_INTERFACE",
 *        # "-DNUMCPP_INCLUDE_BOOST_PYTHON_INTERFACE",
 *        "-DNUMCPP_NO_USE_BOOST",
 *        "-fpermissive",
 *      ]
 *
 *  ----------------------------------------------------------------
 ********************************************************************/
// This file demonstrates how to expose a simple C++ function
// that uses NumCpp's NdArray to Python using pybind11.
// It takes two NumPy arrays from Python, converts them into
// NumCpp NdArray types, computes the dot product, and returns it
// back to Python as a NumPy array.
//
// REQUIREMENTS:
//   - NumCpp installed
//   - pybind11 installed
//   - Compile with:
//     -DINCLUDE_PYTHON_INTERFACE
//     -DINCLUDE_PYBIND_PYTHON_INTERFACE
//
// ===============================================

// nc.cpp
// #pragma once  // Ensures the file is included only once by the compiler

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

#include "nc_dot.cpp"              // Include header with template

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

// -----------------------------------------------------
// PYBIND11_MODULE creates a Python module named "nc"
// -----------------------------------------------------
// PYBIND11_MODULE(<module_name>, <variable>)
// The module name **must match** the Meson `extension_module` target.
// In your Meson config, ext_name = 'nc', so we use 'nc' here.
PYBIND11_MODULE(
    nc,
    m,
    py::mod_gil_not_used(),
    py::multiple_interpreters::per_interpreter_gil()
){
    // -------- Module Docstring --------
    m.doc() = R"pbdoc(
        :py:mod:`~.nc` is a high-performance Python module that wraps the C++ NumCpp library.

        Providing fast numerical functions with seamless NumPy array support across common numeric types.

        .. seealso::
            * https://github.com/dpilger26/NumCpp
            * https://scikit-plots.github.io/dev/user_guide/cexternals/_numcpp/index.html
    )pbdoc";

    m.attr("__version__") = nc::VERSION;

    // -------- Bind the Function --------
    // Expose/Bindings for all common numeric types the dot function to Python
    // https://pybind11.readthedocs.io/en/stable/reference.html#extras
    m.def(
        "dot",              // Python name
        &nc_dot<double>,    // C++ function or &>(), [](), [](py::array_t<double> *arg) {}
        doc_nc_dot,         // docstring
        py::arg("array1"),  // Argument 1 name
        py::arg("array2")   // Argument 2 name
    );
}
