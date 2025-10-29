// module_linalg.cpp
// #pragma once  // Ensures the file is included only once by the compiler

#include <pybind11/pybind11.h>     // Main pybind11 header
#include <pybind11/numpy.h>        // For Pybind11 NumPy support
#include "NumCpp.hpp"              // NumCpp library header (header-only)

#include "nc_linalg_dot.cpp"       // Include header with template

// Creates a namespace alias 'py' for the pybind11 library.
namespace py = pybind11;

// -----------------------------------------------------
// PYBIND11_MODULE(<module_name>, <variable>)
// -----------------------------------------------------
PYBIND11_MODULE(
    _linalg,  // <module_name>
    m,
    py::mod_gil_not_used(),
    py::multiple_interpreters::per_interpreter_gil()
){
    // -------- Module Docstring --------
    m.doc() = R"pbdoc(
        :py:mod:`~._linalg` is a high-performance Python module that wraps the C++ NumCpp library.

        Providing fast numerical functions with seamless NumPy array support across common numeric types.

        .. seealso::
            * https://github.com/dpilger26/NumCpp
            * https://scikit-plots.github.io/dev/user_guide/cexternals/_numcpp/index.html
    )pbdoc";

    // -------- Bind the Function --------
    // Expose/Bindings for all common numeric types the dot function to Python
    // https://pybind11.readthedocs.io/en/stable/reference.html#extras
    m.def(
        "dot",                       // Python name
        &nc_linalg_dot<double>,      // C++ function or &>(), [](), [](py::array_t<double> *arg) {}
        nc_linalg_dot_doc,           // docstring
        py::arg("a"),                // Argument 1 name
        py::arg("b")                 // Argument 2 name
    );
}
