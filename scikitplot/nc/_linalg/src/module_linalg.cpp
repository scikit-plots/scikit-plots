// scikitplot/nc/_linalg/src/module_linalg.cpp
#include <pybind11/pybind11.h>     // Main pybind11 header
#include <pybind11/numpy.h>        // For Pybind11 NumPy support
#include "NumCpp.hpp"              // NumCpp library header (header-only)

#include "nc.hpp"                  // Include header with template

// Creates a namespace alias 'py' for the pybind11 library.
namespace py = pybind11;

// NumCpp header
using nc::pybindInterface::pbArray;

// scikitplot_nc header
using scikitplot_nc::linalg::dot;
using scikitplot_nc::linalg::dot_doc;

// -----------------------------------------------------
// PYBIND11_MODULE(<module_name>, <variable>)
// -----------------------------------------------------
PYBIND11_MODULE(
    _linalg,  // <module_name>, so Python sees scikitplot.nc._linalg._linalg
    m,
    py::mod_gil_not_used(),
    py::multiple_interpreters::per_interpreter_gil()
){
    // -------- Module Docstring --------
    m.doc() = R"pbdoc(
        :py:mod:`~scikitplot.nc._linalg._linalg` - low-level linear
        algebra bindings to the NumCpp C++ library.

        This module exposes small, typed kernels on NumPy arrays.
        End users are expected to use :mod:`~scikitplot.nc._linalg`
        and :mod:`~scikitplot.nc`, which wrap these kernels with
        NumPy-style array_like handling.

        .. seealso::
            * https://github.com/dpilger26/NumCpp
            * https://scikit-plots.github.io/dev/user_guide/cexternals/_numcpp/index.html
    )pbdoc";

    // -------- Bind the Function --------
    // Expose/Bindings for all common numeric types the dot function to Python
    // https://pybind11.readthedocs.io/en/stable/reference.html#extras
    // Single dispatcher-based binding (NumPy-style, dtype-preserving)
    // m.def("dot", [](pbArray<double> a, pbArray<double> b) {
    //     auto A = nc::pybindInterface::pybind2nc(a);
    //     auto B = nc::pybindInterface::pybind2nc(b);
    //     auto R = nc::dot<double>(A, B);
    //     return nc::pybindInterface::nc2pybind(R);
    // }, "...doc...");
    m.def(
        "dot",         // Python name
        &dot,          // C++ function or &>(), [](), [](py::array_t<double> *arg) {}
        dot_doc,       // docstring from header
        py::arg("a"),  // Argument 1 name
        py::arg("b")   // Argument 2 name
    );

    // later:
    // m.def("norm", [](pbArray<double> a) { ... });
    // m.def("det",  [](pbArray<double> a) { ... });
}
