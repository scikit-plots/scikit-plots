// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include "NdArrayCore.hpp"

// namespace py = pybind11;
// using namespace nc_develop;

// PYBIND11_MODULE(nc_develop, m) {
//     m.doc() = "Minimal wrapper for nc_develop::NdArray";
// }

#include <Python.h>

// Module definition
static struct PyModuleDef ncmodule = {
    PyModuleDef_HEAD_INIT,
    "nc_develop", // module name
    "Minimal wrapper for nc_develop::NdArray", // module doc
    -1, // size of per-interpreter state of the module
    nullptr, // no methods
};

// Module initialization
PyMODINIT_FUNC PyInit_nc_develop(void) {
    return PyModule_Create(&ncmodule);
}
