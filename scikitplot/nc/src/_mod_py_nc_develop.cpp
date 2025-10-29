#include <Python.h>

// Module definition
static struct PyModuleDef nc_develop = {
    PyModuleDef_HEAD_INIT,
    "nc_develop", // module name
    "Minimal wrapper for nc_develop::NdArray", // module doc
    -1, // size of per-interpreter state of the module
    nullptr, // no methods PyMethodDef
};

// Module initialization
PyMODINIT_FUNC PyInit_nc_develop(void) {
    return PyModule_Create(&nc_develop);
}
