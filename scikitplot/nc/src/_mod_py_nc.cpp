#include <Python.h>

// Module definition
static struct PyModuleDef nc = {
    PyModuleDef_HEAD_INIT,
    "nc", // module name
    "Minimal wrapper for nc::NdArray", // module doc
    -1, // size of per-interpreter state of the module
    nullptr, // no methods PyMethodDef
};

// Module initialization
PyMODINIT_FUNC PyInit_nc(void) {
    return PyModule_Create(&nc);
}
