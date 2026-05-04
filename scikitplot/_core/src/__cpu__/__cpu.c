// meson generates:
//   scikitplot/_core/include/npy_cpu_dispatch_config.h
//   (identical in content to what you uploaded — baseline/dispatch macros)
// Your C files must include in this order:
//   npy_cpu.h            ← defines NPY_CPU_AMD64 / NPY_CPU_X86 / NPY_CPU_ARM etc.
//   npy_os.h             ← defines NPY_OS_LINUX etc. (used by features.c for auxv path)
//   npy_cpu_dispatch_config.h   ← defines the NPY_WITH_CPU_* macros (generated)
//   npy_cpu_features.h   ← declares npy_cpu_init(), npy_cpu_have(), the three builders
//   npy_cpu_dispatch.h   ← includes config.h again (safe, guarded) + adds CURFX/CALL macros

/* __cpu_.c — correct top of file */
#include <Python.h>
#include <structmember.h>

#include "scikitplot/npy_cpu.h"          /* arch detection */
#include "scikitplot/npy_os.h"           /* OS detection   */
#include "npy_cpu_dispatch_config.h"     /* generated      */
#include "npy_cpu_features.h"            /* the 3 builders */

static int module_loaded = 0;

static int
__cpu___exec(PyObject *m) {
    PyObject *d, *s, *c_api;

    // https://docs.python.org/3/howto/isolating-extensions.html#opt-out-limiting-to-one-module-object-per-process
    if (module_loaded) {
        PyErr_SetString(PyExc_ImportError,
                        "cannot load module more than once per process");
        return -1;
    }
    module_loaded = 1;

    /* Add some symbolic constants to the module */
    d = PyModule_GetDict(m);
    if (!d) {
        return -1;
    }

    s = npy_cpu_features_dict();
    if (s == NULL) {
        return -1;
    }
    if (PyDict_SetItemString(d, "__cpu_features__", s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    s = npy_cpu_baseline_list();
    if (s == NULL) {
        return -1;
    }
    if (PyDict_SetItemString(d, "__cpu_baseline__", s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    s = npy_cpu_dispatch_list();
    if (s == NULL) {
        return -1;
    }
    if (PyDict_SetItemString(d, "__cpu_dispatch__", s) < 0) {
        Py_DECREF(s);
        return -1;
    }
    Py_DECREF(s);

    return 0;
}

static struct PyModuleDef_Slot __cpu___slots[] = {
    {Py_mod_exec, __cpu___exec},
#if PY_VERSION_HEX >= 0x030c00f0  // Python 3.12+
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
#endif
#if PY_VERSION_HEX >= 0x030d00f0  // Python 3.13+
    // signal that this module supports running without an active GIL
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL},
};

static struct PyModuleDef moduledef = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "__cpu__",
    .m_size = 0,
    // .m_methods = array_module_methods,
    .m_slots = __cpu___slots,
};

PyMODINIT_FUNC PyInit___cpu__(void) {
    return PyModuleDef_Init(&moduledef);
}
