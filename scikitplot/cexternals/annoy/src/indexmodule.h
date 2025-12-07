// ======================= NNSIndex Python type ============================
//
// Wraps AnnoyQueryMeta as a Index-like object with:
// Index as pandas.Index or monotonic Index.RangeIndex

// indexmodule.h
#pragma once

#include <Python.h>
#include "metamodule.h"
#include "indexmodule.h"

// High-level "index view" – like pandas.Index / RangeIndex.
extern PyTypeObject NNSIndexType;

int Annoy_InitIndexType(PyObject* module);

// ======================= Module / types ===================================


static PyMethodDef module_methods[] = {
  // {NULL}	/* Sentinel */
  {NULL, NULL, 0, NULL}	/* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef _indexmodule = {
  PyModuleDef_HEAD_INIT,
  "_index",            /* m_name */
  ANNOY_NNSFRAME_DOC,  /* m_doc */
  -1,                  /* m_size */
  module_methods,      /* m_methods */
  NULL,                /* m_slots */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL                 /* m_free */
};
#endif

static PyObject* create_module(void) {
  // Initialize NNSIndex type
  memset(&py_nns_iloc_type, 0, sizeof(PyTypeObject));
  py_nns_iloc_type.ob_base.ob_base.ob_refcnt = 1;
  py_nns_iloc_type.ob_base.ob_base.ob_type   = &PyType_Type;
  py_nns_iloc_type.ob_base.ob_size           = 0;

  py_nns_iloc_type.tp_name      = (char*)"annoy._index.NNSIndex";
  py_nns_iloc_type.tp_basicsize = sizeof(py_nns_iloc);
  py_nns_iloc_type.tp_itemsize  = 0;
  py_nns_iloc_type.tp_dealloc   = (destructor)py_nns_iloc_dealloc;
  py_nns_iloc_type.tp_flags     = Py_TPFLAGS_DEFAULT;
  py_nns_iloc_type.tp_as_mapping = &py_nns_iloc_mapping;

  if (PyType_Ready(&py_nnsindex_type) < 0) return NULL;

  // module
  PyObject* m;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&indexmodule);
#else
  m = Py_InitModule("_index", module_methods);
#endif

  if (!m) return NULL;

  Py_INCREF(&py_nns_iloc_type);
  PyModule_AddObject(m, "NNSIndex", (PyObject*)&py_nns_iloc_type);

  return m;
}
