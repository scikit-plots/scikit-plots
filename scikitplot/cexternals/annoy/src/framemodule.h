// ======================= NNSFrame Python type ============================
//
// Wraps AnnoyQueryMeta as a DataFrame-like object with:
// FRAME as full schema: index / indice / distance / value / metrics
//
//   * len()
//   * __getitem__ (int / slice)
//   * .iloc[...] (via iloc indexer object)
//   * .head(), .tail()
//   * .to_list()
//   * .sort_values(by='distance'|'item', ascending=True, inplace=False)
//   * .sort_index(ascending=True, inplace=False)
//   * .value_counts()
//   * .min(column='distance'), .max(column='distance')
//   * .abs(inplace=False)
//   * .to_dataframe() – requires pandas
// ==========================================================================

// framemodule.h
#pragma once

#include <Python.h>
#include "metamodule.h"

// DataFrame-like view of ANN results.
extern PyTypeObject NNSFrameType;

int Annoy_InitFrameType(PyObject* module);

namespace Annoy {
  // Register DataFrame-like result types (NNSFrame, etc.)
  bool RegisterFrameTypes(PyObject* module);
}




typedef struct {
  PyObject_HEAD
  AnnoyQueryResult* r;
} py_nnsresult;

typedef struct {
  PyObject_HEAD
  py_nnsresult* owner;
} py_nns_iloc;

static PyTypeObject py_nnsresult_type;
static PyTypeObject py_nns_iloc_type;

// ======================= NNSFrame methods ====================================

static void py_nnsresult_dealloc(py_nnsresult* self) {
  if (self->r) {
    delete self->r;
    self->r = NULL;
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* py_nnsresult_new_from_cpp(const AnnoyQueryResult& rr) {
  py_nnsresult* obj = PyObject_New(py_nnsresult, &py_nnsresult_type);
  if (!obj) return PyErr_NoMemory();
  obj->r = new AnnoyQueryResult(rr);
  return (PyObject*)obj;
}

// convert AnnoyQueryResult to list[(item, distance)]
static PyObject* nnsresult_to_list_of_tuples(const AnnoyQueryResult& rr) {
  Py_ssize_t n = (Py_ssize_t)rr.items.size();
  PyObject* list = PyList_New(n);
  if (!list) return NULL;

  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* tup = PyTuple_New(2);
    if (!tup) {
      Py_DECREF(list);
      return NULL;
    }
    PyTuple_SET_ITEM(tup, 0, PyLong_FromLong(rr.items[(size_t)i]));
    PyTuple_SET_ITEM(tup, 1, PyFloat_FromDouble(rr.distances[(size_t)i]));
    PyList_SET_ITEM(list, i, tup);
  }
  return list;
}

// __len__
static Py_ssize_t py_nns_len(py_nnsresult* self) {
  return (Py_ssize_t)self->r->items.size();
}

// __getitem__(int or slice)
static PyObject* py_nns_getitem(PyObject* self_obj, PyObject* key) {
  py_nnsresult* self = (py_nnsresult*)self_obj;
  size_t n = self->r->items.size();

  if (PyLong_Check(key)) {
    long idx = PyLong_AsLong(key);
    if (idx < 0) idx += (long)n;
    if (idx < 0 || (size_t)idx >= n) {
      PyErr_SetString(PyExc_IndexError, "index out of range");
      return NULL;
    }
    PyObject* tup = PyTuple_New(2);
    if (!tup) return NULL;
    PyTuple_SET_ITEM(tup, 0, PyLong_FromLong(self->r->items[(size_t)idx]));
    PyTuple_SET_ITEM(tup, 1, PyFloat_FromDouble(self->r->distances[(size_t)idx]));
    return tup;
  }

  if (PySlice_Check(key)) {
    Py_ssize_t start, stop, step;
    if (PySlice_Unpack(key, &start, &stop, &step) < 0)
      return NULL;

    if (step != 1) {
      PyErr_SetString(PyExc_ValueError, "only step=1 slices are supported");
      return NULL;
    }

    Py_ssize_t length = (Py_ssize_t)n;
    PySlice_AdjustIndices(length, &start, &stop, step);

    if (start < 0) start = 0;
    if (stop < start) stop = start;

    size_t s = (size_t)start;
    size_t e = (size_t)stop;
    AnnoyQueryResult sliced = self->r->slice(s, e);
    return py_nnsresult_new_from_cpp(sliced);
  }

  PyErr_SetString(PyExc_TypeError, "invalid index type");
  return NULL;
}

static PyObject* py_nns_head(py_nnsresult* self, PyObject* args) {
  long n = 5;
  if (!PyArg_ParseTuple(args, "|l", &n)) return NULL;
  if (n < 0) n = 0;
  size_t end = (size_t)std::min<long>(n, (long)self->r->items.size());
  AnnoyQueryResult sliced = self->r->slice(0, end);
  return py_nnsresult_new_from_cpp(sliced);
}

static PyObject* py_nns_tail(py_nnsresult* self, PyObject* args) {
  long n = 5;
  if (!PyArg_ParseTuple(args, "|l", &n)) return NULL;
  if (n < 0) n = 0;

  size_t total = self->r->items.size();
  size_t nn = (size_t)n;
  size_t start = (nn >= total) ? 0 : total - nn;
  AnnoyQueryResult sliced = self->r->slice(start, total);
  return py_nnsresult_new_from_cpp(sliced);
}

// .to_list() → [(item, distance)]
static PyObject* py_nns_to_list(py_nnsresult* self, PyObject* args) {
  (void)args;
  return nnsresult_to_list_of_tuples(*self->r);
}

// properties: items / distances
static PyObject* py_nns_get_items(py_nnsresult* self, void* closure) {
  (void)closure;
  Py_ssize_t n = (Py_ssize_t)self->r->items.size();
  PyObject* list = PyList_New(n);
  if (!list) return NULL;
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyList_SET_ITEM(list, i, PyLong_FromLong(self->r->items[(size_t)i]));
  }
  return list;
}

static PyObject* py_nns_get_distances(py_nnsresult* self, void* closure) {
  (void)closure;
  Py_ssize_t n = (Py_ssize_t)self->r->distances.size();
  PyObject* list = PyList_New(n);
  if (!list) return NULL;
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyList_SET_ITEM(list, i, PyFloat_FromDouble(self->r->distances[(size_t)i]));
  }
  return list;
}

// iloc indexer: wrapper type with __getitem__ delegating to parent
static PyObject* py_nns_iloc_getitem(py_nns_iloc* self, PyObject* key) {
  return py_nns_getitem((PyObject*)self->owner, key);
}

static void py_nns_iloc_dealloc(py_nns_iloc* self) {
  Py_XDECREF(self->owner);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* py_nns_iloc_attr(py_nnsresult* self, void* closure) {
  (void)closure;
  py_nns_iloc* ix = PyObject_New(py_nns_iloc, &py_nns_iloc_type);
  if (!ix) return PyErr_NoMemory();
  Py_INCREF(self);
  ix->owner = self;
  return (PyObject*)ix;
}

// iteration → yields (item, distance) tuples
static PyObject* py_nns_iter(py_nnsresult* self) {
  PyObject* list = nnsresult_to_list_of_tuples(*self->r);
  if (!list) return NULL;
  PyObject* it = PyObject_GetIter(list);
  Py_DECREF(list);
  return it;
}

// sort_values(by="distance"|"item", ascending=True, inplace=False)
static PyObject* py_nns_sort_values(py_nnsresult* self,
                                    PyObject* args,
                                    PyObject* kwargs) {
  static const char* kwlist[] = {"by", "ascending", NULL};

  const char* by = (const char*)"distance";
  int ascending_int = 1;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|sp", (char**)kwlist,
                                   &by, &ascending_int)) {
    return NULL;
  }

  const bool ascending = ascending_int != 0;

  AnnoyQueryResult tmp;
  try {
    tmp = *(self->r);
  } catch (const std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return NULL;
  }

  const size_t n = tmp.items.size();
  std::vector<size_t> idx(n);
  for (size_t i = 0; i < n; ++i) {
    idx[i] = i;
  }

  const bool by_distance = (std::strcmp(by, "distance") == 0);
  const bool by_item     = (std::strcmp(by, "item") == 0);

  if (!by_distance && !by_item) {
    PyErr_SetString(PyExc_ValueError, "by must be 'distance' or 'item'");
    return NULL;
  }

  if (by_distance) {
    std::stable_sort(idx.begin(), idx.end(),
      [&tmp, ascending](size_t a, size_t b) {
        if (ascending) {
          return tmp.distances[a] < tmp.distances[b];
        } else {
          return tmp.distances[a] > tmp.distances[b];
        }
      }
    );
  } else { // by_item
    std::stable_sort(idx.begin(), idx.end(),
      [&tmp, ascending](size_t a, size_t b) {
        if (ascending) {
          return tmp.items[a] < tmp.items[b];
        } else {
          return tmp.items[a] > tmp.items[b];
        }
      }
    );
  }

  // Reorder into a new result object
  AnnoyQueryResult sorted;
  sorted.items.resize(n);
  sorted.distances.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const size_t j = idx[i];
    sorted.items[i]     = tmp.items[j];
    sorted.distances[i] = tmp.distances[j];
  }

  PyObject* obj = py_nnsresult_new_from_cpp(sorted);
  if (!obj) {
    return NULL;
  }

  return obj;
}

// sort_index(ascending=True, inplace=False) → sort by item
static PyObject* py_nns_sort_index(py_nnsresult* self, PyObject* args, PyObject* kwargs) {
  int ascending = 1;
  int inplace = 0;
  static const char* kwlist[] = {"ascending", "inplace", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|pp", (char**)kwlist, &ascending, &inplace))
    return NULL;

  PyObject* py_args = Py_BuildValue("()");
  if (!py_args) return NULL;
  PyObject* py_kwargs = Py_BuildValue("{s:s,s:p,s:p}",
                                      "by", "item",
                                      "ascending", ascending ? 1 : 0,
                                      "inplace", inplace ? 1 : 0);
  if (!py_kwargs) {
    Py_DECREF(py_args);
    return NULL;
  }
  PyObject* res = py_nns_sort_values(self, py_args, py_kwargs);
  Py_DECREF(py_args);
  Py_DECREF(py_kwargs);
  return res;
}

// value_counts() → dict {item: count}
static PyObject* py_nns_value_counts(py_nnsresult* self, PyObject* args) {
  (void)args;
  std::unordered_map<int32_t, Py_ssize_t> counts;
  counts.reserve(self->r->items.size());
  for (size_t i = 0; i < self->r->items.size(); ++i) {
    counts[self->r->items[i]] += 1;
  }

  PyObject* d = PyDict_New();
  if (!d) return NULL;
  for (std::unordered_map<int32_t, Py_ssize_t>::const_iterator it = counts.begin();
       it != counts.end(); ++it) {
    PyObject* k = PyLong_FromLong(it->first);
    PyObject* v = PyLong_FromSsize_t(it->second);
    if (!k || !v || PyDict_SetItem(d, k, v) < 0) {
      Py_XDECREF(k);
      Py_XDECREF(v);
      Py_DECREF(d);
      return NULL;
    }
    Py_DECREF(k);
    Py_DECREF(v);
  }
  return d;
}

// min(column='distance'), max(column='distance')
static PyObject* py_nns_min(py_nnsresult* self, PyObject* args, PyObject* kwargs) {
  const char* column = "distance";
  static const char* kwlist[] = {"column", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|s", (char**)kwlist, &column))
    return NULL;

  std::string col(column ? column : "distance");
  size_t n = self->r->items.size();

  if (n == 0) Py_RETURN_NONE;

  if (col == "distance") {
    float m = self->r->distances[0];
    for (size_t i = 1; i < n; ++i)
      if (self->r->distances[i] < m) m = self->r->distances[i];
    return PyFloat_FromDouble(m);
  } else if (col == "item") {
    int32_t m = self->r->items[0];
    for (size_t i = 1; i < n; ++i)
      if (self->r->items[i] < m) m = self->r->items[i];
    return PyLong_FromLong(m);
  }

  PyErr_SetString(PyExc_ValueError, "column must be 'distance' or 'item'");
  return NULL;
}

static PyObject* py_nns_max(py_nnsresult* self, PyObject* args, PyObject* kwargs) {
  const char* column = "distance";
  static const char* kwlist[] = {"column", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|s", (char**)kwlist, &column))
    return NULL;

  std::string col(column ? column : "distance");
  size_t n = self->r->items.size();

  if (n == 0) Py_RETURN_NONE;

  if (col == "distance") {
    float m = self->r->distances[0];
    for (size_t i = 1; i < n; ++i)
      if (self->r->distances[i] > m) m = self->r->distances[i];
    return PyFloat_FromDouble(m);
  } else if (col == "item") {
    int32_t m = self->r->items[0];
    for (size_t i = 1; i < n; ++i)
      if (self->r->items[i] > m) m = self->r->items[i];
    return PyLong_FromLong(m);
  }

  PyErr_SetString(PyExc_ValueError, "column must be 'distance' or 'item'");
  return NULL;
}

// abs(inplace=False): abs(distance)
static PyObject* py_nns_abs(py_nnsresult* self, PyObject* args, PyObject* kwargs) {
  int inplace = 0;
  static const char* kwlist[] = {"inplace", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p", (char**)kwlist, &inplace))
    return NULL;

  AnnoyQueryResult tmp = *self->r;
  for (size_t i = 0; i < tmp.distances.size(); ++i)
    tmp.distances[i] = static_cast<float>(std::fabs(tmp.distances[i]));

  if (inplace) {
    *(self->r) = tmp;
    PY_RETURN_SELF;
  } else {
    return py_nnsresult_new_from_cpp(tmp);
  }
}

// to_dataframe(): requires pandas
static PyObject* py_nns_to_dataframe(py_nnsresult* self, PyObject* args, PyObject* kwargs) {
  (void)args; (void)kwargs;
  PyObject* pandas = PyImport_ImportModule("pandas");
  if (!pandas) return NULL;

  PyObject* df_class = PyObject_GetAttrString(pandas, "DataFrame");
  Py_DECREF(pandas);
  if (!df_class) return NULL;

  Py_ssize_t n = (Py_ssize_t)self->r->items.size();
  PyObject* items_list = PyList_New(n);
  PyObject* dist_list  = PyList_New(n);
  if (!items_list || !dist_list) {
    Py_XDECREF(items_list);
    Py_XDECREF(dist_list);
    Py_DECREF(df_class);
    return NULL;
  }

  for (Py_ssize_t i = 0; i < n; ++i) {
    PyList_SET_ITEM(items_list, i, PyLong_FromLong(self->r->items[(size_t)i]));
    PyList_SET_ITEM(dist_list,  i, PyFloat_FromDouble(self->r->distances[(size_t)i]));
  }

  PyObject* data_dict = PyDict_New();
  if (!data_dict ||
      PyDict_SetItemString(data_dict, "item", items_list) < 0 ||
      PyDict_SetItemString(data_dict, "distance", dist_list) < 0) {
    Py_XDECREF(data_dict);
    Py_DECREF(items_list);
    Py_DECREF(dist_list);
    Py_DECREF(df_class);
    return NULL;
  }

  Py_DECREF(items_list);
  Py_DECREF(dist_list);

  PyObject* args_df = PyTuple_Pack(1, data_dict);
  Py_DECREF(data_dict);
  if (!args_df) {
    Py_DECREF(df_class);
    return NULL;
  }

  PyObject* df = PyObject_CallObject(df_class, args_df);
  Py_DECREF(df_class);
  Py_DECREF(args_df);
  return df;
}

// --- type metadata for NNSFrame & iloc ---

static PyMethodDef py_nns_methods[] = {
  {"head",         (PyCFunction)py_nns_head,        METH_VARARGS,                   "First n rows (default 5)"},
  {"tail",         (PyCFunction)py_nns_tail,        METH_VARARGS,                   "Last n rows (default 5)"},
  {"to_list",      (PyCFunction)py_nns_to_list,     METH_NOARGS,                    "Return [(item, distance), ...]"},
  {"sort_values",  (PyCFunction)py_nns_sort_values, METH_VARARGS | METH_KEYWORDS,   "Sort by 'distance' or 'item'"},
  {"sort_index",   (PyCFunction)py_nns_sort_index,  METH_VARARGS | METH_KEYWORDS,   "Sort by index (item)"},
  {"value_counts", (PyCFunction)py_nns_value_counts,METH_NOARGS,                    "Return {item: count}"},
  {"min",          (PyCFunction)py_nns_min,         METH_VARARGS | METH_KEYWORDS,   "Min over 'distance' or 'item'"},
  {"max",          (PyCFunction)py_nns_max,         METH_VARARGS | METH_KEYWORDS,   "Max over 'distance' or 'item'"},
  {"abs",          (PyCFunction)py_nns_abs,         METH_VARARGS | METH_KEYWORDS,   "Absolute distance"},
  {"to_dataframe", (PyCFunction)py_nns_to_dataframe,METH_VARARGS | METH_KEYWORDS,   "Convert to pandas.DataFrame"},
  {NULL, NULL, 0, NULL}
};

static PyGetSetDef py_nns_getset[] = {
  {(char*)"iloc",      (getter)py_nns_iloc_attr,    NULL, (char*)"row indexer", NULL},
  {(char*)"items",     (getter)py_nns_get_items,    NULL, (char*)"item ids",    NULL},
  {(char*)"distances", (getter)py_nns_get_distances,NULL, (char*)"distances",   NULL},
  {NULL}
};

static PySequenceMethods py_nns_seq = {
  (lenfunc)py_nns_len,    /* sq_length */
  0,0,0,0,0,0,0,0,0
};

static PyMappingMethods py_nns_mapping = {
  0,                                  /* mp_length (not used) */
  (binaryfunc)py_nns_getitem,         /* mp_subscript */
  0                                   /* mp_ass_subscript */
};

static PyMappingMethods py_nns_iloc_mapping = {
  0,
  (binaryfunc)py_nns_iloc_getitem,
  0
};

// ======================= Module / types ===================================


static PyMethodDef module_methods[] = {
  // {NULL}	/* Sentinel */
  {NULL, NULL, 0, NULL}	/* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef _framemodule = {
  PyModuleDef_HEAD_INIT,
  "_frame",             /* m_name */
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
  // Initialize NNSFrame type
  memset(&py_nnsresult_type, 0, sizeof(PyTypeObject));
  py_nnsresult_type.ob_base.ob_base.ob_refcnt = 1;
  py_nnsresult_type.ob_base.ob_base.ob_type   = &PyType_Type;
  py_nnsresult_type.ob_base.ob_size           = 0;

  py_nnsresult_type.tp_name      = (char*)"annoy._frame.NNSFrame";
  py_nnsresult_type.tp_basicsize = sizeof(py_nnsresult);
  py_nnsresult_type.tp_itemsize  = 0;
  py_nnsresult_type.tp_dealloc   = (destructor)py_nnsresult_dealloc;
  py_nnsresult_type.tp_flags     = Py_TPFLAGS_DEFAULT;
  py_nnsresult_type.tp_doc       = (char*)"Annoy nearest-neighbor query result";
  py_nnsresult_type.tp_as_sequence = &py_nns_seq;
  py_nnsresult_type.tp_as_mapping  = &py_nns_mapping;
  py_nnsresult_type.tp_iter        = (getiterfunc)py_nns_iter;
  py_nnsresult_type.tp_methods     = py_nns_methods;
  py_nnsresult_type.tp_getset      = py_nns_getset;

  if (PyType_Ready(&py_nnsframe_type) < 0) return NULL;

  // module
  PyObject* m;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&_framemodule);
#else
  m = Py_InitModule("_frame", module_methods);
#endif

  if (!m) return NULL;

  Py_INCREF(&py_nnsresult_type);
  PyModule_AddObject(m, "NNSFrame", (PyObject*)&py_nnsresult_type);

  Py_INCREF(&py_nns_iloc_type);
  PyModule_AddObject(m, "NNSIndex", (PyObject*)&py_nns_iloc_type);

  return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit__frame(void) {
  return create_module();
}
#else
PyMODINIT_FUNC initannoylib(void) {
  create_module();
}
#endif

// vim: tabstop=2 shiftwidth=2
