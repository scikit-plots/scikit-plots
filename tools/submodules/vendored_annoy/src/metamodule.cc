// ======================= NNSMeta Python type ============================
// NNSMeta -> NNSFrame / NNSDistances / NNSEmbeddings
// These keep naming canonical:
// index / NNSIndex → pandas.Index / RangeIndex
// series / NNSSeries (NNSDistances / NNSEmbeddings) → pandas.Series
// frame / NNSFrame → pandas.DataFrame
// meta / NNSMeta → schema + metadata view
// ==========================================================================

// metamodule.cc
#include "metamodule.h"

// ---------------------------------------------------------
// Python object: NNSMeta
// ---------------------------------------------------------

typedef struct {
    PyObject_HEAD
    AnnoyQueryMeta* meta;  // owned pointer, can be nullptr
} py_nnsmeta;


// Deallocator
static void
py_nnsmeta_dealloc(py_nnsmeta* self)
{
    delete self->meta;
    self->meta = nullptr;
    Py_TYPE(self)->tp_free((PyObject*) self);
}


// ======================= NNSMeta methods ====================================

// __repr__: NNSMeta(size=..., result_type=...)
static PyObject*
py_nnsmeta_repr(py_nnsmeta* self)
{
    size_t n = self->meta ? self->meta->size() : 0;
    const char* kind = "UNKNOWN";

    if (self->meta) {
        switch (self->meta->result_type) {
        case ResultType::SERIES:      kind = "SERIES"; break;
        case ResultType::EMBEDDINGS:  kind = "EMBEDDINGS"; break;
        case ResultType::FRAME:       kind = "FRAME"; break;
        default:                      kind = "UNKNOWN"; break;
        }
    }

    return PyUnicode_FromFormat("NNSMeta(size=%zu, type=%s)", n, kind);
}


// .size property
static PyObject*
py_nnsmeta_get_size(py_nnsmeta* self, void*)
{
    size_t n = self->meta ? self->meta->size() : 0;
    return PyLong_FromSize_t(n);
}


// .info() -> str
static PyObject*
py_nnsmeta_info(py_nnsmeta* self, PyObject*)
{
    if (!self->meta) {
        return PyUnicode_FromString("NNSMeta(empty)");
    }

    std::string s;
    s.reserve(256);

    s += "NNSMeta\n";
    s += "  size: " + std::to_string(self->meta->size()) + "\n";
    s += "  metric: " + self->meta->query.metric_name + "\n";
    s += "  result_type: ";

    switch (self->meta->result_type) {
    case ResultType::SERIES:      s += "SERIES";      break;
    case ResultType::EMBEDDINGS:  s += "EMBEDDINGS";  break;
    case ResultType::FRAME:       s += "FRAME";       break;
    default:                      s += "UNKNOWN";     break;
    }
    s += "\n";

    return PyUnicode_FromStringAndSize(s.c_str(), (Py_ssize_t) s.size());
}


// Methods table
static PyMethodDef py_nnsmeta_methods[] = {
    {"info", (PyCFunction) py_nnsmeta_info, METH_NOARGS,
     (char*)"Return a human-readable summary (pandas-like .info())."},
    {nullptr, nullptr, 0, nullptr}
};


// Get/set table
static PyGetSetDef py_nnsmeta_getset[] = {
    {(char*)"size", (getter) py_nnsmeta_get_size, nullptr,
     (char*)"Number of rows in the result", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}
};


// Type object
PyTypeObject NNSMetaType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "annoy.NNSMeta",                       // tp_name
    sizeof(py_nnsmeta),                    // tp_basicsize
    0,                                     // tp_itemsize
    (destructor) py_nnsmeta_dealloc,       // tp_dealloc
    0,                                     // tp_vectorcall_offset
    0,                                     // tp_getattr
    0,                                     // tp_setattr
    0,                                     // tp_as_async
    (reprfunc) py_nnsmeta_repr,            // tp_repr
    0,                                     // tp_as_number
    0,                                     // tp_as_sequence
    0,                                     // tp_as_mapping
    0,                                     // tp_hash
    0,                                     // tp_call
    0,                                     // tp_str
    0,                                     // tp_getattro
    0,                                     // tp_setattro
    0,                                     // tp_as_buffer
    Py_TPFLAGS_DEFAULT,                    // tp_flags
    (char*)"Metadata + schema for NNS results",  // tp_doc
    0,                                     // tp_traverse
    0,                                     // tp_clear
    0,                                     // tp_richcompare
    0,                                     // tp_weaklistoffset
    0,                                     // tp_iter
    0,                                     // tp_iternext
    py_nnsmeta_methods,                    // tp_methods
    0,                                     // tp_members
    py_nnsmeta_getset,                     // tp_getset
    0,                                     // tp_base
    0,                                     // tp_dict
    0,                                     // tp_descr_get
    0,                                     // tp_descr_set
    0,                                     // tp_dictoffset
    0,                                     // tp_init
    0,                                     // tp_alloc
    0,                                     // tp_new (created only from C)
};


// Public init function called from annoymodule.cc
int
Annoy_InitMetaType(PyObject* module)
{
    if (PyType_Ready(&NNSMetaType) < 0)
        return -1;

    Py_INCREF(&NNSMetaType);
    if (PyModule_AddObject(module, "NNSMeta",
                           (PyObject*) &NNSMetaType) < 0) {
        Py_DECREF(&NNSMetaType);
        return -1;
    }
    return 0;
}
