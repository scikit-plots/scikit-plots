// Copyright (c) 2013 Spotify AB
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include "Python.h"
#include "bytesobject.h"
#include "annoylib.h"
#include "kissrandom.h"

// Fix deprecated in Python 3.11+
#include "structmember.h"

#include <exception>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <cctype>
#include <ctime>
#include <cstdlib>
#include <unordered_map>

#if defined(_MSC_VER) && _MSC_VER == 1500
typedef signed __int32    int32_t;
#else
#include <stdint.h>
#endif


#if defined(ANNOYLIB_USE_AVX512)
#define AVX_INFO "Using 512-bit AVX instructions"
#elif defined(ANNOYLIB_USE_AVX128)
#define AVX_INFO "Using 128-bit AVX instructions"
#else
#define AVX_INFO "Not using AVX instructions"
#endif

#if defined(_MSC_VER)
#define COMPILER_INFO "Compiled using MSC"
#elif defined(__GNUC__)
#define COMPILER_INFO "Compiled on GCC"
#else
#define COMPILER_INFO "Compiled on unknown platform"
#endif

#define ANNOY_DOC (COMPILER_INFO ". " AVX_INFO ".")

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#ifndef Py_TYPE
    #define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif

#ifdef IS_PY3K
    #define PyInt_FromLong PyLong_FromLong
#endif

using namespace Annoy;

#ifdef ANNOYLIB_MULTITHREADED_BUILD
  typedef AnnoyIndexMultiThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#else
  typedef AnnoyIndexSingleThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#endif

// ErrorBuffer automatically frees char* error messages.
struct ErrorBuffer {
  // by Google, LLVM, Chromium, Qt: Type* variable;
  char* ptr = nullptr;
  ~ErrorBuffer() { if(ptr) free(ptr); }
};

// TempAnnoyFile auto-deletes temporary files on destruction. (C++ RAII):
struct TempAnnoyFile {
  std::string path;
  ~TempAnnoyFile() { if (!path.empty()) std::remove(path.c_str()); }
};

// For demonstration, let's use a placeholder returns a path like "/tmp/annoy_temp_XYZ.annoy".
static std::string create_temp_filename() {
    // In a real C API, you'd use a robust mechanism (mkstemp, Python tempfile module access)
    std::string path = "/tmp/.annoy_temp" + std::to_string(std::time(0)) + "_" + std::to_string(rand()) + ".annoy";
    return path;
}

// Only use NULL (0) when maintaining legacy C/C++ code that must compile in C or pre-C++11 environments.
// nullptr is std::nullptr_t, only works with pointers. Recommended in C++11 and later, C++ only (not C).
template class Annoy::AnnoyIndexInterface<int32_t, float>;

class HammingWrapper : public AnnoyIndexInterface<int32_t, float> {
  // Wrapper class for Hamming distance, using composition.
  // This translates binary (float) vectors into packed uint64_t vectors.
  // This is questionable from a performance point of view. Should reconsider this solution.
private:
  // ✅ C++ initializes class members in the order they are declared
  // safe: 0 or 1 Hamming bits are only 0 or 1 uint64_t → float implicit conversion.
  // 1       // int (usually 32-bit)
  // 1ULL    // unsigned long long (typically 64-bit)
  //
  // -------------------------
  // Internal Annoy index and metadata
  // -------------------------
  int32_t _f_external; // number of bits in original vector
  int32_t _f_internal; // number of uint64_t per vector
  AnnoyIndex<int32_t, uint64_t, Hamming, Kiss64Random, AnnoyIndexThreadedBuildPolicy> _index;

  // Header structure for serialization
  struct HammingHeader {
      uint32_t f_external;
      uint32_t f_internal;
      uint32_t n_items;
      uint32_t reserved; // future use
  };

  // Packing: float [0,1] -> uint64_t bits
  void _pack(const float* src, uint64_t* dst) const {
    for (int32_t i = 0; i < _f_internal; i++) {
      dst[i] = 0;
      for (int32_t j = 0; j < 64 && i * 64 + j < _f_external; j++) {
        // Explicitly produce 0ULL or 1ULL, then shift
        // dst[i] |= (uint64_t)(src[i * 64 + j] > 0.5) << j;
        dst[i] |= (src[i * 64 + j] > 0.5f ? 1ULL : 0ULL) << j;
      }
    }
  };

  // Unpacking: uint64_t bits -> float [0,1]
  void _unpack(const uint64_t* src, float* dst) const {
    for (int32_t i = 0; i < _f_external; i++) {
      // Extract bit as uint64_t
      uint64_t bit = (src[i / 64] >> (i % 64)) & 1ULL; // stays uint64_t
      // Convert safely to float
      // dst[i] = (src[i / 64] >> (i % 64)) & 1;
      dst[i] = static_cast<float>(bit); // explicit, safe
    }
  };

public:
  // Constructor
  HammingWrapper(int f)
    : _f_external(f), _f_internal((f + 63) / 64), _index((f + 63) / 64) {};

  // Index operations
  bool add_item(int32_t item_idx, const float* w, char**error) {
    vector<uint64_t> w_internal(_f_internal, 0);
    _pack(w, &w_internal[0]);
    return _index.add_item(item_idx, &w_internal[0], error);
  };

  bool build(int q, int n_threads, char** error) { return _index.build(q, n_threads, error); };

  // bool deserialize(vector<uint8_t>* bytes, bool prefault, char** error) { return _index.deserialize(bytes, prefault, error); };
  bool deserialize(vector<uint8_t>* bytes, bool prefault, char** error) {
    if (!bytes || bytes->empty()) {
        if (error) *error = strdup("Empty byte vector");
        return false;
    }
    // return _index.deserialize(bytes, prefault, error);

    if (!bytes || bytes->size() < sizeof(HammingHeader)) {
        if (error) *error = strdup("Serialized buffer too small for HammingWrapper");
        return false;
    }

    HammingHeader hdr;
    memcpy(&hdr, bytes->data(), sizeof(HammingHeader));

    // if (hdr.f_external == 0 || hdr.f_internal == 0) {
    //     if (error) *error = strdup("Invalid HammingWrapper header");
    //     return false;
    // }

    _f_external = hdr.f_external;
    _f_internal = hdr.f_internal;

    const uint8_t* idx_data = bytes->data() + sizeof(HammingHeader);
    size_t idx_size = bytes->size() - sizeof(HammingHeader);
    vector<uint8_t> idx_bytes(idx_data, idx_data + idx_size);

    return _index.deserialize(&idx_bytes, prefault, error);
  }

  float get_distance(int32_t i, int32_t j) const { return _index.get_distance(i, j); };

  void get_item(int32_t item_idx, float* v) const {
    vector<uint64_t> v_internal(_f_internal, 0);
    _index.get_item(item_idx, &v_internal[0]);
    _unpack(&v_internal[0], v);
  };

  int32_t get_n_items() const { return _index.get_n_items(); };
  int32_t get_n_trees() const { return _index.get_n_trees(); };

  // Nearest neighbors queries
  void get_nns_by_item(int32_t item_idx, size_t n, int search_k,
                       vector<int32_t>* result, vector<float>* distances) const {
    if (distances) {
      vector<uint64_t> distances_internal;
      _index.get_nns_by_item(item_idx, n, search_k, result, &distances_internal);

      // distances->insert(distances->begin(), distances_internal.begin(), distances_internal.end());
      distances->resize(distances_internal.size());
      // for (size_t i = 0; i < distances_internal.size(); i++)
      //   distances->at(i) = (float)distances_internal[i];
      for (size_t i = 0; i < distances_internal.size(); i++) {
          uint64_t d = distances_internal[i];

          // sanity: Hamming distance can never exceed _f_external
          // (not required, but defensively future-proof)
          if (d > (uint64_t)_f_external) d = (uint64_t)_f_external;

          distances->at(i) = static_cast<float>(d);
      }
    } else {
      // _index.get_nns_by_item(item_idx, n, search_k, result, NULL);
      _index.get_nns_by_item(item_idx, n, search_k, result, nullptr);
    }
  };
  void get_nns_by_vector(const float* w, size_t n, int search_k,
                         vector<int32_t>* result, vector<float>* distances) const {
    vector<uint64_t> w_internal(_f_internal, 0);
    _pack(w, &w_internal[0]);
    if (distances) {
      vector<uint64_t> distances_internal;
      _index.get_nns_by_vector(&w_internal[0], n, search_k, result, &distances_internal);

      // distances->insert(distances->begin(), distances_internal.begin(), distances_internal.end());
      distances->resize(distances_internal.size());
      for (size_t i = 0; i < distances_internal.size(); i++) {
          uint64_t d = distances_internal[i];
          if (d > (uint64_t)_f_external) d = (uint64_t)_f_external;
          distances->at(i) = static_cast<float>(d);
      }
    } else {
      // _index.get_nns_by_vector(&w_internal[0], n, search_k, result, NULL);
      _index.get_nns_by_vector(&w_internal[0], n, search_k, result, nullptr);
    }
  };

  // Disk I/O
  bool load(const char* filename, bool prefault, char** error) { return _index.load(filename, prefault, error); };
  bool on_disk_build(const char* filename, char** error) { return _index.on_disk_build(filename, error); };
  bool save(const char* filename, bool prefault, char** error) { return _index.save(filename, prefault, error); };
  void set_seed(uint64_t q) { _index.set_seed(q); };

  // Robust serialization
  vector<uint8_t> serialize(char** error) const {
    // return _index.serialize(error);

    vector<uint8_t> bytes;

    // Header
    HammingHeader hdr;
    hdr.f_external = static_cast<uint32_t>(_f_external);
    hdr.f_internal = static_cast<uint32_t>(_f_internal);
    hdr.n_items = static_cast<uint32_t>(_index.get_n_items());
    hdr.reserved = 0;

    const uint8_t* hdr_bytes = reinterpret_cast<const uint8_t*>(&hdr);
    bytes.insert(bytes.end(), hdr_bytes, hdr_bytes + sizeof(HammingHeader));

    // Append internal Annoy index bytes
    vector<uint8_t> idx_bytes = _index.serialize(error);
    if (idx_bytes.empty() && error && *error) return {};
    bytes.insert(bytes.end(), idx_bytes.begin(), idx_bytes.end());

    return bytes;
  };

  bool unbuild(char** error) { return _index.unbuild(error); };
  void unload() { _index.unload(); };
  void verbose(bool v) { _index.verbose(v); };
};

// -----------------------------
// Typedefs for concrete Annoy types
// -----------------------------
typedef AnnoyIndex<int32_t, float, Angular, Kiss64Random, AnnoyIndexThreadedBuildPolicy> AnnoyAngular;
typedef AnnoyIndex<int32_t, float, Euclidean, Kiss64Random, AnnoyIndexThreadedBuildPolicy> AnnoyEuclidean;
typedef AnnoyIndex<int32_t, float, Manhattan, Kiss64Random, AnnoyIndexThreadedBuildPolicy> AnnoyManhattan;
typedef AnnoyIndex<int32_t, float, DotProduct, Kiss64Random, AnnoyIndexThreadedBuildPolicy> AnnoyDot;


// Trim whitespace from both ends
static inline std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// Normalize metric strings with aliases
static inline std::string normalize_metric(const std::string& m) {
    static const std::unordered_map<std::string, std::string> metric_map = {
        {"angular", "angular"}, {"cosine", "angular"},
        {"euclidean", "euclidean"}, {"l2", "euclidean"}, {"euclid", "euclidean"},
        {"manhattan", "manhattan"}, {"l1", "manhattan"}, {"cityblock", "manhattan"}, {"taxicab", "manhattan"},
        {"dot", "dot"}, {"dotproduct", "dot"}, {"inner_product", "dot"}, {"ip", "dot"}, {".", "dot"},
        {"hamming", "hamming"}, {"ham", "hamming"}
    };

    auto s = trim(m);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);

    auto it = metric_map.find(s);
    if (it != metric_map.end()) return it->second;

    // Option 3: Python-style error directly in C extension
    PyErr_Format(
      PyExc_ValueError,
      "Invalid metric: '%s'. Valid options: angular, euclidean, manhattan, dot, hamming.",
      m.c_str()
    );
    // Option 2: Throw a C++ exception
    // throw std::invalid_argument(
    //   "Invalid metric: " + m +
    //   ". Valid options: angular, euclidean, manhattan, dot, hamming."
    // );
    // Option 1: Return empty string (caller must check for empty string and raise an error explicitly).
    return ""; // invalid metric
}

// Helper: metric_from_ptr -> canonical metric string
static inline std::string metric_from_ptr(AnnoyIndexInterface<int32_t,float>* ptr) {
    if (!ptr) return "";
    if (dynamic_cast<AnnoyAngular*>(ptr)) return "angular";
    if (dynamic_cast<AnnoyEuclidean*>(ptr)) return "euclidean";
    if (dynamic_cast<AnnoyManhattan*>(ptr)) return "manhattan";
    if (dynamic_cast<AnnoyDot*>(ptr)) return "dot";
    if (dynamic_cast<HammingWrapper*>(ptr)) return "hamming";
    return "";
}

// Factory
AnnoyIndexInterface<int32_t, float>* create_index_for_metric(int f, const std::string& metric) {
    if (metric == "angular") return new AnnoyAngular(f);
    if (metric == "euclidean") return new AnnoyEuclidean(f);
    if (metric == "manhattan") return new AnnoyManhattan(f);
    if (metric == "dot") return new AnnoyDot(f);
    if (metric == "hamming") return new HammingWrapper(f);
    return nullptr;
}


// annoy python object
typedef struct {
  PyObject_HEAD
  AnnoyIndexInterface<int, float> *ptr;
  int f;
  std::string metric;
  // NEW FIELDS
  PyObject *raw_f_obj;
  PyObject *raw_metric_obj;
} py_annoy;


static PyObject *
py_an_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    py_annoy *self = (py_annoy *)type->tp_alloc(type, 0);
    // if (!self) return NULL;
    if (self == NULL) { return NULL; };

    PyObject *f_obj = NULL;
    PyObject *metric_obj = NULL;

    static char const * kwlist[] = {"f", "metric", NULL};
    // kwlist arrays must use const char* because string literals are immutable.
    // Parse f as PyObject (accept None)
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", (char**)kwlist, &f_obj, &metric_obj)) {
        return NULL;
    }

    Py_XINCREF(f_obj);
    self->raw_f_obj = f_obj;

    Py_XINCREF(metric_obj);
    self->raw_metric_obj = metric_obj;

    return (PyObject *)self;
}


static int
py_an_init(py_annoy *self, PyObject *args, PyObject *kwargs) {

    // Defaults
    // self->f = 0;
    // self->ptr = nullptr;
    // self->raw_f_obj = NULL;
    // self->raw_metric_obj = NULL;

    // ---- decode f ----
    if (self->raw_f_obj == NULL || self->raw_f_obj == Py_None) {
        self->f = 0;  // unknown dimension
    } else {
        auto fv = (int)PyLong_AsLong(self->raw_f_obj);
        if (PyErr_Occurred())
            return -1;
        self->f = fv;
    }

    // ---- decode metric ----
    std::string metric;
    if (self->raw_metric_obj == NULL || self->raw_metric_obj == Py_None) {
        metric = "";   // metric not known yet
    } else {
        const char *m = PyUnicode_AsUTF8(self->raw_metric_obj);
        if (!m) return -1;
        metric = m;
    }

    // ---- lazy initialization rules ----

    // case 1: user gave nothing → delayed init
    if (self->f == 0 && metric.empty()) {
        self->ptr = nullptr;
        return 0;
    }

    // case 2: no metric, but f known → future warning
    if (metric.empty()) {
        // This keeps coming up, see #368 etc
        PyErr_WarnEx(PyExc_FutureWarning, "The default argument for metric will be removed "
        "in future version of Annoy. Please pass metric='angular' explicitly.", 1);
        self->metric = "angular";         // <-- ADD
        self->ptr = new AnnoyAngular(self->f);
        return 0;
    }

    // case 3: both given → initialize fully
    std::string nm = normalize_metric(metric);
    if (nm.empty()) {
        PyErr_SetString(PyExc_ValueError, "Invalid metric");
        return -1;
    }
    self->metric = nm;               // <-- ADD THIS
    self->ptr = create_index_for_metric(self->f, self->metric);
    if (!self->ptr) {
        PyErr_SetString(PyExc_ValueError, "Failed to create index");
        return -1;
    }

    return 0;
}


static void
py_an_dealloc(py_annoy* self) {
    delete self->ptr;

    Py_XDECREF(self->raw_f_obj);
    Py_XDECREF(self->raw_metric_obj);

    Py_TYPE(self)->tp_free((PyObject*)self);
}


static PyMemberDef py_annoy_members[] = {
  // Keep 'f' in tp_members
  // {(char*)"f", T_INT, offsetof(py_annoy, f), 0, (char*)""},
  {(char*)"f", T_INT, offsetof(py_annoy, f), 0, (char*)"dimension of vectors"},
  {NULL}	/* Sentinel */
};

// Getter for 'metric'
static PyObject* py_annoy_get_metric(py_annoy* self, void* closure) {
    std::string m;
    if (!self->ptr) m = "";
    else if (dynamic_cast<AnnoyAngular*>(self->ptr)) m = "angular";
    else if (dynamic_cast<AnnoyEuclidean*>(self->ptr)) m = "euclidean";
    else if (dynamic_cast<AnnoyManhattan*>(self->ptr)) m = "manhattan";
    else if (dynamic_cast<AnnoyDot*>(self->ptr)) m = "dot";
    else if (dynamic_cast<HammingWrapper*>(self->ptr)) m = "hamming";
    return PyUnicode_FromStringAndSize(m.c_str(), m.size());
}

// Setter for 'metric'
static int py_annoy_set_metric(py_annoy* self, PyObject* value, void* closure) {
  if (!PyUnicode_Check(value)) {
      PyErr_SetString(PyExc_TypeError, "metric must be a string");
      return -1;
  }

  const char* s = PyUnicode_AsUTF8(value);
  if (!s) return -1;

  // normalize & validate metric
  std::string nm = normalize_metric(s);
  if (nm.empty()) {
      PyErr_SetString(PyExc_ValueError, "Invalid metric");
      return -1;
  }
  self->metric = nm; // store validated metric <-- ADD THIS

  // Lazy init only if index not yet initialized, defer actual ptr creation If Only first keep data
  if (self->f == 0) {
    if (!self->ptr) {
      self->ptr = create_index_for_metric(self->f, self->metric);
      if (!self->ptr) {
        PyErr_SetString(PyExc_RuntimeError,
          "AnnoyIndex not initialized. Pass `f` and/or `metric`, or call add_item(0) first.");
        return -1;
      }
    }
    return 0;
  }
  // if f > 0, do not rebuild, keep existing ptr and data
  // metric change will take effect when a new index is created or rebuilt
  if (self->f > 0) {
      // create new index with new metric
      auto new_ptr = create_index_for_metric(self->f, self->metric);
      if (!new_ptr) { PyErr_SetString(PyExc_RuntimeError, "Failed to rebuild index"); return -1; }

      // copy all existing items
      for (int i = 0; i < self->ptr->get_n_items(); i++) {
          std::vector<float> v(self->f);
          self->ptr->get_item(i, &v[0]);
          new_ptr->add_item(i, &v[0]);
      }

      delete self->ptr;
      self->ptr = new_ptr;
  }
  return 0;
}

// Get/Set table
static PyGetSetDef py_annoy_getset[] = {
    {"metric", (getter)py_annoy_get_metric, (setter)py_annoy_set_metric, "metric name", NULL},
    {NULL}  /* Sentinel */
};


static PyObject *
py_an_load(py_annoy *self, PyObject *args, PyObject *kwargs) {
  // filename is received from Python as a borrowed pointer to immutable UTF-8 storage → must be const char*.
  // error is dynamically allocated by Annoy (strdup) → must remain char* so free() is valid.
  const char *filename;
  char *error;
  bool prefault = false;

  if (!self->ptr) return NULL;

  static char const * kwlist[] = {"fn", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|b", (char**)kwlist, &filename, &prefault))
    return NULL;

  if (!self->ptr->load(filename, prefault, &error)) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  Py_RETURN_TRUE;
}


static PyObject *
py_an_save(py_annoy *self, PyObject *args, PyObject *kwargs) {
  // filename is received from Python as a borrowed pointer to immutable UTF-8 storage → must be const char*.
  // error is dynamically allocated by Annoy (strdup) → must remain char* so free() is valid.
  const char *filename;
  char *error;
  bool prefault = false;

  if (!self->ptr) return NULL;

  static char const * kwlist[] = {"fn", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|b", (char**)kwlist, &filename, &prefault))
    return NULL;

  if (!self->ptr->save(filename, prefault, &error)) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  Py_RETURN_TRUE;
}

// Helper: Convert C++ vectors to Python (NN results)
static PyObject*
get_nns_to_python(const vector<int32_t>& result,
                  const vector<float>& distances,
                  // bool include_distances
                  int include_distances) {

  // PyObject* py_result = PyList_New(result.size());
  // if (!py_result) return NULL;

  PyObject* l = NULL;
  PyObject* d = NULL;
  PyObject* t = NULL;

  if ((l = PyList_New(result.size())) == NULL) {
    goto error;
  }

  for (size_t i = 0; i < result.size(); i++) {
    PyObject* res = PyInt_FromLong(result[i]);
    if (res == NULL) {
      goto error;
    }
    PyList_SetItem(l, i, res);  // steals reference
  }

  if (!include_distances) return l;

  if ((d = PyList_New(distances.size())) == NULL) {
    goto error;
  }

  for (size_t i = 0; i < distances.size(); i++) {
    PyObject* dist = PyFloat_FromDouble(distances[i]);
    if (dist == NULL) {
      goto error;
    }
    PyList_SetItem(d, i, dist);  // steals reference
  }

  if ((t = PyTuple_Pack(2, l, d)) == NULL) {
    goto error;
  }

  Py_XDECREF(l);
  Py_XDECREF(d);

  return t;

  error:
    Py_XDECREF(l);
    Py_XDECREF(d);
    Py_XDECREF(t);
    return NULL;
}


bool check_constraints(py_annoy *self, int32_t item_idx, bool building) {
  if (item_idx < 0) {
    PyErr_SetString(PyExc_IndexError, "Item index can not be negative");
    return false;
  } else if (!building && item_idx >= self->ptr->get_n_items()) {
    PyErr_SetString(PyExc_IndexError, "Item index larger than the largest item index");
    return false;
  } else {
    return true;
  }
}

static PyObject*
py_an_get_nns_by_item(py_annoy *self, PyObject *args, PyObject *kwargs) {
  // int32_t item_idx, n, search_k=-1, include_distances=0;
  int32_t item_idx;
  int32_t n;
  int search_k = -1;
  int include_distances = 0;

  static char const * kwlist[] = {"i", "n", "search_k", "include_distances", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ii|ii", (char**)kwlist, &item_idx, &n, &search_k, &include_distances))
    return NULL;

  if (!self->ptr) return NULL;

  if (!check_constraints(self, item_idx, false)) {
    return NULL;
  }

  vector<int32_t> result;
  vector<float> distances;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_item(item_idx, n, search_k, &result, include_distances ? &distances : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(result, distances, include_distances);
}


bool
convert_list_to_vector(PyObject* v, int f, vector<float>* w) {

  Py_ssize_t length = PyObject_Size(v);

  if (length == -1) { return false; };

  // If f is undefined → infer automatically
  if (f == 0) {
    f = (int)length;
    w->assign(f, 0.0f);   // resize w correctly
  }
  else if (length != f) {
    PyErr_Format(PyExc_IndexError,
      "Vector has wrong length (expected %d, got %ld)",
      f, length
    );
    return false;
  }

  // Fill vector
  // w->resize(f);
  for (int z = 0; z < f; z++) {
    // PyObject *pf = PySequence_GetItem(v, z);
    // if (!pf) return false;

    PyObject *idx = PyInt_FromLong(z);  // PyLong_FromLong
    // Py_ssize_t idx = PyLong_AsSsize_t(idx);  // Convert PyObject* to C integer
    if (idx == NULL) { return false; };

    // ?? remove redundant PyObject_GetItem loops (replace with PySequence_Fast)
    PyObject *pf = PyObject_GetItem(v, idx);  // Works for any mapping or sequence
    Py_DECREF(idx);
    if (pf == NULL) { return false; };

    double value = PyFloat_AsDouble(pf);
    Py_DECREF(pf);
    if (value == -1.0 && PyErr_Occurred()) { return false; };

    (*w)[z] = (float)value;
  }
  return true;
}

static PyObject*
py_an_get_nns_by_vector(py_annoy *self, PyObject *args, PyObject *kwargs) {
  PyObject* v;

  // int32_t n, search_k=-1, include_distances=0;
  int32_t n;
  int search_k = -1;
  int include_distances = 0;

  static char const * kwlist[] = {"vector", "n", "search_k", "include_distances", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|ii", (char**)kwlist, &v, &n, &search_k, &include_distances))
    return NULL;

  // if (!self->ptr) return NULL;
  if (!self->ptr) {
      PyErr_SetString(PyExc_RuntimeError,
          "AnnoyIndex not initialized. Pass `f` and/or `metric`, or call add_item(0) first.");
      return NULL;
  }

  // Temporary vector, dimension = current self->f
  vector<float> w(self->f);
  int old_f = self->f;

  // Convert Python vector to C++
  if (!convert_list_to_vector(v, self->f, &w)) { return NULL; };

  // If f was 0 → this was first vector ever seen, so initialize index now
  if (old_f == 0) {
    PyErr_SetString(PyExc_RuntimeError,
      "Cannot query nearest neighbors before index dimension is known. "
      "Call add_item() first.");
    return NULL;
  }

  vector<int32_t> result;
  vector<float> distances;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_vector(&w[0], n, search_k, &result, include_distances ? &distances : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(result, distances, include_distances);
}


static PyObject*
py_an_get_item_vector(py_annoy *self, PyObject *args) {
  int32_t item_idx;

  if (!PyArg_ParseTuple(args, "i", &item_idx))
    return NULL;

  if (!self->ptr) return NULL;

  if (!check_constraints(self, item_idx, false)) {
    return NULL;
  }

  vector<float> v(self->f);
  self->ptr->get_item(item_idx, &v[0]);
  PyObject* l = PyList_New(self->f);
  if (l == NULL) {
    return NULL;
  }
  for (int z = 0; z < self->f; z++) {
    PyObject* dist = PyFloat_FromDouble(v[z]);
    if (dist == NULL) {
      goto error;
    }
    PyList_SetItem(l, z, dist);
  }

  return l;

  error:
    Py_XDECREF(l);
    return NULL;
}


static PyObject*
py_an_add_item(py_annoy *self, PyObject *args, PyObject* kwargs) {
  int32_t item_idx;
  // PyObject* v;
  PyObject* vector_obj;

  static char const * kwlist[] = {"i", "vector", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO", (char**)kwlist, &item_idx, &vector_obj))
    return NULL;

  if (!check_constraints(self, item_idx, true)) {
    return NULL;
  }

  // Step 1. Prepare vector holder (initially size f, may be resized later)
  vector<float> vec(self->f);

  // Step 2. Conversion may CHANGE self->f if it was 0
  int old_f = self->f;
  if (!convert_list_to_vector(vector_obj, self->f, &vec)) { return NULL; };

  // Step 3. If f was 0 before → recreate AnnoyIndex
  if (old_f == 0) {
    // Destroy previous ptr (dimension 0)
    // if (self->ptr) { delete self->ptr; self->ptr = nullptr; }
    if (self->ptr) {
        delete self->ptr;
        self->ptr = nullptr;
    }
    // Infer dimension
    self->f = (int)vec.size();          // Update real f
    // Set default metric if not yet set → future warning
    if (self->metric.empty()) {
        // This keeps coming up, see #368 etc
        PyErr_WarnEx(PyExc_FutureWarning, "The default argument for metric will be removed "
        "in future version of Annoy. Please pass metric='angular' explicitly.", 1);
        self->metric = "angular";         // <-- ADD
        // self->ptr = new AnnoyAngular(self->f);
        // return 0;
    }

    // Update raw_f_obj so Python sees correct value
    Py_XDECREF(self->raw_f_obj);
    self->raw_f_obj = PyLong_FromLong(self->f);

    // Recreate with inferred dimension
    self->ptr = create_index_for_metric(self->f, self->metric);
    if (!self->ptr) {
        PyErr_SetString(PyExc_ValueError, "Failed to recreate index for f, metric");
        return NULL;
    }
  }
  // Now ptr MUST exist
  // if (!self->ptr) return NULL;
  if (!self->ptr) {
      PyErr_SetString(PyExc_RuntimeError,
        "AnnoyIndex not initialized. Pass `f` and/or `metric`, or call add_item(0) first."
      );
      return NULL;
  }

  // Step 4. Normal add_item
  // std::unique_ptr<char, decltype(&free)> error_ptr(nullptr, free);
  // if (!self->ptr->add_item(item_idx, &vec[0], &error_ptr)) {
  //     PyErr_SetString(PyExc_Exception, error_ptr.get());
  //     return NULL;
  // }
  char *error;
  // if (!self->ptr->add_item(item_idx, vec.data(), &error)) {
  //     PyErr_SetString(PyExc_Exception, error);
  //     free(error);
  //     return NULL;
  // }
  if (!self->ptr->add_item(item_idx, &vec[0], &error)) {
    PyErr_SetString(PyExc_Exception, error);
    free(error);
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject *
py_an_on_disk_build(py_annoy *self, PyObject *args, PyObject *kwargs) {
  // filename is received from Python as a borrowed pointer to immutable UTF-8 storage → must be const char*.
  // error is dynamically allocated by Annoy (strdup) → must remain char* so free() is valid.
  const char *filename;
  char *error;

  if (!self->ptr) return NULL;

  static char const * kwlist[] = {"fn", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &filename))
    return NULL;

  if (!self->ptr->on_disk_build(filename, &error)) {
    PyErr_SetString(PyExc_IOError, error);
    free(error);
    return NULL;
  }
  Py_RETURN_TRUE;
}

static PyObject *
py_an_build(py_annoy *self, PyObject *args, PyObject *kwargs) {
  int q;
  int n_jobs = -1;

  if (!self->ptr) return NULL;

  static char const * kwlist[] = {"n_trees", "n_jobs", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i|i", (char**)kwlist, &q, &n_jobs))
    return NULL;

  bool res;
  char* error;
  Py_BEGIN_ALLOW_THREADS;
  res = self->ptr->build(q, n_jobs, &error);
  Py_END_ALLOW_THREADS;
  if (!res) {
    PyErr_SetString(PyExc_Exception, error);
    free(error);
    return NULL;
  }

  Py_RETURN_TRUE;
}


static PyObject *
py_an_unbuild(py_annoy *self) {
  if (!self->ptr)
    return NULL;

  char* error;
  if (!self->ptr->unbuild(&error)) {
    PyErr_SetString(PyExc_Exception, error);
    free(error);
    return NULL;
  }

  Py_RETURN_TRUE;
}


static PyObject *
py_an_unload(py_annoy *self) {
  if (!self->ptr)
    return NULL;

  self->ptr->unload();

  Py_RETURN_TRUE;
}


static PyObject *
py_an_get_distance(py_annoy *self, PyObject *args) {
  int32_t i, j;

  if (!PyArg_ParseTuple(args, "ii", &i, &j))
    return NULL;

  if (!self->ptr) return NULL;

  if (!check_constraints(self, i, false) || !check_constraints(self, j, false)) {
    return NULL;
  }

  double d = self->ptr->get_distance(i,j);
  return PyFloat_FromDouble(d);
}


static PyObject *
py_an_get_n_items(py_annoy *self) {
  if (!self->ptr)
    return NULL;

  int32_t n = self->ptr->get_n_items();
  return PyInt_FromLong(n);
}

static PyObject *
py_an_get_n_trees(py_annoy *self) {
  if (!self->ptr)
    return NULL;

  int32_t n = self->ptr->get_n_trees();
  return PyInt_FromLong(n);
}

static PyObject *
py_an_verbose(py_annoy *self, PyObject *args) {
  int verbose;
  if (!self->ptr)
    return NULL;
  if (!PyArg_ParseTuple(args, "i", &verbose))
    return NULL;

  self->ptr->verbose((bool)verbose);

  Py_RETURN_TRUE;
}


static PyObject *
py_an_set_seed(py_annoy *self, PyObject *args) {
  int q;
  if (!self->ptr)
    return NULL;
  if (!PyArg_ParseTuple(args, "i", &q))
    return NULL;

  self->ptr->set_seed(q);

  Py_RETURN_NONE;
}


static PyObject *
py_an_serialize(py_annoy *self, PyObject *args, PyObject *kwargs) {
  if (!self->ptr) return NULL;

  // vector<uint8_t> bytes = self->ptr->serialize(NULL);

  char* error = nullptr;
  vector<uint8_t> bytes = self->ptr->serialize(&error);
  if (bytes.empty() && error) {
      PyErr_SetString(PyExc_RuntimeError, error);
      free(error);
      return NULL;
  }

  return PyBytes_FromStringAndSize((const char*)bytes.data(), bytes.size());
}


static PyObject *
py_an_deserialize(py_annoy *self, PyObject *args, PyObject *kwargs) {
  PyObject* bytes_object;
  bool prefault = false;

  static char const * kwlist[] = {"bytes", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "S|b", (char**)kwlist, &bytes_object, &prefault))
    return NULL;

  if (!self->ptr) return NULL;

  if (bytes_object == NULL) {
    PyErr_SetString(PyExc_TypeError, "Expected bytes but NULL");
    return NULL;
  }

  if (!PyBytes_Check(bytes_object)) {
    PyErr_SetString(PyExc_TypeError, "Expected bytes");
    return NULL;
  }

  Py_ssize_t length = PyBytes_Size(bytes_object);
  // uint8_t* raw_bytes = (uint8_t*)PyBytes_AsString(bytes_object);
  uint8_t* raw_bytes = reinterpret_cast<uint8_t*>(PyBytes_AsString(bytes_object));
  vector<uint8_t> v(raw_bytes, raw_bytes + length);

  // char* error = nullptr;
  char *error;
  if (!self->ptr->deserialize(&v, prefault, &error)) {
      PyErr_SetString(PyExc_IOError, error);
      free(error);
      return NULL;
  }

  Py_RETURN_TRUE;
}


static PyMethodDef AnnoyMethods[] = {
  {"add_item",(PyCFunction)py_an_add_item, METH_VARARGS | METH_KEYWORDS, "Adds item `i` (any nonnegative integer) with vector `v`.\n\nNote that it will allocate memory for `max(i)+1` items."},
  {"build",(PyCFunction)py_an_build, METH_VARARGS | METH_KEYWORDS, "Builds a forest of `n_trees` trees.\n\nMore trees give higher precision when querying. After calling `build`,\nno more items can be added. `n_jobs` specifies the number of threads used to build the trees. `n_jobs=-1` uses all available CPU cores."},
  {"deserialize", (PyCFunction)py_an_deserialize, METH_VARARGS | METH_KEYWORDS, "Deserializes the index from bytes."},
  {"get_distance",(PyCFunction)py_an_get_distance, METH_VARARGS, "Returns the distance between items `i` and `j`."},
  {"get_item_vector",(PyCFunction)py_an_get_item_vector, METH_VARARGS, "Returns the vector for item `i` that was previously added."},
  {"get_n_items",(PyCFunction)py_an_get_n_items, METH_NOARGS, "Returns the number of items in the index."},
  {"get_n_trees",(PyCFunction)py_an_get_n_trees, METH_NOARGS, "Returns the number of trees in the index."},
  {"get_nns_by_item",(PyCFunction)py_an_get_nns_by_item, METH_VARARGS | METH_KEYWORDS, "Returns the `n` closest items to item `i`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances."},
  {"get_nns_by_vector",(PyCFunction)py_an_get_nns_by_vector, METH_VARARGS | METH_KEYWORDS, "Returns the `n` closest items to vector `vector`.\n\n:param search_k: the query will inspect up to `search_k` nodes.\n`search_k` gives you a run-time tradeoff between better accuracy and speed.\n`search_k` defaults to `n_trees * n` if not provided.\n\n:param include_distances: If `True`, this function will return a\n2 element tuple of lists. The first list contains the `n` closest items.\nThe second list contains the corresponding distances."},
  {"load",	(PyCFunction)py_an_load, METH_VARARGS | METH_KEYWORDS, "Loads (mmaps) an index from disk."},
  {"on_disk_build",(PyCFunction)py_an_on_disk_build, METH_VARARGS | METH_KEYWORDS, "Build will be performed with storage on disk instead of RAM."},
  {"save",	(PyCFunction)py_an_save, METH_VARARGS | METH_KEYWORDS, "Saves the index to disk."},
  {"serialize",  (PyCFunction)py_an_serialize, METH_VARARGS | METH_KEYWORDS, "Serializes the index to bytes."},
  {"set_seed",(PyCFunction)py_an_set_seed, METH_VARARGS, "Sets the seed of Annoy's random number generator."},
  {"unbuild",(PyCFunction)py_an_unbuild, METH_NOARGS, "Unbuilds the tree in order to allows adding new items.\n\nbuild() has to be called again afterwards in order to\nrun queries."},
  {"unload",(PyCFunction)py_an_unload, METH_NOARGS, "Unloads an index from disk."},
  {"verbose",(PyCFunction)py_an_verbose, METH_VARARGS, ""},
  {NULL, NULL, 0, NULL}		 /* Sentinel */
};


static PyTypeObject PyAnnoyType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "annoy.Annoy",                            /*tp_name*/
  sizeof(py_annoy),                         /*tp_basicsize*/
  0,                                        /*tp_itemsize*/
  (destructor)py_an_dealloc,                /*tp_dealloc*/
  0,                                        /*tp_print*/
  0,                                        /*tp_getattr*/
  0,                                        /*tp_setattr*/
  0,                                        /*tp_compare*/
  0,                                        /*tp_repr*/
  0,                                        /*tp_as_number*/
  0,                                        /*tp_as_sequence*/
  0,                                        /*tp_as_mapping*/
  0,                                        /*tp_hash */
  0,                                        /*tp_call*/
  0,                                        /*tp_str*/
  0,                                        /*tp_getattro*/
  0,                                        /*tp_setattro*/
  0,                                        /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
  ANNOY_DOC,                                /* tp_doc */
  0,                                        /* tp_traverse */
  0,                                        /* tp_clear */
  0,                                        /* tp_richcompare */
  0,                                        /* tp_weaklistoffset */
  0,                                        /* tp_iter */
  0,                                        /* tp_iternext */
  AnnoyMethods,                             /* tp_methods */
  py_annoy_members,                         /* tp_members */
  py_annoy_getset,                          /* tp_getset */
  0,                                        /* tp_base */
  0,                                        /* tp_dict */
  0,                                        /* tp_descr_get */
  0,                                        /* tp_descr_set */
  0,                                        /* tp_dictoffset */
  (initproc)py_an_init,                     /* tp_init */
  0,                                        /* tp_alloc */
  py_an_new,                                /* tp_new */
};

static PyMethodDef module_methods[] = {
  {NULL}	/* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "annoylib",          /* m_name */
    ANNOY_DOC,           /* m_doc */
    -1,                  /* m_size */
    module_methods,      /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#endif

PyObject *create_module(void) {
  PyObject *m;

  if (PyType_Ready(&PyAnnoyType) < 0)
    return NULL;

#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&moduledef);
#else
  m = Py_InitModule("annoylib", module_methods);
#endif

  if (m == NULL)
    return NULL;

  Py_INCREF(&PyAnnoyType);
  PyModule_AddObject(m, "Annoy", (PyObject *)&PyAnnoyType);
  return m;
}

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_annoylib(void) {
    return create_module();      // it should return moudule object in py3
  }
#else
  PyMODINIT_FUNC initannoylib(void) {
    create_module();
  }
#endif


// vim: tabstop=2 shiftwidth=2
