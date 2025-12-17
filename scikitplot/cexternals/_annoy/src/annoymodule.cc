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

// std::optional<int32_t> / std::nullopt style semantics are emulated in C-API as:
//   (!has_value)  ->  Py_RETURN_NONE

#define PY_SSIZE_T_CLEAN

#ifndef NPY_NO_DEPRECATED_API
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif

// NumPy C-API -----------------------------------------------------------------
// #include <numpy/arrayobject.h>

// Core Annoy C++ implementation -----------------------------------------------
#include "annoylib.h"
#include "kissrandom.h"

// Python C-API ----------------------------------------------------------------
#include <Python.h>

#include "bytesobject.h"
#include "structmember.h"  // PyMemberDef, T_INT, READONLY  // TODO: ?Some fields deprecated in Python 3.11+

// System / STL ----------------------------------------------------------------
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <algorithm>  // std::transform
#include <cctype>  // std::tolower
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>  // std::bad_alloc
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER) && _MSC_VER == 1500
  typedef signed __int32 int32_t;
#else
  #include <stdint.h>
#endif

#if PY_MAJOR_VERSION >= 3
  #define IS_PY3K
#endif

#ifndef Py_TYPE
  #define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif

#ifdef IS_PY3K
  #define PyInt_FromLong PyLong_FromLong
#endif

// AVX / compiler info (exposed in module docstring) --------------------------
#if defined(ANNOYLIB_USE_AVX512)
  #define AVX_INFO "Using 512-bit AVX instructions"
#elif defined(ANNOYLIB_USE_AVX256)  // Placeholder
  #define AVX_INFO "Using 256-bit AVX instructions"
#elif defined(ANNOYLIB_USE_AVX128)  // Placeholder
  #define AVX_INFO "Using 128-bit AVX instructions"
#elif defined(ANNOYLIB_USE_AVX)
  #define AVX_INFO "Using AVX instructions"
#else
  #define AVX_INFO "Not using AVX instructions"
#endif

#if defined(_MSC_VER)
  #define COMPILER_INFO "Compiled with MSVC"
#elif defined(__GNUC__)
  #define COMPILER_INFO "Compiled with GCC/Clang"
#else
  #define COMPILER_INFO "Compiled on unknown toolchain"
#endif

// Minimal C-extension docstring. Rich, user-facing docs live in the Python
// layer (annoy.Annoy / annoylib.Annoy / AnnoyIndex).
static const char ANNOY_MOD_DOC[] =
    COMPILER_INFO ". " AVX_INFO ".\n"
    "\n"
    "High-performance approximate nearest neighbours (Annoy) C++ core.\n"
    "\n"
    "This module is a low-level backend (``annoylib``). It exposes the\n"
    "C++-powered :class:`~.Annoy` type. For day-to-day work, prefer the\n"
    "high-level Python API in :mod:`~scikitplot.annoy`::\n"
    "\n"
    "    >>> from annoy import Annoy, AnnoyIndex\n"
    "    >>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex\n"
    "    >>> from scikitplot.annoy import Annoy, AnnoyIndex, Index\n";

// Safe “return self” helper macro for methods that mutate in-place and
// return the same Python object (fluent style).
// Chaining: a.build(...).save(...).info()
#ifdef PY_RETURN_SELF
  #undef PY_RETURN_SELF
#endif
#define PY_RETURN_SELF           \
  do {                           \
    Py_INCREF(self);             \
    return reinterpret_cast<PyObject*>(self); \
  } while (0)

// Bring core Annoy namespace into this translation unit. The Python
// wrapper is strictly C-API, but delegates all heavy lifting to Annoy::.
using namespace Annoy;

// core Kiss64Random already uses a fixed default seed, so if the user doesn’t
// Deterministic default seed for Annoy indices.
// Using an explicit constant makes runs reproducible as soon as the user calls set_seed().
// static const uint64_t ANNOY_DEFAULT_SEED = 1729ULL;

// Build policy: single-threaded vs multi-threaded tree construction
#ifdef ANNOYLIB_MULTITHREADED_BUILD
  typedef AnnoyIndexMultiThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#else
  typedef AnnoyIndexSingleThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#endif

// ---------------------------------------------------------------------------
// Temporary index file helper (for on-disk build / future pipeline support)
// ---------------------------------------------------------------------------
#if defined(_WIN32)
  #include <windows.h>

  static std::string get_temp_annoy_filename() {
    char path[MAX_PATH];
    char file[MAX_PATH];

    if (!GetTempPathA(MAX_PATH, path)) {
      // Fallback: current directory
      std::snprintf(path, sizeof(path), ".");
    }
    if (!GetTempFileNameA(path, "ann", 0, file)) {
      // As a last resort, use a simple pattern in CWD
      return std::string("annoy_tmp_index.annoy");
    }
    return std::string(file);
  }

#else   // POSIX --------------------------------------------------------------

  #include <unistd.h>

  static std::string get_temp_annoy_filename() {
    char tmpl[] = "/tmp/annoy.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd != -1) {
      close(fd);
    }
    // mkstemp replaces XXXXXX in-place; tmpl is now a unique path
    return std::string(tmpl);
  }

#endif  // _WIN32

// Lightweight RAII for temporary files. Ensures that any temporary index
// backing file is removed even if an exception / Python error occurs.
struct TempAnnoyIndexFile {
  std::string path;

  TempAnnoyIndexFile()
    : path(get_temp_annoy_filename()) {}

  ~TempAnnoyIndexFile() {
    if (!path.empty()) {
      std::remove(path.c_str());
    }
  }

  TempAnnoyIndexFile(const TempAnnoyIndexFile&) = delete;
  TempAnnoyIndexFile& operator=(const TempAnnoyIndexFile&) = delete;
};

// Check if a file exists and raise a Python FileNotFoundError if not.
// Returns true on success, false if the error has been set.
static inline bool file_exists(const char* filename) {
  if (!filename || !*filename) {
    PyErr_SetString(PyExc_ValueError, "filename is empty");
    return false;
  }

  struct stat st;
  if (stat(filename, &st) != 0) {
    PyErr_SetFromErrnoWithFilename(PyExc_FileNotFoundError, filename);
    return false;
  }
  return true;
}

// Safely free a C string if set. Used for Annoy error buffers allocated
// on the C++ side with malloc / strdup.
static inline void safe_free(char** ptr) {
  if (ptr && *ptr) {
    free(*ptr);
    *ptr = NULL;  // nullptr
  }
}

// RAII wrapper for Annoy error buffers (char* error). Any _index method that
// takes a char** error out-parameter can be wrapped with ScopedError to
// guarantee cleanup.
struct ScopedError {
  char* err;

  ScopedError() : err(NULL) {}
  ~ScopedError() { safe_free(&err); }

  // Allow passing &ScopedError to APIs expecting char**.
  char** operator&() { return &err; }

  // Convenience accessor.
  const char* c_str() const { return err ? err : ""; }

  // Transfer ownership out if needed.
  char* release() {
    char* tmp = err;
    err = NULL;
    return tmp;
  }
};

// ---------------------------------------------------------------------
// Concrete interface instantiation used by the Python wrapper
// ---------------------------------------------------------------------

// Only use NULL (0) when maintaining legacy C code.
// nullptr is std::nullptr_t, recommended in C++11 and later (not C).
template class Annoy::AnnoyIndexInterface<int32_t, float>;

// R"( ... )";
static const char kAnnoyTypeDoc[] =
COMPILER_INFO ". " AVX_INFO "."
"\n"
R"ANN(
Annoy(f=None, metric=None)

Approximate Nearest Neighbors index (Annoy) with a small, lazy C-extension wrapper.

Parameters
----------
f : int or None, optional
    Vector dimension. If ``0`` or ``None``, dimension may be inferred from the
    first vector passed to ``add_item`` (lazy mode).
metric : str or None, optional
    Distance metric (one of 'angular', 'euclidean', 'manhattan', 'dot', 'hamming').
    If omitted and ``f > 0``, defaults to ``'angular'`` (cosine-like).
    If omitted and ``f == 0``, metric may be set later before construction.

Attributes
----------
f : int
    Vector dimension. ``0`` means "unknown / lazy".
metric : {'angular', 'euclidean', 'manhattan', 'dot', 'hamming'} or None
    Canonical metric name, or None if not configured yet (lazy).
on_disk_path : str or None
    Path for on-disk build/load. None if not configured.

Notes
-----
- The underlying C++ index is created lazily when enough information exists.
  The index is created when both ``f > 0`` and ``metric`` are known, or on first
  operation that requires it.
- Once the C++ index is created, ``f`` and ``metric`` become immutable to keep
  the object consistent and avoid undefined behavior.
- This wrapper stores user configuration (seed/verbose) even before the C++ index
  exists and applies it deterministically upon construction.

Developer Notes:

- Source of truth:
  * ``f`` (int) and ``metric_id`` (enum) describe configuration.
  * ``ptr`` is NULL when index is not constructed.
- Invariant:
  * ``ptr != NULL`` implies ``f > 0`` and ``metric_id != METRIC_UNKNOWN``.

Examples
--------
>>> from annoy import Annoy, AnnoyIndex
>>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
>>> from scikitplot.annoy import Annoy, AnnoyIndex, Index
)ANN";

static const char kFDoc[] =
R"FDOC(
Vector dimension.

Returns
-------
int
    Dimension of each item vector. ``0`` means unknown / lazy.

Notes
-----
- ``f`` may be set only before the underlying C++ index is created.
- After construction, attempting to change ``f`` raises AttributeError.
)FDOC";

static const char kMetricDoc[] =
R"METRIC(
Distance metric for the index. Valid values:

* 'angular' -> Cosine-like distance on normalized vectors.
* 'euclidean' -> L2 distance.
* 'manhattan' -> L1 distance.
* 'dot' -> Negative dot-product distance (inner product).
* 'hamming' -> Hamming distance for binary vectors.

Aliases (case-insensitive):

* angular : cosine
* euclidean : l2, lstsq
* manhattan : l1, cityblock, taxicab
* dot : @, ., dotproduct, inner, innerproduct
* hamming : hamming

Returns
-------
str or None
    Canonical metric name, or None if not configured yet.

Notes
-----
- The metric may be set only before the underlying C++ index is created.
- After construction, attempting to change ``metric`` raises AttributeError.

.. seealso::
  * :py:mod:`~scipy.spatial.distance.cosine`
  * :py:mod:`~scipy.spatial.distance.euclidean`
  * :py:mod:`~scipy.spatial.distance.cityblock`
  * :py:mod:`~scipy.sparse.coo_array.dot`
  * :py:mod:`~scipy.spatial.distance.hamming`
)METRIC";

static const char kOnDiskPathDoc[] =
R"ODP(
Path for on-disk build/load.

Returns
-------
str or None
    Filesystem path used for on-disk operations, or None if not configured.

Notes
-----
- Clearing/changing this while an on-disk index is active is disallowed.
  Call ``unload()`` first.
)ODP";

// ---------------------------------------------------------------------
// HammingWrapper
//
// A thin adapter that exposes a float-based AnnoyIndexInterface
// while internally using a binary (uint64_t) Hamming index.
//
// Canonical semantics:
//
//   * "index|indice"  : integer ID returned by Annoy
//   * "index"         : 0..n-1 row position in a result set (Python side)
//   * "distance"      : Hamming distance (clipped to [0, f_external])
//   * "embedding"     : binary embedding represented as float[0,1] on the API
//
// The wrapper:
//   - Packs float[0,1] → uint64_t bit-embeddings for storage/search
//   - Unpacks uint64_t → float[0,1] when returning vectors
//   - Adds a small header around the raw Annoy index for robustness
// ---------------------------------------------------------------------
class HammingWrapper : public AnnoyIndexInterface<int32_t, float> {
private:
  // External binary dimension (number of Hamming bits)
  int32_t _f_external;  // number of bits in the user-facing embedding
  // Internal representation: number of uint64_t chunks
  int32_t _f_internal;  // ceil(_f_external / 64)

  // Underlying Annoy index working on packed uint64_t vectors
  AnnoyIndex<int32_t, uint64_t, Hamming, Kiss64Random, AnnoyIndexThreadedBuildPolicy> _index;

  // Header structure for serialization
  struct HammingHeader {
    uint32_t f_external;  // user embedding dimension (bits)
    uint32_t f_internal;  // internal uint64_t words
    uint32_t n_items;     // number of stored indices
    uint32_t reserved;    // reserved for future use (versions, flags, etc.)
  };

  // -------------------------------------------------------------------
  // Packing: float[0,1] embedding → uint64_t bit representation
  //
  // For each bit position b:
  //   embedding[b] > 0.5 → bit = 1
  //   embedding[b] ≤ 0.5 → bit = 0
  // -------------------------------------------------------------------
  void _pack(const float* embedding, uint64_t* packed) const {
    for (int32_t word = 0; word < _f_internal; ++word) {
      packed[word] = 0ULL;
      for (int32_t bit = 0; bit < 64 && word * 64 + bit < _f_external; ++bit) {
        const int32_t pos = word * 64 + bit;
        const uint64_t bit_val = (embedding[pos] > 0.5f) ? 1ULL : 0ULL;
        packed[word] |= (bit_val << bit);
      }
    }
  }

  // -------------------------------------------------------------------
  // Unpacking: uint64_t bit representation → float[0,1] embedding
  //
  // For each bit position b:
  //   bit = 1 → embedding[b] = 1.0f
  //   bit = 0 → embedding[b] = 0.0f
  // -------------------------------------------------------------------
  void _unpack(const uint64_t* packed, float* embedding) const {
    for (int32_t bit = 0; bit < _f_external; ++bit) {
      const int32_t word = bit / 64;
      const int32_t offset = bit % 64;
      const uint64_t bit_val = (packed[word] >> offset) & 1ULL;
      embedding[bit] = static_cast<float>(bit_val);
    }
  }

  // Portable strdup replacement (POSIX strdup is not guaranteed on all toolchains)
  // strdup is POSIX, not standard C++ → can fail on MSVC.
  static char* dup_cstr(const char* s) {
    if (!s) return NULL;
    const size_t n = strlen(s) + 1;
    char* out = (char*)malloc(n);
    if (!out) return NULL;
    memcpy(out, s, n);
    return out;
  }

public:
  // -------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------
  explicit HammingWrapper(int f)
    : _f_external(f),
      _f_internal((f + 63) / 64),
      _index((f + 63) / 64) {}

  // -------------------------------------------------------------------
  // Index operations (AnnoyIndexInterface)
  // -------------------------------------------------------------------
  bool add_item(int32_t indice,
                const float* embedding,
                char** error) override {
    vector<uint64_t> packed(_f_internal, 0ULL);
    _pack(embedding, &packed[0]);
    return _index.add_item(indice, &packed[0], error);
  }

  bool build(int n_trees,
             int n_threads,
             char** error) override {
    return _index.build(n_trees, n_threads, error);
  }

  //  - validates header shape
  //  - passes remaining payload to underlying index
  bool deserialize(
    vector<uint8_t>* byte,
    bool prefault,
    char** error) override {
    if (!byte || byte->empty()) {
      // if (error) *error = strdup("Empty byte vector");
      if (error) *error = dup_cstr("Empty byte vector");
      return false;
    }

    if (byte->size() < sizeof(HammingHeader)) {
      if (error) *error = dup_cstr("Corrupted Hamming index: header too small");
      return false;
    }

    HammingHeader hdr;
    memcpy(&hdr, byte->data(), sizeof(HammingHeader));

    // Basic sanity checks on header
    if (hdr.f_external == 0 ||
        hdr.f_internal == 0 ||
        // Defensive upper bound to avoid absurd allocations
        hdr.f_internal > (1U << 26)) {
      if (error) *error = dup_cstr("Corrupted or invalid Hamming header");
      return false;
    }

    // _f_external = static_cast<int32_t>(hdr.f_external);
    // _f_internal = static_cast<int32_t>(hdr.f_internal);
    // IMPORTANT: _index dimension is fixed at construction time. Do NOT
    // overwrite _f_external/_f_internal here; enforce a strict match instead.
    if (static_cast<int32_t>(hdr.f_external) != _f_external ||
        static_cast<int32_t>(hdr.f_internal) != _f_internal) {
      if (error) {
        char buf[160];
        snprintf(buf, sizeof(buf),
                 "Hamming index dimension mismatch "
                 "(expected f_external=%d, f_internal=%d; got f_external=%u, f_internal=%u)",
                 _f_external, _f_internal, hdr.f_external, hdr.f_internal);
        *error = dup_cstr(buf);
      }
      return false;
    }

    const uint8_t* payload = byte->data() + sizeof(HammingHeader);
    const size_t   payload_size = byte->size() - sizeof(HammingHeader);

    if (payload_size == 0) {
      if (error) *error = dup_cstr("Corrupted Hamming index: missing payload");
      return false;
    }

    vector<uint8_t> idx_byte(payload, payload + payload_size);
    return _index.deserialize(&idx_byte, prefault, error);
  }

  float get_distance(int32_t i,
                     int32_t j) const override {
    // Underlying Hamming index returns an integer distance (uint64_t).
    // We convert to float and clip to [0, _f_external] for robustness.
    const uint64_t d_raw = _index.get_distance(i, j);
    const uint64_t max_d = static_cast<uint64_t>(_f_external);
    const uint64_t d_clipped = (d_raw > max_d) ? max_d : d_raw;
    return static_cast<float>(d_clipped);
  }

  void get_item(int32_t indice,
                float* embedding) const override {
    vector<uint64_t> packed(_f_internal, 0ULL);
    _index.get_item(indice, &packed[0]);
    _unpack(&packed[0], embedding);
  }

  int32_t get_n_items() const override {
    return _index.get_n_items();
  }

  int32_t get_n_trees() const override {
    return _index.get_n_trees();
  }

  // -------------------------------------------------------------------
  // Nearest-neighbour queries (Hamming-specific adapter)
  //
  // Canonical semantics on output:
  //   * result[i]    : indice (Annoy item id)
  //   * distances[i] : Hamming distance as float
  // -------------------------------------------------------------------
  void get_nns_by_item(int32_t    query_indice,
                       size_t     n,
                       int        search_k,
                       vector<int32_t>*  result,
                       vector<float>*    distances) const {
    if (distances) {
      vector<uint64_t> internal_distances;
      _index.get_nns_by_item(query_indice,
                             n,
                             search_k,
                             result,
                             &internal_distances);

      distances->resize(internal_distances.size());
      for (size_t i = 0; i < internal_distances.size(); ++i) {
        uint64_t d = internal_distances[i];
        const uint64_t max_d = static_cast<uint64_t>(_f_external);
        if (d > max_d) d = max_d;
        (*distances)[i] = static_cast<float>(d);
      }
    } else {
      _index.get_nns_by_item(query_indice,
                             n,
                             search_k,
                             result,
                             NULL);
    }
  }

  void get_nns_by_vector(const float*       query_embedding,
                         size_t             n,
                         int                search_k,
                         vector<int32_t>*   result,
                         vector<float>*     distances) const {
    vector<uint64_t> packed_query(_f_internal, 0ULL);
    _pack(query_embedding, &packed_query[0]);

    if (distances) {
      vector<uint64_t> internal_distances;
      _index.get_nns_by_vector(&packed_query[0],
                               n,
                               search_k,
                               result,
                               &internal_distances);

      distances->resize(internal_distances.size());
      for (size_t i = 0; i < internal_distances.size(); ++i) {
        uint64_t d = internal_distances[i];
        const uint64_t max_d = static_cast<uint64_t>(_f_external);
        if (d > max_d) d = max_d;
        (*distances)[i] = static_cast<float>(d);
      }
    } else {
      _index.get_nns_by_vector(&packed_query[0],
                               n,
                               search_k,
                               result,
                               NULL);
    }
  }

  // -------------------------------------------------------------------
  // Disk I/O
  // -------------------------------------------------------------------
  bool load(const char* filename,
            bool        prefault,
            char**      error) override {
    return _index.load(filename, prefault, error);
  }

  bool on_disk_build(const char* filename,
                     char**      error) override {
    return _index.on_disk_build(filename, error);
  }

  bool save(const char* filename,
            bool        prefault,
            char**      error) override {
    return _index.save(filename, prefault, error);
  }

  void set_seed(uint64_t seed) override {
    _index.set_seed(seed);
  }

  // -------------------------------------------------------------------
  // Serialization with HammingHeader in front of the raw index byte
  // -------------------------------------------------------------------
  vector<uint8_t> serialize(char** error) const override {
    vector<uint8_t> byte;

    HammingHeader hdr;
    hdr.f_external = static_cast<uint32_t>(_f_external);
    hdr.f_internal = static_cast<uint32_t>(_f_internal);
    hdr.n_items    = static_cast<uint32_t>(_index.get_n_items());
    hdr.reserved   = 0U;

    const uint8_t* hdr_byte =
        reinterpret_cast<const uint8_t*>(&hdr);
    byte.insert(byte.end(),
                 hdr_byte,
                 hdr_byte + sizeof(HammingHeader));

    // Serialize underlying index
    vector<uint8_t> idx_byte = _index.serialize(error);

    // If serialization failed and an error message is set, propagate failure
    if (idx_byte.empty() && error && *error) {
      return {};
    }

    byte.insert(byte.end(),
                 idx_byte.begin(),
                 idx_byte.end());
    return byte;
  }

  bool unbuild(char** error) override {
    return _index.unbuild(error);
  }

  void unload() override {
    _index.unload();
  }

  void verbose(bool v) override {
    _index.verbose(v);
  }
};

// ======================= Typedefs ========================================
// Concrete Annoy types used in the Python binding.
// These keep the original Annoy naming, but conceptually map to:
//
//   * Angular   → cosine-like distance on embeddings
//   * Euclidean → L2 distance on embeddings
//   * Manhattan → L1 distance on embeddings
//   * Dot       → negative dot product distance
//   * Hamming   → bitwise Hamming distance on binary embeddings
// -------------------------------------------------------------------------
typedef AnnoyIndex<int32_t, float, Angular,   Kiss64Random, AnnoyIndexThreadedBuildPolicy> AnnoyAngular;
typedef AnnoyIndex<int32_t, float, Euclidean, Kiss64Random, AnnoyIndexThreadedBuildPolicy> AnnoyEuclidean;
typedef AnnoyIndex<int32_t, float, Manhattan, Kiss64Random, AnnoyIndexThreadedBuildPolicy> AnnoyManhattan;
typedef AnnoyIndex<int32_t, float, DotProduct,Kiss64Random, AnnoyIndexThreadedBuildPolicy> AnnoyDot;
typedef HammingWrapper AnnoyHamming;

// ======================= MetricId =========================================
// ✅ Enum system: MetricId + metric_from_string() + ensure_index()
enum MetricId : uint8_t {
  METRIC_UNKNOWN = 0,
  METRIC_ANGULAR,
  METRIC_EUCLIDEAN,
  METRIC_MANHATTAN,
  METRIC_DOT,
  METRIC_HAMMING
};
// Convert MetricId → canonical string
static inline const char* metric_to_cstr(MetricId m) {
  switch (m) {
    case METRIC_ANGULAR:   return "angular";
    case METRIC_EUCLIDEAN: return "euclidean";
    case METRIC_MANHATTAN: return "manhattan";
    case METRIC_DOT:       return "dot";
    case METRIC_HAMMING:   return "hamming";
    default:               return NULL;
  }
}

// Trim whitespace from both ends (ASCII whitespace is enough here)
static inline std::string trim(const std::string& s) {
  const auto start = s.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) return std::string();
  const auto end = s.find_last_not_of(" \t\n\r");
  return s.substr(start, end - start + 1);
}

struct MetricAlias { const char* alias; MetricId id; };

static inline MetricId metric_from_string(const char* in) {
  if (!in) return METRIC_UNKNOWN;

  std::string s = trim(std::string(in));
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

  static const MetricAlias aliases[] = {
    {"angular", METRIC_ANGULAR}, {"cosine", METRIC_ANGULAR},
    {"euclidean", METRIC_EUCLIDEAN}, {"l2", METRIC_EUCLIDEAN}, {"lstsq", METRIC_EUCLIDEAN},
    {"manhattan", METRIC_MANHATTAN}, {"l1", METRIC_MANHATTAN},
    {"cityblock", METRIC_MANHATTAN}, {"taxicab", METRIC_MANHATTAN},
    {"dot", METRIC_DOT}, {"@", METRIC_DOT}, {".", METRIC_DOT},
    {"dotproduct", METRIC_DOT}, {"inner", METRIC_DOT}, {"innerproduct", METRIC_DOT},
    {"hamming", METRIC_HAMMING},
  };

  for (size_t i = 0; i < sizeof(aliases) / sizeof(aliases[0]); ++i) {
    if (s == aliases[i].alias) return aliases[i].id;
  }
  return METRIC_UNKNOWN;
}

// ======================= Python Annoy type ================================
//
// Low-level C-extension wrapper around the C++ AnnoyIndexInterface.
//
// This object is intentionally tiny:
//   • ptr    → pointer to the actual C++ index (owns the heavy data)
//   • f      → embedding dimension (number of features)
//   • metric → canonical metric name: "angular", "euclidean", "manhattan",
//              "dot", "hamming"
//   • pending_seed / pending_verbose → configuration requested before the
//                                      C++ index exists (lazy mode).
typedef struct {
  PyObject_HEAD

  AnnoyIndexInterface<int32_t, float>* ptr;  // underlying C++ index dynamic_cast<AnnoyAngular*>(ptr)

  int f;                                     // 0 means "unknown / lazy" (dimension inferred from first add_item)
  MetricId metric_id;                        // METRIC_UNKNOWN means "unknown / lazy"
  // std::string metric;                     // empty char "" or NULL

  // --- Optional on-disk path (for on_disk_build / load) ---
  bool on_disk_active;          // true if ptr is currently backed by disk (load() or on_disk_build())
  std::string on_disk_path;     // empty char "" or NULL => none no active on-disk index

  // --- Pending runtime configuration (before C++ index exists) ---
  uint64_t pending_seed;        // last seed requested via set_seed()
  int      pending_verbose;     // last verbosity level requested
  bool     has_pending_seed;    // whether user explicitly set a seed
  bool     has_pending_verbose; // whether user explicitly set verbosity
} py_annoy;

// ======================= ensure_index =====================================
// ensure_index: safely construct index if not yet created
// NOTE: must be AFTER py_annoy is defined.
static bool ensure_index(py_annoy* self) {
  if (self->ptr) return true;
  if (self->f <= 0) {
    PyErr_SetString(PyExc_RuntimeError, "Index dimension f is not set");
    return false;
  }
  if (self->metric_id == METRIC_UNKNOWN) {
    PyErr_SetString(PyExc_RuntimeError, "Index metric is not set");
    return false;
  }
  try {
    switch (self->metric_id) {
      case METRIC_ANGULAR:   self->ptr = new AnnoyAngular(self->f); break;
      case METRIC_EUCLIDEAN: self->ptr = new AnnoyEuclidean(self->f); break;
      case METRIC_MANHATTAN: self->ptr = new AnnoyManhattan(self->f); break;
      case METRIC_DOT:       self->ptr = new AnnoyDot(self->f); break;
      case METRIC_HAMMING:   self->ptr = new AnnoyHamming(self->f); break;
      default:
        PyErr_SetString(PyExc_RuntimeError, "Internal error: unknown metric");
        return false;
    }
  } catch (const std::bad_alloc&) {
    PyErr_NoMemory();
    return false;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create Annoy index");
    return false;
  }
  if (self->has_pending_seed)    self->ptr->set_seed(self->pending_seed);
  if (self->has_pending_verbose) self->ptr->verbose(self->pending_verbose >= 1);
  return true;
}

// ============== Metric helpers ===========================================

static inline std::string normalize_metric(const std::string& m) {
  // static const std::unordered_map<std::string, std::string> metric_map = {
  //   {"angular",   "angular"}, {"cosine",    "angular"},
  //   {"euclidean", "euclidean"}, {"l2",        "euclidean"}, {"lstsq",     "euclidean"},
  //   {"manhattan", "manhattan"}, {"l1",        "manhattan"},
  //   {"cityblock", "manhattan"}, {"taxicab",   "manhattan"},
  //   {"dot",          "dot"}, {"@",            "dot"}, {".",            "dot"},
  //   {"dotproduct",   "dot"}, {"inner",        "dot"}, {"innerproduct", "dot"},
  //   {"hamming", "hamming"},
  // };
  // std::string s = trim(m);
  // std::transform(
  //   s.begin(), s.end(), s.begin(),
  //   // ::tolower  // technically unsafe for negative char values undefined behavior (UB).
  //   // [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
  //   [](char ch) {
  //     return static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  //   }
  // );
  // const auto it = metric_map.find(s);
  // if (it != metric_map.end()) return it->second;
  // return std::string();  // unknown
  MetricId id = metric_from_string(m.c_str());
  const char* c = metric_to_cstr(id);
  return c ? std::string(c) : std::string();
}

// Single source of truth for the metric is `py_annoy::metric_id` (MetricId).
//
// - metric_from_string(const char*) parses user input (aliases included) -> MetricId
// - metric_to_cstr(MetricId) converts MetricId -> canonical metric name
//
// We intentionally avoid RTTI / dynamic_cast so this extension can be built with
// RTTI disabled (e.g. -fno-rtti) while keeping the wrapper and C++ core in sync.
//
// ===================== Annoy Helper / Utility =============================
// Helper: validate a row identifier ("index") against the current index.
//
// Parameters
// ----------
// self     : py_annoy*
// indice   : int32_t   (ID previously passed to add_item())
// building : bool      (true while we are still adding before build())
static bool check_constraints(
  py_annoy* self,
  int32_t   indice,
  bool      building) {
  if (indice < 0) {
    PyErr_SetString(PyExc_IndexError,
                    "index (row id) cannot be negative");
    return false;
  }

  // During build we allow gaps; after build, indices must be in-range.
  if (!building && self->ptr) {
    const int32_t n_items = self->ptr->get_n_items();
    if (indice >= n_items) {
      PyErr_SetString(
        PyExc_IndexError,
        "index (row id) exceeds current number of samples"
      );
      return false;
    }
  }
  return true;
}

// Convert a Python sequence into a contiguous float embedding vector.
//
// This is the main bridge from Python (list/tuple) → C++ (vector<float>):
//
//   • On first use (expected_f == 0) it *discovers* the dimension,
//     updates `self->f`, and allocates the target vector.
//
//   • On subsequent calls it enforces the same dimension, raising
//     ValueError if the embedding length changes.
//
// The function accepts any Python sequence that PySequence_Fast can
// handle (lists, tuples, many array-likes).
//
// Example (C side):
//   vector<float> w;
//   if (!convert_list_to_vector(self, py_vec, self->f, &w))
//     return NULL;  // Python exception already set
//
static bool convert_list_to_vector_strict(
  PyObject* v,
  int expected_f,
  vector<float>* w) {

  if (expected_f <= 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Index dimension `f` is not set");
    return false;
  }

  PyObject* seq = PySequence_Fast(v, "expected a 1D sequence of floats");
  if (!seq) return false;

  const Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
  if (len != expected_f) {
    PyErr_Format(PyExc_ValueError,
                 "embedding length mismatch: expected %d, got %ld",
                 expected_f, (long)len);
    Py_DECREF(seq);
    return false;
  }

  w->assign((size_t)expected_f, 0.0f);

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
    double val = PyFloat_AsDouble(item);
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      return false;
    }
    (*w)[(size_t)i] = (float)val;
  }

  Py_DECREF(seq);
  return true;
}

static bool convert_list_to_vector_infer(
  py_annoy* self,
  PyObject* v,
  vector<float>* w) {

  PyObject* seq = PySequence_Fast(v, "expected a 1D sequence of floats");
  if (!seq) return false;

  const Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
  const int inferred_f = static_cast<int>(len);

  if (self->ptr) {
    Py_DECREF(seq);
    PyErr_SetString(PyExc_RuntimeError,
                    "Internal error: cannot infer `f` after index construction");
    return false;
  }

  if (inferred_f <= 0) {
    Py_DECREF(seq);
    PyErr_SetString(PyExc_ValueError, "embedding cannot be empty");
    return false;
  }

  self->f = inferred_f;
  w->assign(static_cast<size_t>(inferred_f), 0.0f);

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
    double val = PyFloat_AsDouble(item);
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      return false;
    }
    (*w)[static_cast<size_t>(i)] = static_cast<float>(val);
  }

  Py_DECREF(seq);
  return true;
}

// ============================================================================
// Object lifecycle: tp_new / tp_init / tp_dealloc
// ============================================================================
// ============================= Annoy Dunder =================================
//
// Core Python object lifecycle for AnnoyIndex:
//   - tp_new   : allocate the Python wrapper and construct C++ members
//   - tp_init  : parse (f, metric), optionally create the C++ index
//   - tp_dealloc : destroy the C++ index and free the wrapper
//
// Canonical semantics:
//   * f >= 0           : number of features (dimension); 0 means “infer later”.
//   * metric (string)  : parsed via metric_from_string(..) (aliases supported).
//   * Lazy mode        : you can pass f=None / f=0 and/or omit metric;
//                        the actual index will be constructed on first use.

// tp_new: allocate + initialize placement-new fields
static PyObject* py_an_new(
  PyTypeObject* type,
  PyObject* args,
  PyObject* kwargs) {
  // (void)args; (void)kwargs;  // unused
  py_annoy* self = (py_annoy*)type->tp_alloc(type, 0);
  if (!self) return NULL;

  self->ptr = NULL;
  self->f = 0;
  self->metric_id = METRIC_UNKNOWN;
  self->on_disk_active = false;

  // Pending configuration: default to “nothing explicitly requested”
  self->pending_seed        = 0ULL;
  self->pending_verbose     = 0;
  self->has_pending_seed    = false;
  self->has_pending_verbose = false;

  // Construct std::string members (placement new because this is a C struct)
  // new (&self->on_disk_path) std::string(); self->on_disk_path.clear();
  try {
    new (&self->on_disk_path) std::string();
    self->on_disk_path.clear();
  } catch (const std::bad_alloc&) {
    PyErr_NoMemory();
    Py_TYPE(self)->tp_free((PyObject*)self);
    return NULL;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Annoy object");
    Py_TYPE(self)->tp_free((PyObject*)self);
    return NULL;
  }
  // (must-do): don’t use PY_RETURN_SELF in py_an_new
  return (PyObject*)self;
}

// tp_init: handle initialization logic and eager index creation
// tp_init must return 0 on success, -1 on failure
static int py_an_init(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  // Allow re-initialization safely (rare but possible in CPython)
  if (self->ptr) {
    delete self->ptr;
    self->ptr = NULL;
  }

  self->f = 0;
  self->metric_id = METRIC_UNKNOWN;
  self->on_disk_path.clear();
  self->on_disk_active = false;

  // Reset pending configuration as well (true re-init).
  self->pending_seed        = 0ULL;
  self->pending_verbose     = 0;
  self->has_pending_seed    = false;
  self->has_pending_verbose = false;

  // Python signature (public):
  //   Annoy(), Annoy(f=0, metric='angular')
  //
  // Here we accept:
  //   f      : int or None (None/0 → infer later)
  //   metric : str or None (None/empty → lazy/unknown unless f>0, see below)
  PyObject*   f_obj      = NULL;
  PyObject*   metric_obj = NULL;
  const char* metric_c   = NULL;
  // parse by PyArg_ParseTuple or PyArg_ParseTupleAndKeywords
  // "O|s" f (required, PyObject), metric (optional, const char*)
  static const char* kwlist[] = {"f", "metric", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|OO", (char**)kwlist, &f_obj, &metric_obj)) {
    return -1;
  }

  if (metric_obj && metric_obj != Py_None) {
    if (!PyUnicode_Check(metric_obj)) {
      PyErr_SetString(PyExc_TypeError, "`metric` must be a string or None");
      return -1;
    }
    metric_c = PyUnicode_AsUTF8(metric_obj);
    if (!metric_c) return -1;
  }

  // ---- dimension (f) ----
  if (!f_obj || f_obj == Py_None) {
    self->f = 0; // lazy
  } else {
    long fv = PyLong_AsLong(f_obj);
    if (fv == -1 && PyErr_Occurred()) return -1;
    if (fv < 0) {
      PyErr_SetString(PyExc_ValueError,
        "`f` must be non-negative (0 means infer from first vector)");
      return -1;
    }
    // self->f = (int)fv;
    self->f = static_cast<int>(fv);
  }

  // ---- metric ----
  // Validate metric (aliases supported). We set a Python error on failure.
  if (metric_c) {
    MetricId id = metric_from_string(metric_c);
    if (id == METRIC_UNKNOWN) {
      PyErr_SetString(PyExc_ValueError,
        "Invalid metric; expected one of: angular, euclidean, manhattan, dot, hamming.");
      return -1;
    }
    self->metric_id = id;
  }

  // default metric if f>0 and metric omitted
  if (self->f > 0 && self->metric_id == METRIC_UNKNOWN) {
    PyErr_WarnEx(PyExc_FutureWarning,
      "The default argument for metric will be removed in a future version. "
      "Please pass metric='angular' explicitly.", 1);
    self->metric_id = METRIC_ANGULAR;
  }

  // Construct the underlying C++ index eagerly only when both f and metric are known.
  //
  // Cases:
  //   * f > 0 and metric set      → create index now (eager mode)
  //   * f <= 0 or metric empty    → leave ptr == NULL (lazy mode);
  //                                 index will be created in py_an_add_item / load.
  // eager build only if both known
  if (self->f > 0 && self->metric_id != METRIC_UNKNOWN) {
    if (!ensure_index(self)) return -1;
  }
  return 0;
}

// tp_dealloc: safe destruction with Py_CLEAR for GC safety
// tp_dealloc: destroy C++ resources and free the Python wrapper
static void py_an_dealloc(py_annoy* self) {
  // if (!self) return;
  // 1) Release OS-backed resources first
  if (self->ptr) {
    // unload() should be idempotent in Annoy core
    // but we guard anyway in case of future changes.
    try {
      self->ptr->unload();
    } catch (...) {
      // Never let exceptions escape tp_dealloc
    }
    delete self->ptr;
    self->ptr = NULL;
  }
  // 2) Destroy placement-new std::string members
  try {
    self->on_disk_path.~basic_string();
  } catch (...) {
    // ignore destruction errors
  }
  // 3) Free the Python object
  // Py_CLEAR(self->ptr);  // Safe GC clear and refcount protection
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// ===================== Annoy Attr =========================

// The metric is exposed via the get/set table (py_annoy_getset).
// Expose only the core numeric attribute via tp_members.
// Internal / debug-only snapshot (must be READONLY to prevent bypassing setters)
// _f, _metric_id  (and _on_disk_path via getset alias)
static PyMemberDef py_annoy_members[] = {
  {
    (char*)"_f",
    T_INT, offsetof(py_annoy, f),
    // READONLY is mandatory: otherwise obj._f = -5 bypasses your .f setter and breaks sync.
    READONLY,  // 0
    (char*)"internal: raw f (dimension) value (read-only). Use .f property instead."
  },
  {
    (char*)"_metric_id",
    T_UBYTE, offsetof(py_annoy, metric_id),
    // READONLY is mandatory: otherwise obj._metric_id = 0 bypasses your .metric setter and breaks sync.
    READONLY,  // 0
    (char*)"internal: raw metric id value (read-only). Use .metric property instead."
  },
  // {
  //   (char*)"_on_disk_path",
  //   T_UBYTE, offsetof(py_annoy, on_disk_path),
  //   // READONLY is mandatory: otherwise obj.on_disk_path = "." bypasses your .on_disk_path setter and breaks sync.
  //   READONLY,  // 0
  //   (char*)"internal: raw on_disk_path value (read-only). Use .on_disk_path property instead."
  // },
  {NULL}  /* Sentinel */
};

// ===================== Getters/Setters (CORRECT signatures) ===============
// getter: PyObject* (py_annoy*, void*)
// setter: int (py_annoy*, PyObject*, void*)

// Getter for 'f'
static PyObject* py_annoy_get_f(
  py_annoy* self,
  void*) {
  return PyLong_FromLong((long)self->f);
}

// Setter for 'f'
static int py_annoy_set_f(
  py_annoy* self,
  PyObject* value,
  void*) {
  if (!value || value == Py_None) {
    PyErr_SetString(PyExc_ValueError, "f cannot be None (use 0 for lazy inference)");
    return -1;
  }
  if (!PyLong_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "f must be an integer");
    return -1;
  }
  if (self->ptr) {
    PyErr_SetString(PyExc_AttributeError, "Cannot change f after the index has been created.");
    return -1;
  }
  long fv = PyLong_AsLong(value);
  if (fv == -1 && PyErr_Occurred()) return -1;
  if (fv < 0) {
    PyErr_SetString(PyExc_ValueError, "f must be non-negative (0 means infer from first vector)");
    return -1;
  }
  self->f = (int)fv;
  return 0;
}

// Getter for 'metric'
static PyObject* py_annoy_get_metric(
  py_annoy* self,
  void*) {
  const char* m = metric_to_cstr(self->metric_id);
  if (!m) Py_RETURN_NONE;
  return PyUnicode_FromString(m);
}

// Setter for 'metric'
static int py_annoy_set_metric(
  py_annoy* self,
  PyObject* value,
  void*) {
  if (value == NULL) {
    PyErr_SetString(PyExc_AttributeError, "Cannot delete metric attribute");
    return -1;
  }
  // Allow resetting to "unknown / lazy" before the C++ index exists.
  if (value == Py_None) {
    if (self->ptr) {
      PyErr_SetString(PyExc_AttributeError,
        "Cannot unset metric after the index has been created.");
      return -1;
    }
    self->metric_id = METRIC_UNKNOWN;
    return 0;
  }
  if (!PyUnicode_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "metric must be a string (or None)");
    return -1;
  }
  if (self->ptr) {
    PyErr_SetString(PyExc_AttributeError,
      "Cannot change metric after the index has been created.");
    return -1;
  }
  const char* s = PyUnicode_AsUTF8(value);
  if (!s) return -1;
  MetricId id = metric_from_string(s);
  if (id == METRIC_UNKNOWN) {
    PyErr_SetString(PyExc_ValueError,
      "Invalid metric. Valid options: angular, euclidean, manhattan, dot, hamming.");
    return -1;
  }
  self->metric_id = id;
  return 0;
}

// Getter for 'on_disk_path'
static PyObject* py_annoy_get_on_disk_path(
  py_annoy* self,
  void*) {
  if (self->on_disk_path.empty()) Py_RETURN_NONE;
  return PyUnicode_FromStringAndSize(
    self->on_disk_path.c_str(), (Py_ssize_t)self->on_disk_path.size()
  );
}

// Setter for 'on_disk_path'
static int py_annoy_set_on_disk_path(
  py_annoy* self,
  PyObject* value,
  void*) {
  if (!value || value == Py_None) {
    if (self->on_disk_active) {
      PyErr_SetString(PyExc_AttributeError,
        "Cannot clear on_disk_path while an on-disk index is active. Call unload() first.");
      return -1;
    }
    self->on_disk_path.clear();
    return 0;
  }
  if (!PyUnicode_Check(value)) {
    PyErr_SetString(PyExc_TypeError, "on_disk_path must be a string or None");
    return -1;
  }
  const char* s = PyUnicode_AsUTF8(value);
  if (!s) return -1;

  if (self->on_disk_active) {
    PyErr_SetString(PyExc_AttributeError,
      "Cannot change on_disk_path while an on-disk index is active. Call unload() first.");
    return -1;
  }
  self->on_disk_path.assign(s);
  return 0;
}

// ===================== Get/Set table ======================================

static PyGetSetDef py_annoy_getset[] = {
  {
    (char*)"f",
    (getter)py_annoy_get_f,
    (setter)py_annoy_set_f,
    (char*)kFDoc,
    NULL
  },

  {
    (char*)"metric",
    (getter)py_annoy_get_metric,
    (setter)py_annoy_set_metric,
    (char*)kMetricDoc,
    NULL
  },

  {
    (char*)"on_disk_path",
    (getter)py_annoy_get_on_disk_path,
    (setter)py_annoy_set_on_disk_path,
    (char*)kOnDiskPathDoc,
    NULL
  },

  {
    (char*)"_on_disk_path",
    (getter)py_annoy_get_on_disk_path,
    NULL,  // read-only alias of on_disk_path (prevents bypassing validation)
    (char*)"internal: alias of on_disk_path (read-only). Use .on_disk_path to set.",
    NULL
  },

  {NULL}  /* Sentinel */
};

// ======================= Annoy methods ====================================
//
// Verbosity levels (inspired by CatBoost / XGBoost):
//   -2, -1 : fatal / critical (currently mapped to "quiet" at C++ level)
//    0     : warning / normal (quiet at C++ level)
//    1     : info        (verbose=true)
//    2     : debug (same as info at core, but future-proof)
//
// Internally Annoy only has a boolean `verbose(bool)`;
// we map level >= 1 → verbose=true, level <= 0 → verbose=false.
static PyObject* py_an_verbose(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  int level = 1;  // default; keeps backwards compatibility with verbose(1)
  static const char* kwlist[] = {"level", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", (char**)kwlist, &level))
    return NULL;

  // Clamp
  if (level < -2) level = -2;
  if (level >  2) level =  2;

  // Always remember the user’s choice
  self->pending_verbose     = level;
  self->has_pending_verbose = true;

  // If index not yet created → just store; will apply in py_an_init or
  // when the index is constructed lazily.
  if (!self->ptr) {
    // Chaining: a.build(...).save(...).info()
    PY_RETURN_SELF;  // Py_RETURN_TRUE;  // stored for later, not an error
  }

  bool verbose_flag = (level >= 1);
  self->ptr->verbose(verbose_flag);
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF;  // Py_RETURN_TRUE;
}

// Seed control:
//   set_seed(seed: int = 0) -> None | Annoy
//
// The underlying RNG is deterministic for a given `seed`,
// so calling `set_seed` with the same value and same data / n_trees
// yields reproducible trees and queries (subject to CPU / threading).
//
// If the user does NOT call set_seed, core Annoy still uses a fixed
// internal default seed (see kissrandom.h), so behaviour is already
// deterministic. set_seed() is for explicit control or experimentation.
//
// If called before the index is created, we just store the seed
// and apply it when the C++ index appears.
static PyObject* py_an_set_seed(
  py_annoy*  self,
  PyObject*  args,
  PyObject*  kwargs) {
  // long seed_arg = static_cast<long>(0ULL);
  unsigned long long seed_arg = 0ULL;
  static const char* kwlist[] = {"seed", NULL};

  // Optional integer argument: set_seed(seed=0)
  // if (!PyArg_ParseTuple(args, "|K", &seed_arg))
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "|K", (char**)kwlist, &seed_arg)) {
    // Keep a stable, user-friendly error type/message for negatives/overflow
    if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
      PyErr_Clear();
      PyErr_SetString(PyExc_ValueError,
                      "seed must be an integer in the range [0, 2**64 - 1]");
    }
    return NULL;  // Py_RETURN_NONE;
  }
  // if (seed_arg < 0) {
  //   PyErr_SetString(PyExc_ValueError, "seed must be a non-negative integer");
  //   return NULL;  // Py_RETURN_NONE;
  // }

  const uint64_t seed = static_cast<uint64_t>(seed_arg);

  // Remember user preference (for lazy construction)
  self->pending_seed     = seed;
  self->has_pending_seed = true;

  // If index doesn’t exist yet → defer
  // stored for later, not an error
  if (!self->ptr) {
    // Chaining: a.set_seed(...).build(...).save(...)
    PY_RETURN_SELF;  // Py_RETURN_NONE
  }
  // Else apply immediately
  self->ptr->set_seed(seed);
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_TRUE;
}

// joblib serialize / deserialize need to dunder  __reduce__ and _rebuild and/or _deserialize_into_self
// Annoy takes the exact internal data it holds in RAM and writes it out as a binary blob (a byte array).
// That blob contains the whole index:
// . number of trees
// . all nodes
// . the split values used in the trees
// . child pointers (how nodes connect)
// . all item vectors (the stored vectors)
// In short: serialize = a RAM snapshot of the entire index.
// When you deserialize, Annoy reads that binary blob and restores the same in-memory index. No rebuilding is needed.
static PyObject* py_an_serialize(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  (void)args; (void)kwargs;
  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy index is not initialized");
    return NULL;
  }

  // char* error = NULL;
  // vector<uint8_t> byte = self->ptr->serialize(&error);
  // if (byte.empty() && error) {
  //   PyErr_SetString(PyExc_RuntimeError, error);
  //   free(error);
  //   return NULL;
  // }
  ScopedError error;
  vector<uint8_t> byte = self->ptr->serialize(&error.err);
  if (byte.empty() && error.err) {
    PyErr_SetString(PyExc_RuntimeError, error.err);
     return NULL;
   }

  return PyBytes_FromStringAndSize(
      reinterpret_cast<const char*>(byte.data()),
      static_cast<Py_ssize_t>(byte.size()));
}

static PyObject* py_an_deserialize(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  PyObject* byte_object = NULL;
  int prefault = 0;

  static const char* kwlist[] = {"byte", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "S|p",
                                   (char**)kwlist,
                                   &byte_object, &prefault))
    return NULL;

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy index is not initialized");
    return NULL;
  }

  if (!PyBytes_Check(byte_object)) {
    PyErr_SetString(PyExc_TypeError, "Expected byte");
    return NULL;
  }

  Py_ssize_t length = PyBytes_Size(byte_object);
  // uint8_t* raw_byte = reinterpret_cast<uint8_t*>(PyBytes_AsString(byte_object));
  uint8_t* raw_byte =
      reinterpret_cast<uint8_t*>(PyBytes_AsString(byte_object));
  vector<uint8_t> v(raw_byte, raw_byte + length);

  ScopedError error;
  if (!self->ptr->deserialize(&v, prefault != 0, &error.err)) {
    PyErr_SetString(PyExc_IOError, error.err ? error.err : (char*)"deserialize failed");
    return NULL;
  }
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_TRUE;
}

// Internal helper: get memory usage in byte.
// Returns true on success, false if a Python error was set.
static bool get_memory_usage_byte(
  py_annoy* self,
  uint64_t* out_byte) {
  if (!out_byte) return false;

  if (!self->ptr) {
    *out_byte = 0;
    return true;
  }

  ScopedError err;
  vector<uint8_t> tmp = self->ptr->serialize(&err.err);
  if (err.err) {
    PyErr_SetString(PyExc_RuntimeError, err.err);
    return false;
  }
  *out_byte = static_cast<uint64_t>(tmp.size());
  return true;
}

// Approximate memory usage: exact if ANNOY_HAS_GET_N_BYTES is defined,
// otherwise via serialize() size.
static PyObject* py_an_memory_usage(
  py_annoy*  self,
  PyObject*  args,
  PyObject*  kwargs) {
  // (void)args; (void)kwargs;
  // No arguments allowed; enforce this so mistakes are caught early.
  static const char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "", (char**)kwlist))
    return NULL;

  if (!self->ptr)
    Py_RETURN_NONE;

  uint64_t byte = 0;
  if (!get_memory_usage_byte(self, &byte))
    return NULL;  // error already set

  return PyLong_FromUnsignedLongLong(
      static_cast<unsigned long long>(byte));
}

static PyObject* py_an_info(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  // No arguments allowed; enforce this so mistakes are caught early.
  static const char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
    return NULL;
  }

  // -----------------------------
  // Core fields: f, metric, trees
  // -----------------------------
  int f = self->f;

  PyObject* py_metric = NULL;
  const char* metric_c = metric_to_cstr(self->metric_id);
  if (!metric_c) {
    py_metric = Py_None;
    Py_INCREF(Py_None);
  } else {
    py_metric = PyUnicode_FromString(metric_c);
    if (!py_metric)
      return NULL;
  }

  int64_t n_items = 0;
  int64_t n_trees = 0;
  if (self->ptr) {
    n_items = static_cast<int64_t>(self->ptr->get_n_items());
    n_trees = static_cast<int64_t>(self->ptr->get_n_trees());
  }

  // -----------------------------
  // Memory usage: byte + MiB
  // -----------------------------
  uint64_t byte_u64 = 0;
  if (!get_memory_usage_byte(self, &byte_u64)) {
    Py_DECREF(py_metric);
    return NULL;
  }
  double mib = static_cast<double>(byte_u64) / 1024.0 / 1024.0;

  // -----------------------------
  // On-disk path (if any)
  // -----------------------------
  PyObject* path_obj = NULL;
  if (self->on_disk_path.empty()) {
    path_obj = Py_None;
    Py_INCREF(Py_None);
  } else {
    path_obj = PyUnicode_FromString(self->on_disk_path.c_str());
    if (!path_obj) {
      Py_DECREF(py_metric);
      return NULL;
    }
  }

  // -----------------------------
  // Build structured info dict
  // -----------------------------
  PyObject* d = PyDict_New();
  if (!d) {
    Py_DECREF(path_obj);
    return NULL;
  }

  PyObject* py_f     = PyLong_FromLong(f);
  PyObject* py_items = PyLong_FromLongLong(n_items);
  PyObject* py_trees = PyLong_FromLongLong(n_trees);
  PyObject* py_byte  = PyLong_FromUnsignedLongLong(byte_u64);
  PyObject* py_mib   = PyFloat_FromDouble(mib);

  if (!py_f || !py_items || !py_trees || !py_byte || !py_mib) {
    Py_XDECREF(py_f);
    Py_XDECREF(py_items);
    Py_XDECREF(py_trees);
    Py_XDECREF(py_byte);
    Py_XDECREF(py_mib);
    Py_DECREF(path_obj);
    Py_DECREF(d);
    return NULL;
  }

  int ok = 0;
  ok |= PyDict_SetItemString(d, "f", py_f);
  ok |= PyDict_SetItemString(d, "metric", py_metric);
  ok |= PyDict_SetItemString(d, "n_items", py_items);
  ok |= PyDict_SetItemString(d, "n_trees", py_trees);
  ok |= PyDict_SetItemString(d, "memory_usage_byte", py_byte);
  ok |= PyDict_SetItemString(d, "memory_usage_mib", py_mib);
  ok |= PyDict_SetItemString(d, "on_disk_path", path_obj);

  Py_DECREF(py_f);
  Py_DECREF(py_metric);
  Py_DECREF(py_items);
  Py_DECREF(py_trees);
  Py_DECREF(py_byte);
  Py_DECREF(py_mib);
  Py_DECREF(path_obj);

  if (ok != 0) {
    Py_DECREF(d);
    return NULL;
  }
  return d;
}


// add_item: accepts a 1D sequence (embedding) and supports lazy f/metric/init.
// Public Python signature (kw names kept for backward compatibility):
//   add_item(i: int, vector: Sequence[float]) -> Annoy
static PyObject* py_an_add_item(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  int32_t indice;           // Annoy item id (row id)
  PyObject* embedding_obj;  // Python sequence of floats

  // NOTE: kwlist uses "i" and "vector" for backward compatibility,
  // but conceptually they are (indice, embedding).
  static const char* kwlist[] = {"i", "vector", NULL};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "iO", (char**)kwlist, &indice, &embedding_obj)) {
    return NULL;
  }

  // During build stage: allow gaps, but forbid negative ids.
  if (!check_constraints(self, indice, /*building=*/true))
    return NULL;

  // Convert Python sequence → contiguous vector<float>.
  // - If f is unknown (lazy) and index not constructed, infer f from the first vector.
  // - Otherwise enforce exact length match.
  // vector<float> embedding(self->f);
  vector<float> embedding;
  // Infer only if index is not constructed AND f is unknown
  if (!self->ptr && self->f == 0) {
    if (!convert_list_to_vector_infer(self, embedding_obj, &embedding))
      return NULL;
  } else {
    if (!convert_list_to_vector_strict(embedding_obj, self->f, &embedding))
      return NULL;
  }

  // Default metric in truly-lazy mode (no metric configured yet).
  // We intentionally do NOT warn here to keep exploration noise-free.
  if (self->metric_id == METRIC_UNKNOWN) {
    self->metric_id = METRIC_ANGULAR;
  }

  // Ensure underlying C++ index exists (applies pending seed/verbose if set).
  if (!ensure_index(self))
    return NULL;

  // Disallow adding items after the forest is built (prevents silent wrong queries).
  if (self->ptr->get_n_trees() > 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Index is already built; call unbuild() before add_item().");
    return NULL;
  }

  ScopedError error;
  if (!self->ptr->add_item(indice, embedding.data(), &error.err)) {
    PyErr_SetString(PyExc_RuntimeError,
      error.err ? error.err : (char*)"add_item failed");
    return NULL;
  }
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_NONE;
}

static PyObject* py_an_on_disk_build(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  const char* filename = NULL;
  static const char* kwlist[] = {"fn", NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "s", (char**)kwlist, &filename)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Annoy index is not initialized; construct it with Annoy(f, metric) "
        "and add items before on_disk_build().");
    return NULL;
  }

  // NOTE:
  //  * on_disk_build is allowed to create a new file; we do NOT call file_exists().
  //  * Errors (invalid path, permission, etc.) are reported via `error`.
  ScopedError error;
  if (!self->ptr->on_disk_build(filename, &error.err)) {
    PyErr_SetString(PyExc_IOError, error.err ? error.err : (char*)"on_disk_build failed");
    return NULL;
  }

  // Remember the last on-disk path for __repr__ / info()
  self->on_disk_path = filename;
  self->on_disk_active = true;
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_TRUE;
}

static PyObject* py_an_build(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  int n_trees;
  int n_jobs = -1;  // -1 → "auto" in core Annoy

  static const char* kwlist[] = {"n_trees", "n_jobs", NULL};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "i|i", (char**)kwlist, &n_trees, &n_jobs)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy index is not initialized");
    return NULL;
  }

  if (n_trees <= 0 && n_trees != -1) {
    PyErr_SetString(PyExc_ValueError, "n_trees must be a positive integer or -1");
    return NULL;
  }

  bool ok = false;
  ScopedError error;

  // Heavy work → release GIL
  Py_BEGIN_ALLOW_THREADS;
  ok = self->ptr->build(n_trees, n_jobs, &error.err);
  Py_END_ALLOW_THREADS;

  if (!ok) {
    PyErr_SetString(PyExc_RuntimeError, error.err ? error.err : (char*)"build failed");
    return NULL;
  }
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_TRUE;
}

static PyObject* py_an_unbuild(
  py_annoy*  self,
  PyObject*  args,
  PyObject*  kwargs) {
  // No arguments allowed; enforce this so mistakes are caught early.
  static const char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy index is not initialized");
    return NULL;
  }

  ScopedError error;
  if (!self->ptr->unbuild(&error.err)) {
    PyErr_SetString(PyExc_RuntimeError, error.err ? error.err : (char*)"unbuild failed");
    return NULL;
  }
  // Trees are gone, items remain; we keep on_disk_path (still refers to same file, if any).
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_TRUE;
}

// ---------------------------------------------------------------------
// save(fn: str, prefault: bool = False) -> Annoy
//
// Save the current in-memory index to disk. Returns self so you can
// chain calls: index.build(10).save("idx.ann").
// ---------------------------------------------------------------------
// want Annoy().load("idx.ann") to Just Work™, we’d add a small helper:
// Read a tiny header from the .ann file (dimension, metric id).
// Based on that, set self->f / self->metric_id and call ensure_index().
// Then ptr->load(...) on the newly-constructed index.
static PyObject* py_an_save(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  const char* filename = NULL;
  int prefault_flag = 0;
  static const char* kwlist[] = {"fn", "prefault", NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "s|p", (char**)kwlist, &filename, &prefault_flag)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(
      PyExc_RuntimeError,
      "Annoy index is not initialized. "
      "Call add_item() + build() (or load()) before save()."
    );
    return NULL;
  }

  bool prefault = (prefault_flag != 0);

  ScopedError error;
  if (!self->ptr->save(filename, prefault, &error.err)) {
    PyErr_SetString(
      PyExc_IOError,
      error.err ? error.err : (char*)"save failed"
    );
    return NULL;
  }

  // This file is now a valid on-disk representation of this index
  self->on_disk_path = filename;
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_TRUE;
}

// ---------------------------------------------------------------------
// load(fn: str, prefault: bool = False) -> Annoy
//
// Classic Annoy semantics: the index type (f, metric) must already
// be known, i.e. you must have constructed Annoy(f, metric=...)
// before calling load().
//
// If you ever want Annoy().load("file.ann") with no f/metric, you
// would need a small header reader that introspects the file and
// chooses the correct concrete C++ type. We deliberately *do not*
// do that here to stay 100% compatible with legacy Annoy.
// ---------------------------------------------------------------------
static PyObject* py_an_load(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  const char* filename = NULL;
  int prefault_flag = 0;
  static const char* kwlist[] = {"fn", "prefault", NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "s|p", (char**)kwlist, &filename, &prefault_flag)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Annoy index is not initialized.\n"
        "Construct it first, e.g., Annoy(f, metric='angular'), "
        "then call .load(fn).");
    return NULL;
  }

  // Clear FileNotFoundError early if path is bad
  if (!file_exists(filename)) {
    // file_exists already set FileNotFoundError
    return NULL;
  }

  bool prefault = (prefault_flag != 0);

  ScopedError error;
  if (!self->ptr->load(filename, prefault, &error.err)) {
    PyErr_SetString(
      PyExc_IOError,
      error.err ? error.err : (char*)"load failed"
    );
    return NULL;
  }

  // Track backing path for __repr__ / .info()
  self->on_disk_path = filename;
  self->on_disk_active = true;
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_TRUE;
}

// ---------------------------------------------------------------------
// unload() -> Annoy
//
// Drop the mmap / on-disk mapping but keep the Python wrapper.
// ---------------------------------------------------------------------
static PyObject* py_an_unload(
  py_annoy*  self,
  PyObject*  args,
  PyObject*  kwargs) {
  // No arguments allowed; enforce this so mistakes are caught early.
  static const char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Annoy index is not initialized");
    return NULL;
  }

  self->ptr->unload();
  self->on_disk_path.clear();   // no longer backed by any file
  self->on_disk_active = false;
  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_TRUE;
}

// ======================================================================
//  Nearest neighbors → Python conversion
// ======================================================================

// Build Python (indices, distances) from C++ vectors.
//
// If include_distances == 0, returns:
//   list[int]
//
// If include_distances != 0, returns:
//   (list[int], list[float])
static PyObject* get_nns_to_python(
    const vector<int32_t>& indices,
    const vector<float>&   distances,
    int                    include_distances) {

  PyObject* py_indices   = NULL;
  PyObject* py_distances = NULL;
  PyObject* py_tuple     = NULL;

  // IMPORTANT: declare before any possible `goto error;`
  size_t     idx_sz     = 0;
  size_t     dist_sz    = 0;
  Py_ssize_t py_idx_sz  = 0;
  Py_ssize_t py_dist_sz = 0;

  // ------------------------------------------------------------------
  // indices list → list[int]
  // ------------------------------------------------------------------
  idx_sz = indices.size();
  if (idx_sz > static_cast<size_t>(PY_SSIZE_T_MAX)) {
    PyErr_SetString(PyExc_OverflowError,
      "Too many neighbors to convert to a Python list");
    goto error;
  }
  py_idx_sz = static_cast<Py_ssize_t>(idx_sz);

  if ((py_indices = PyList_New(py_idx_sz)) == NULL) {
    goto error;
  }
  for (Py_ssize_t i = 0; i < py_idx_sz; ++i) {
    PyObject* v = PyLong_FromLong(
      static_cast<long>(indices[static_cast<size_t>(i)]));
    if (!v)
      goto error;
    PyList_SET_ITEM(py_indices, i, v); // Steals reference
  }

  // Only indices requested → return list[int]
  if (!include_distances)
    return py_indices;

  // ------------------------------------------------------------------
  // distances list → list[float]
  // ------------------------------------------------------------------
  dist_sz = distances.size();

  // include_distances should mean 1 distance per returned index.
  if (dist_sz != idx_sz) {
    PyErr_SetString(PyExc_RuntimeError,
      "Internal error: Annoy returned mismatched indices and distances");
    goto error;
  }

  if (dist_sz > static_cast<size_t>(PY_SSIZE_T_MAX)) {
    PyErr_SetString(PyExc_OverflowError,
      "Too many distances to convert to a Python list");
    goto error;
  }
  py_dist_sz = static_cast<Py_ssize_t>(dist_sz);

  if ((py_distances = PyList_New(py_dist_sz)) == NULL) {
    goto error;
  }
  for (Py_ssize_t i = 0; i < py_dist_sz; ++i) {
    PyObject* v = PyFloat_FromDouble(
      static_cast<double>(distances[static_cast<size_t>(i)]));
    if (!v)
      goto error;
    PyList_SET_ITEM(py_distances, i, v);  // steals reference
  }

  // ------------------------------------------------------------------
  // Pack (indices, distances) tuple PyTuple_Pack, PyTuple_SET_ITEM
  // ------------------------------------------------------------------
  if ((py_tuple = PyTuple_Pack(2, py_indices, py_distances)) == NULL) {  // steals reference
    goto error;
  }
  Py_DECREF(py_indices);
  Py_DECREF(py_distances);
  return py_tuple;

error:
  Py_XDECREF(py_indices);
  Py_XDECREF(py_distances);
  Py_XDECREF(py_tuple);
  return NULL;
}

// ======================================================================
//  get_nns_by_item (by indice / item id)
// ======================================================================
//
// Python signature:
//
//   get_nns_by_item(
//       i: int,
//       n: int,
//       search_k: int = -1,
//       include_distances: bool = False,
//   ) -> list[int] | tuple[list[int], list[float]]
//
// * i              → indice (row id / item id passed to add_item)
// * n              → number of nearest neighbors
// * search_k       → search budget (-1 means “auto” like classic Annoy)
// * include_distances → if True, also return distances
//
static PyObject* py_an_get_nns_by_item(
  py_annoy*  self,
  PyObject*  args,
  PyObject*  kwargs) {

  int32_t indice;
  int32_t n_neighbors;
  int32_t search_k          = -1;
  int32_t include_distances = 0;

  static const char* kwlist[] = {
    "i",
    "n",
    "search_k",
    "include_distances",
    NULL
  };

  if (!PyArg_ParseTupleAndKeywords(
    args,
    kwargs,
    "ii|ii",
    (char**)kwlist,
    &indice,
    &n_neighbors,
    &search_k,
    &include_distances)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(
      PyExc_RuntimeError,
      "Annoy index is not initialized");
    return NULL;
  }

  if (n_neighbors <= 0) {
    PyErr_SetString(
      PyExc_ValueError,
      "`n` (number of neighbors) must be positive");
    return NULL;
  }

  // Validate indice against current index state
  if (!check_constraints(self, indice, /*building=*/false))
    return NULL;

  vector<int32_t> indice_result;
  vector<float>   distance_result;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_item(
    indice,
    static_cast<size_t>(n_neighbors),
    search_k,
    &indice_result,
    include_distances ? &distance_result : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(indice_result, distance_result, include_distances);
}


// ======================================================================
//  get_nns_by_vector
// ======================================================================
//
// Python signature:
//
//   get_nns_by_vector(
//       vector: Sequence[float],
//       n: int,
//       search_k: int = -1,
//       include_distances: bool = False,
//   ) -> list[int] | tuple[list[int], list[float]]
//
// * vector         → query embedding (length must equal f)
// * n              → number of nearest neighbors
// * search_k       → search budget (-1 means “auto”)
// * include_distances → if True, also return distances
//
static PyObject* py_an_get_nns_by_vector(
  py_annoy*  self,
  PyObject*  args,
  PyObject*  kwargs) {

  PyObject* vector_obj = NULL;
  int32_t   n_neighbors;
  int32_t   search_k          = -1;
  int32_t   include_distances = 0;

  static const char* kwlist[] = {
    "vector",
    "n",
    "search_k",
    "include_distances",
    NULL
  };

  if (!PyArg_ParseTupleAndKeywords(
    args,
    kwargs,
    "Oi|ii",
    (char**)kwlist,
    &vector_obj,
    &n_neighbors,
    &search_k,
    &include_distances)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(
      PyExc_RuntimeError,
      "Annoy index is not initialized. "
      "Create it with Annoy(f, metric='angular') and add_item() before querying.");
    return NULL;
  }

  if (n_neighbors <= 0) {
    PyErr_SetString(
      PyExc_ValueError,
      "`n` (number of neighbors) must be positive");
    return NULL;
  }

  if (self->f <= 0) {
    PyErr_SetString(
      PyExc_RuntimeError,
      "Index dimension `f` is not set. "
      "Call add_item() at least once before get_nns_by_vector().");
    return NULL;
  }

  // Convert Python sequence → C++ embedding vector<float>
  vector<float> query(self->f);
  if (!convert_list_to_vector_strict(vector_obj, self->f, &query)) {
    // convert list to vector already set a Python exception
    return NULL;
  }

  vector<int32_t> indice_result;
  vector<float>   distance_result;

  Py_BEGIN_ALLOW_THREADS;
  self->ptr->get_nns_by_vector(
    query.data(),
    static_cast<size_t>(n_neighbors),
    search_k,
    &indice_result,
    include_distances ? &distance_result : NULL);
  Py_END_ALLOW_THREADS;

  return get_nns_to_python(indice_result, distance_result, include_distances);
}

// ======================================================================
//  Single-row accessors: embedding, distance, counters
//  All functions use (self, args, kwargs) so they can be bound as
//  METH_VARARGS | METH_KEYWORDS in the method table.
// ======================================================================

// get_item_vector / get_index_vector → plain list[float]
//
// Python-facing semantics:
//
//   get_item_vector(i: int) -> list[float]
//
// * i is the Annoy item id (“indice”) you passed to add_item()
// * return is the stored embedding (length == f), as Python list[float]
//
static PyObject* py_an_get_item_vector(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  (void)kwargs;
  int32_t indice;

  static const char* kwlist[] = {"i", NULL};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "i", (char**)kwlist, &indice)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy index is not initialized");
    return NULL;
  }

  // Validate indice against current index state (not building stage)
  if (!check_constraints(self, indice, /*building=*/false))
    return NULL;

  if (self->f <= 0) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Index dimension `f` is not set; cannot fetch embedding vector");
    return NULL;
  }

  // Pull embedding from C++ core
  vector<float> embedding(static_cast<size_t>(self->f));
  self->ptr->get_item(indice, embedding.data());

  // Convert to Python list[float]
  PyObject* py_list = PyList_New(self->f);
  if (!py_list)
    return NULL;

  for (int i = 0; i < self->f; ++i) {
    PyObject* v = PyFloat_FromDouble(
        static_cast<double>(embedding[static_cast<size_t>(i)]));
    if (!v) {
      Py_DECREF(py_list);
      return NULL;
    }
    // Steals reference
    PyList_SET_ITEM(py_list, i, v);
  }

  return py_list;
}


// get_distance → float
//
// Python-facing semantics:
//
//   get_distance(i: int, j: int) -> float
//
static PyObject* py_an_get_distance(
  py_annoy*  self,
  PyObject*  args,
  PyObject*  kwargs) {
  // (void)kwargs;
  int32_t i = 0;
  int32_t j = 0;
  static const char* kwlist[] = {"i", "j", NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "ii", (char**)kwlist, &i, &j)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy index is not initialized");
    return NULL;
  }

  // Validate both indices
  if (!check_constraints(self, i, /*building=*/false) ||
      !check_constraints(self, j, /*building=*/false)) {
    return NULL;
  }

  const double d = static_cast<double>(self->ptr->get_distance(i, j));
  return PyFloat_FromDouble(d);
}


// get_n_items → number of stored samples (rows)
//
// Python-facing semantics:
//
//   get_n_items() -> int
//
static PyObject* py_an_get_n_items(
  py_annoy*  self,
  PyObject*  args,
  PyObject*  kwargs) {
  // No arguments allowed; enforce this so mistakes are caught early.
  static const char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy index is not initialized");
    return NULL;
  }

  const int32_t n = self->ptr->get_n_items();
  // if (n < 0) n = 0;
  return PyInt_FromLong(static_cast<long>(n));
}


// get_n_trees → number of built trees
//
// Python-facing semantics:
//
//   get_n_trees() -> int
//
static PyObject* py_an_get_n_trees(
  py_annoy*  self,
  PyObject*  args,
  PyObject*  kwargs) {
  // No arguments allowed; enforce this so mistakes are caught early.
  static const char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", (char**)kwlist)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy index is not initialized");
    return NULL;
  }

  const int32_t n = self->ptr->get_n_trees();
  // if (n < 0) n = 0;
  return PyInt_FromLong(static_cast<long>(n));
}

// ------------------------------------------------------------------
// Pickle / joblib support (deterministic)
// ------------------------------------------------------------------

static PyObject* py_an_getstate(
  py_annoy* self,
  PyObject* Py_UNUSED(ignored)) {
  PyObject* state = PyDict_New();
  if (!state) return NULL;

  // versioned state for forward compatibility.
  // Keep this stable: readers should ignore unknown keys.
  PyObject* v = PyLong_FromLong(1);
  if (!v || PyDict_SetItemString(state, "_pickle_version", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  v = PyLong_FromLong((long)self->f);
  if (!v || PyDict_SetItemString(state, "f", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // Metric: store canonical string (empty string means "unknown / lazy")
  const char* metric_c = metric_to_cstr(self->metric_id);
  v = PyUnicode_FromString(metric_c ? metric_c : "");
  if (!v || PyDict_SetItemString(state, "metric", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // Optional metric_id for faster restoration / future compatibility.
  v = PyLong_FromLong((long)self->metric_id);
  if (!v || PyDict_SetItemString(state, "metric_id", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // Optional on-disk path metadata
  if (!self->on_disk_path.empty()) {
    v = PyUnicode_FromString(self->on_disk_path.c_str());
  } else {
    Py_INCREF(Py_None);
    v = Py_None;
  }
  if (PyDict_SetItemString(state, "on_disk_path", v) < 0) {
    Py_DECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  v = PyBool_FromLong(self->has_pending_seed ? 1 : 0);
  if (!v || PyDict_SetItemString(state, "has_pending_seed", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  v = PyLong_FromUnsignedLongLong((unsigned long long)self->pending_seed);
  if (!v || PyDict_SetItemString(state, "pending_seed", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  v = PyBool_FromLong(self->has_pending_verbose ? 1 : 0);
  if (!v || PyDict_SetItemString(state, "has_pending_verbose", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  v = PyLong_FromLong((long)self->pending_verbose);
  if (!v || PyDict_SetItemString(state, "pending_verbose", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // Index payload snapshot (or None if lazy/uninitialized)
  if (!self->ptr) {
    Py_INCREF(Py_None);
    v = Py_None;
  } else {
    ScopedError error;
    vector<uint8_t> bytes = self->ptr->serialize(&error.err);
    if (bytes.empty() && error.err) {
      PyErr_SetString(PyExc_RuntimeError, error.err);
      Py_DECREF(state);
      return NULL;
    }
    v = PyBytes_FromStringAndSize(
        (const char*)bytes.data(), (Py_ssize_t)bytes.size());
    if (!v) {
      Py_DECREF(state);
      return NULL;
    }
  }
  if (PyDict_SetItemString(state, "data", v) < 0) {
    Py_DECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);
  return state;
}

static PyObject* py_an_setstate(
  py_annoy* self,
  PyObject* state) {
  if (!PyDict_Check(state)) {
    PyErr_SetString(PyExc_TypeError, "__setstate__ expects a dict");
    return NULL;
  }

  // Reset object state (state hygiene).
  if (self->ptr) { delete self->ptr; self->ptr = NULL; }
  self->f = 0;
  self->metric_id = METRIC_UNKNOWN;
  self->on_disk_path.clear();
  self->on_disk_active = false;

  self->pending_seed        = 0ULL;
  self->pending_verbose     = 0;
  self->has_pending_seed    = false;
  self->has_pending_verbose = false;

  // f
  PyObject* f_obj = PyDict_GetItemString(state, "f");  // borrowed
  if (f_obj && f_obj != Py_None) {
    long fv = PyLong_AsLong(f_obj);
    if (fv == -1 && PyErr_Occurred()) return NULL;
    if (fv < 0) {
      PyErr_SetString(PyExc_ValueError,
        "`f` in pickle state must be non-negative");
      return NULL;
    }
    self->f = (int)fv;
  }

  // metric_id (optional)
  PyObject* mid_obj = PyDict_GetItemString(state, "metric_id");  // borrowed
  if (mid_obj && mid_obj != Py_None) {
    long mid = PyLong_AsLong(mid_obj);
    if (mid == -1 && PyErr_Occurred()) return NULL;
    if (mid < (long)METRIC_UNKNOWN || mid > (long)METRIC_HAMMING) {
      PyErr_SetString(PyExc_ValueError, "`metric_id` in pickle state is invalid");
      return NULL;
    }
    self->metric_id = (MetricId)mid;
  }

  // metric (string; empty string means "unknown / lazy")
  PyObject* metric_obj = PyDict_GetItemString(state, "metric");  // borrowed
  if (self->metric_id == METRIC_UNKNOWN && metric_obj && metric_obj != Py_None) {
    if (!PyUnicode_Check(metric_obj)) {
      PyErr_SetString(PyExc_TypeError,
        "`metric` in pickle state must be str or None");
      return NULL;
    }
    const char* m = PyUnicode_AsUTF8(metric_obj);
    if (!m) return NULL;
    if (m[0] != '\0') {
      MetricId id = metric_from_string(m);
      if (id == METRIC_UNKNOWN) {
        PyErr_Format(PyExc_ValueError,
          "Unknown metric '%s' in pickle state", m);
        return NULL;
      }
      self->metric_id = id;
    }
  }

  // on_disk_path (optional metadata)
  PyObject* path_obj = PyDict_GetItemString(state, "on_disk_path");  // borrowed
  if (path_obj && path_obj != Py_None) {
    if (!PyUnicode_Check(path_obj)) {
      PyErr_SetString(PyExc_TypeError,
        "`on_disk_path` must be str or None");
      return NULL;
    }
    const char* p = PyUnicode_AsUTF8(path_obj);
    if (!p) return NULL;
    self->on_disk_path = p;
  }

  // pending seed/verbose (optional)
  PyObject* has_seed_obj = PyDict_GetItemString(state, "has_pending_seed");  // borrowed
  if (has_seed_obj && has_seed_obj != Py_None) {
    int truth = PyObject_IsTrue(has_seed_obj);
    if (truth < 0) return NULL;
    self->has_pending_seed = (truth != 0);
  }

  PyObject* seed_obj = PyDict_GetItemString(state, "pending_seed");  // borrowed
  if (seed_obj && seed_obj != Py_None) {
    unsigned long long sv = PyLong_AsUnsignedLongLong(seed_obj);
    if (sv == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    self->pending_seed = (uint64_t)sv;
  }

  PyObject* has_verbose_obj = PyDict_GetItemString(state, "has_pending_verbose");  // borrowed
  if (has_verbose_obj && has_verbose_obj != Py_None) {
    int truth = PyObject_IsTrue(has_verbose_obj);
    if (truth < 0) return NULL;
    self->has_pending_verbose = (truth != 0);
  }

  PyObject* verbose_obj = PyDict_GetItemString(state, "pending_verbose");  // borrowed
  if (verbose_obj && verbose_obj != Py_None) {
    long vv = PyLong_AsLong(verbose_obj);
    if (vv == -1 && PyErr_Occurred()) return NULL;
    self->pending_verbose = (int)vv;
  }

  // data (optional)
  PyObject* data = PyDict_GetItemString(state, "data");  // borrowed
  if (data && data != Py_None) {
    if (!PyBytes_Check(data)) {
      PyErr_SetString(PyExc_TypeError,
        "`data` in pickle state must be bytes or None");
      return NULL;
    }
    if (self->f <= 0) {
      PyErr_SetString(PyExc_ValueError,
        "Pickle state has `data` but missing/invalid `f`");
      return NULL;
    }
    if (self->metric_id == METRIC_UNKNOWN) {
      PyErr_SetString(PyExc_ValueError,
        "Pickle state has `data` but missing/empty `metric`");
      return NULL;
    }

    // Create index and restore snapshot
    if (!ensure_index(self))
      return NULL;

    char* buf = NULL;
    Py_ssize_t n = 0;
    if (PyBytes_AsStringAndSize(data, &buf, &n) < 0) {
      delete self->ptr; self->ptr = NULL;
      return NULL;
    }
    vector<uint8_t> v((uint8_t*)buf, (uint8_t*)buf + (size_t)n);

    ScopedError derr;
    if (!self->ptr->deserialize(&v, false, &derr.err)) {
      const std::string deser_msg = derr.err ? derr.err : "deserialize failed";
      // Compatibility: if snapshot deserialization fails, but a backing .annoy
      // file exists, try to restore from disk (keeps old/broken pickles usable).
      if (!self->on_disk_path.empty()) {
        const char* path = self->on_disk_path.c_str();

        // Deterministic: raise FileNotFoundError if the path is missing.
        if (!file_exists(path)) {
          delete self->ptr; self->ptr = NULL;
          return NULL;
        }

        // Reset index before attempting disk load (avoid partially-initialized state).
        delete self->ptr; self->ptr = NULL;

        if (!ensure_index(self))  // recreate (applies pending config)
          return NULL;

        ScopedError lerr;
        if (!self->ptr->load(path, false, &lerr.err)) {
          PyErr_Format(PyExc_IOError,
                       "deserialize failed (%s); fallback load('%s') also failed (%s)",
                       deser_msg.c_str(),
                       path,
                       lerr.err ? lerr.err : "load failed");
          delete self->ptr; self->ptr = NULL;
          return NULL;
        }
        self->on_disk_active = true;
      } else {
        PyErr_SetString(PyExc_IOError, deser_msg.c_str());
        delete self->ptr; self->ptr = NULL;
        return NULL;
      }
    }
  }
  Py_RETURN_NONE;
}

static PyObject* py_an_reduce_ex(
  py_annoy* self,
  PyObject* args) {
  int protocol = 0;
  if (!PyArg_ParseTuple(args, "i", &protocol)) return NULL;
  (void)protocol;

  PyObject* state = py_an_getstate(self, NULL);
  if (!state) return NULL;

  PyObject* cls = (PyObject*)Py_TYPE(self);
  Py_INCREF(cls);
  PyObject* empty_args = PyTuple_New(0);
  if (!empty_args) { Py_DECREF(cls); Py_DECREF(state); return NULL; }

  PyObject* out = PyTuple_New(3);
  if (!out) { Py_DECREF(cls); Py_DECREF(empty_args); Py_DECREF(state); return NULL; }
  PyTuple_SET_ITEM(out, 0, cls);
  PyTuple_SET_ITEM(out, 1, empty_args);
  PyTuple_SET_ITEM(out, 2, state);
  return out;
}

static PyObject* py_an_reduce(
  py_annoy* self,
  PyObject* Py_UNUSED(ignored)) {
  PyObject* state = py_an_getstate(self, NULL);
  if (!state) return NULL;

  PyObject* cls = (PyObject*)Py_TYPE(self);
  Py_INCREF(cls);
  PyObject* empty_args = PyTuple_New(0);
  if (!empty_args) { Py_DECREF(cls); Py_DECREF(state); return NULL; }

  PyObject* out = PyTuple_New(3);
  if (!out) { Py_DECREF(cls); Py_DECREF(empty_args); Py_DECREF(state); return NULL; }
  PyTuple_SET_ITEM(out, 0, cls);
  PyTuple_SET_ITEM(out, 1, empty_args);
  PyTuple_SET_ITEM(out, 2, state);
  return out;
}


// TODO: Enable method chaining by self : Annoy (Annoy(...).add_item(...).add_item(...).build(...))
// METH_NOARGS | METH_VARARGS | METH_KEYWORDS
static PyMethodDef py_annoy_methods[] = {
  // ------------------------------------------------------------------
  // Core index construction / mutation
  // ------------------------------------------------------------------

  {
    "add_item",
    (PyCFunction)py_an_add_item,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "add_item(i, vector)\n"
    "\n"
    "Add a single embedding vector to the index.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "i : int\n"
    "    Item id (index) must be non-negative.\n"
    "    Ids may be non-contiguous; the index allocates up to ``max(i) + 1``.\n"
    "vector : sequence of float\n"
    "    1D embedding of length ``f``. Values are converted to ``float``.\n"
    "    If ``f == 0`` and this is the first item, ``f`` is inferred from\n"
    "    ``vector`` and then fixed for the lifetime of this index.\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "Items must be added *before* calling :meth:`build`. After building\n"
    "the forest, further calls to :meth:`add_item` are not supported.\n"
    "\n"
    "Examples\n"
    "--------\n"
    ">>> index.add_item(0, [1.0, 0.0, 0.0])\n"
    ">>> index.add_item(1, [0.0, 1.0, 0.0])\n"
    ">>> idx.add_item(0, [1.0, 0.0, 0.0]).add_item(1, [0.0, 1.0, 0.0])\n"
  },

  {
    "build",
    (PyCFunction)py_an_build,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "build(n_trees, n_jobs=-1)\n"
    "\n"
    "Build a forest of random projection trees.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "n_trees : int\n"
    "    Number of trees in the forest. Larger values typically improve recall\n"
    "    at the cost of slower build time and higher memory usage.\n"
    "n_jobs : int, optional, default=-1\n"
    "    Number of threads to use while building. ``-1`` means \"auto\" (use\n"
    "    the implementation's default, typically all available CPU cores).\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "After :meth:`build` completes, the index becomes read-only for queries.\n"
    "To add more items, call :meth:`unbuild`, add items, and then rebuild.\n"
    "\n"
    "References\n"
    "----------\n"
    "Erik Bernhardsson, \"Annoy: Approximate Nearest Neighbours in C++/Python\".\n"
  },

  {
    "unbuild",
    (PyCFunction)py_an_unbuild,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "unbuild()\n"
    "\n"
    "Discard the current forest, allowing new items to be added.\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "After calling :meth:`unbuild`, you must call :meth:`build`\n"
    "again before running nearest-neighbour queries.\n"
  },

  // ------------------------------------------------------------------
  // Persistence: disk + byte + memory usage
  // ------------------------------------------------------------------

  {
    "save",
    (PyCFunction)py_an_save,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "save(fn, prefault=False)\n"
    "\n"
    "Persist the index to a binary file on disk.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "fn : str\n"
    "    Path to the output file. Existing files will be overwritten.\n"
    "prefault : bool, optional, default=False\n"
    "    If True, aggressively fault pages into memory during save.\n"
    "    Primarily useful on some platforms for very large indexes.\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Raises\n"
    "------\n"
    "IOError\n"
    "    If the file cannot be written.\n"
    "RuntimeError\n"
    "    If the index is not initialized or save fails.\n"
  },

  {
    "load",
    (PyCFunction)py_an_load,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "load(fn, prefault=False)\n"
    "\n"
    "Load (mmap) an index from disk into the current object.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "fn : str\n"
    "    Path to a file previously created by :meth:`save` or\n"
    "    :meth:`on_disk_build`.\n"
    "prefault : bool, optional, default=False\n"
    "    If True, fault pages into memory when the file is mapped.\n"
    "\n"
    "Raises\n"
    "------\n"
    "IOError\n"
    "    If the file cannot be opened or mapped.\n"
    "RuntimeError\n"
    "    If the index is not initialized or the file is incompatible.\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "The in-memory index must have been constructed with the same dimension\n"
    "and metric as the on-disk file.\n"
  },

  {
    "unload",
    (PyCFunction)py_an_unload,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "unload()\n"
    "\n"
    "Unmap any memory-mapped file backing this index.\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This releases OS-level resources associated with the mmap,\n"
    "but keeps the Python object alive.\n"
  },

  {
    "on_disk_build",
    (PyCFunction)py_an_on_disk_build,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "on_disk_build(fn)\n"
    "\n"
    "Configure the index to build using an on-disk backing file.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "fn : str\n"
    "    Path to a file that will hold the index during build.\n"
    "    The file is created or overwritten as needed.\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This mode is useful for very large datasets that do not fit\n"
    "comfortably in RAM during construction.\n"
  },

  {
    "serialize",
    (PyCFunction)py_an_serialize,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "serialize() -> bytes\n"
    "\n"
    "Serialize the built in-memory index into a byte string.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "data : bytes\n"
    "    Opaque binary blob containing the Annoy index.\n"
    "\n"
    "Raises\n"
    "------\n"
    "RuntimeError\n"
    "    If the index is not initialized or serialization fails.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "The serialized form is a snapshot of the internal C++ data structures.\n"
    "It can be stored, transmitted, or used with joblib without rebuilding trees.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "deserialize : Restore an index from a serialized byte string.\n"
  },

  {
    "deserialize",
    (PyCFunction)py_an_deserialize,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "deserialize(byte, prefault=False)\n"
    "\n"
    "Restore the index from a serialized byte string.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "byte : bytes\n"
    "    Byte string produced by :meth:`serialize`.\n"
    "prefault : bool, optional, default=False\n"
    "    If True, fault pages into memory while restoring.\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Raises\n"
    "------\n"
    "IOError\n"
    "    If deserialization fails due to invalid or incompatible data.\n"
    "RuntimeError\n"
    "    If the index is not initialized.\n"
  },

  {
    "memory_usage",
    (PyCFunction)py_an_memory_usage,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "memory_usage() -> int\n"
    "\n"
    "Approximate memory usage of the index in bytess.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "n_bytess : int or None\n"
    "    Approximate number of bytes used by the index. Returns ``None`` if the\n"
    "    index is not initialized.\n"
    "\n"
    "Raises\n"
    "------\n"
    "RuntimeError\n"
    "    If memory usage cannot be computed.\n"
  },

  // ------------------------------------------------------------------
  // Query API
  // ------------------------------------------------------------------

  {
    "get_nns_by_item",
    (PyCFunction)py_an_get_nns_by_item,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "get_nns_by_item(i, n, search_k=-1, include_distances=False)\n"
    "\n"
    "Return the `n` nearest neighbours for a stored item id.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "i : int\n"
    "    Item id (index) previously passed to :meth:`add_item(i, embedding)`.\n"
    "n : int\n"
    "    Number of nearest neighbours to return.\n"
    "search_k : int, optional, default=-1\n"
    "    Maximum number of nodes to inspect. Larger values usually improve recall\n"
    "    at the cost of slower queries. If ``-1``, defaults to approximately\n"
    "    ``n_trees * n``.\n"
    "include_distances : bool, optional, default=False\n"
    "    If True, return a ``(indexs, distances)`` tuple. Otherwise return only\n"
    "    the list of indexs.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "indexs : list[int] | tuple[list[int], list[float]]\n"
    "    If ``include_distances=False``: list of neighbour item ids.\n"
    "    If ``include_distances=True``: ``(indexs, distances)``.\n"
    "\n"
    "Raises\n"
    "------\n"
    "RuntimeError\n"
    "    If the index is not initialized or has not been built.\n"
    "IndexError\n"
    "    If ``i`` is out of range.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "get_nns_by_vector : Query with an explicit query embedding.\n"
  },

  {
    "get_nns_by_vector",
    (PyCFunction)py_an_get_nns_by_vector,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "get_nns_by_vector(vector, n, search_k=-1, include_distances=False)\n"
    "\n"
    "Return the `n` nearest neighbours for a query embedding.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "vector : sequence of float\n"
    "    Query embedding of length ``f``.\n"
    "n : int\n"
    "    Number of nearest neighbours to return.\n"
    "search_k : int, optional, default=-1\n"
    "    Maximum number of nodes to inspect. Larger values typically improve recall\n"
    "    at the cost of slower queries. If ``-1``, defaults to approximately\n"
    "    ``n_trees * n``.\n"
    "include_distances : bool, optional, default=False\n"
    "    If True, return a ``(indexs, distances)`` tuple. Otherwise return only\n"
    "    the list of indexs.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "indexs : list[int] | tuple[list[int], list[float]]\n"
    "    If ``include_distances=False``: list of neighbour item ids.\n"
    "    If ``include_distances=True``: ``(indexs, distances)``.\n"
    "\n"
    "Raises\n"
    "------\n"
    "RuntimeError\n"
    "    If the index is not initialized or has not been built.\n"
    "ValueError\n"
    "    If ``len(vector) != f``.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "get_nns_by_item : Query by stored item id.\n"
  },

  {
    "get_item_vector",
    (PyCFunction)py_an_get_item_vector,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "get_item_vector(i) -> list[float]\n"
    "\n"
    "Return the stored embedding vector for a given item id.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "i : int\n"
    "    Item id (index) previously passed to :meth:`add_item`.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "vector : list[float]\n"
    "    Stored embedding of length ``f``.\n"
    "\n"
    "Raises\n"
    "------\n"
    "RuntimeError\n"
    "    If the index is not initialized.\n"
    "IndexError\n"
    "    If ``i`` is out of range.\n"
  },

  {
    "get_distance",
    (PyCFunction)py_an_get_distance,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "get_distance(i, j) -> float\n"
    "\n"
    "Return the distance between two stored items.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "i, j : int\n"
    "    Item ids (index) of two stored samples.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "d : float\n"
    "    Distance between items ``i`` and ``j`` under the current metric.\n"
    "\n"
    "Raises\n"
    "------\n"
    "RuntimeError\n"
    "    If the index is not initialized.\n"
    "IndexError\n"
    "    If either index is out of range.\n"
  },

  {
    "get_n_items",
    (PyCFunction)py_an_get_n_items,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "get_n_items() -> int\n"
    "\n"
    "Return the number of stored items in the index.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "n_items : int\n"
    "    Number of items that have been added and are currently addressable.\n"
    "\n"
    "Raises\n"
    "------\n"
    "RuntimeError\n"
    "    If the index is not initialized.\n"
  },

  {
    "get_n_trees",
    (PyCFunction)py_an_get_n_trees,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "get_n_trees() -> int\n"
    "\n"
    "Return the number of trees in the current forest.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "n_trees : int\n"
    "    Number of trees that have been built.\n"
    "\n"
    "Raises\n"
    "------\n"
    "RuntimeError\n"
    "    If the index is not initialized.\n"
  },

  // ------------------------------------------------------------------
  // RNG / logging controls
  // ------------------------------------------------------------------

  {
    "set_seed",
    (PyCFunction)py_an_set_seed,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "set_seed(seed=0)\n"
    "\n"
    "Set the random seed used for tree construction.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "seed : int, optional\n"
    "    Non-negative integer seed. If called before the index is constructed,\n"
    "    the seed is stored and applied when the C++ index is created.\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "Annoy is deterministic by default. Setting an explicit seed is useful for\n"
    "reproducible experiments and debugging.\n"
  },

  {
    "verbose",
    (PyCFunction)py_an_verbose,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "verbose(level=1)\n"
    "\n"
    "Control verbosity of the underlying C++ index.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "level : int, optional, default=1\n"
    "    Verbosity level. Values are clamped to the range ``[-2, 2]``.\n"
    "    ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.\n"
    "    Logging level inspired by gradient-boosting libraries:\n"
    "\n"
    "    * ``<= 0`` : quiet (warnings only)\n"
    "    * ``1``    : info (Annoy's ``verbose=True``)\n"
    "    * ``>= 2`` : debug (currently same as info, reserved for future use)\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
  },

  // ------------------------------------------------------------------
  // Introspection
  // ------------------------------------------------------------------

  {
    "info",
    (PyCFunction)py_an_info,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "info() -> dict\n"
    "\n"
    "Return a structured summary of the index.\n"
    "\n"
    "This method returns a JSON-like Python dictionary that is easier to\n"
    "inspect programmatically than the legacy multi-line string format.\n"
    "\n"
    "Keys\n"
    "----\n"
    "dimension : int\n"
    "    Dimensionality of the index.\n"
    "metric : str\n"
    "    Distance metric name.\n"
    "n_items : int\n"
    "    Number of items currently stored.\n"
    "n_trees : int\n"
    "    Number of trees built.\n"
    "memory_usage_byte : int\n"
    "    Approximate memory usage in bytes.\n"
    "memory_usage_mib : float\n"
    "    Approximate memory usage in MiB.\n"
    "on_disk_path : str | None\n"
    "    Path used for on-disk build, if configured.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "info : dict or None\n"
    "    Dictionary describing the current index state.\n"
    "\n"
    "Examples\n"
    "--------\n"
    ">>> info = idx.info()\n"
    ">>> info['dimension']\n"
    "100\n"
    ">>> info['n_items']\n"
    "1000\n"
  },

  // Pickle / joblib
  {
    "__getstate__",
    (PyCFunction)py_an_getstate,
    METH_NOARGS,
    (char*)
    "__getstate__() -> dict\n"
    "\n"
    "Return a versioned state dictionary for pickle/joblib.\n"
    "\n"
    "This method is primarily used by :mod:`pickle` (and joblib) and is not\n"
    "intended to be called directly in normal workflows.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "state : dict\n"
    "    A JSON-like dictionary with the following keys:\n"
    "\n"
    "    * ``_pickle_version`` : int\n"
    "    * ``f`` : int | None\n"
    "    * ``metric`` : str | None\n"
    "    * ``on_disk_path`` : str | None\n"
    "    * ``has_pending_seed`` : bool\n"
    "    * ``pending_seed`` : int\n"
    "    * ``has_pending_verbose`` : bool\n"
    "    * ``pending_verbose`` : int\n"
    "    * ``data`` : bytes | None\n"
    "\n"
    "Notes\n"
    "-----\n"
    "If the underlying C++ index is initialized, ``data`` contains a serialized\n"
    "snapshot (see :meth:`serialize`). Otherwise, ``data`` is ``None``.\n"
  },
  {
    "__setstate__",
    (PyCFunction)py_an_setstate,
    METH_O,
    (char*)
    "__setstate__(state) -> None\n"
    "\n"
    "Restore object state from a dictionary produced by :meth:`__getstate__`.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "state : dict\n"
    "    State dictionary returned by :meth:`__getstate__`.\n"
    "\n"
    "Raises\n"
    "------\n"
    "TypeError\n"
    "    If ``state`` is not a dictionary.\n"
    "ValueError\n"
    "    If required fields are missing or invalid (e.g., negative ``f``).\n"
    "FileNotFoundError\n"
    "    If disk fallback is required and ``on_disk_path`` does not exist.\n"
    "IOError\n"
    "    If restoring from the serialized snapshot or disk file fails.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "Restoration first attempts to deserialize the in-memory snapshot from\n"
    "``state[\"data\"]``. If that fails and ``on_disk_path`` is present,\n"
    "the index is loaded from disk as a compatibility fallback.\n"
  },
  {
    "__reduce_ex__",
    (PyCFunction)py_an_reduce_ex,
    METH_VARARGS,
    (char*)
    "__reduce_ex__(protocol)\n"
    "\n"
    "Pickle protocol support.\n"
    "\n"
    "This returns the standard 3-tuple ``(cls, args, state)`` used by pickle.\n"
    "Users typically do not need to call this directly.\n"
  },
  {
    "__reduce__",
    (PyCFunction)py_an_reduce,
    METH_NOARGS,
    (char*)
    "__reduce__()\n"
    "\n"
    "Pickle support.\n"
    "\n"
    "Equivalent to :meth:`__reduce_ex__` with the default protocol.\n"
  },

  {NULL, NULL, 0, NULL}  // Sentinel
};

// ======================= Annoy dunder helpers ============================

// __len__(self) → number of items in the index
static Py_ssize_t py_an_len(PyObject* obj) {
  py_annoy* self = reinterpret_cast<py_annoy*>(obj);
  if (!self->ptr)
    return 0;

  int32_t n = self->ptr->get_n_items();
  if (n < 0)
    n = 0;  // defensive
  return static_cast<Py_ssize_t>(n);
}

// Sequence protocol: we only implement sq_length so that len(Annoy) works.
// All other operations are intentionally left NULL.
static PySequenceMethods py_annoy_as_sequence = {
  py_an_len,   /* sq_length            */
  0,           /* sq_concat            */
  0,           /* sq_repeat            */
  0,           /* sq_item              */
  0,           /* was sq_slice         */
  0,           /* sq_ass_item          */
  0,           /* was sq_ass_slice     */
  0,           /* sq_contains          */
  0,           /* sq_inplace_concat    */
  0            /* sq_inplace_repeat    */
};

// __repr__(self) → "Annoy(f=128, metric='angular', n_items=1000, n_trees=10, on_disk_path=/path)"
static PyObject* py_an_repr(
  PyObject* obj) {
  py_annoy* self = reinterpret_cast<py_annoy*>(obj);

  // Dimension (may still be 0 if we are in lazy mode)
  int f = self->f;

  // Metric (single source of truth: metric_id)
  const char* metric = metric_to_cstr(self->metric_id);
  if (!metric) metric = "unknown";

  long n_items = 0;
  long n_trees = 0;
  if (self->ptr) {
    n_items = static_cast<long>(self->ptr->get_n_items());
    n_trees = static_cast<long>(self->ptr->get_n_trees());
  }

  // Represent on_disk_path using Python's repr for correct quoting/escaping.
  PyObject* path_obj = NULL;
  if (self->on_disk_path.empty()) {
    path_obj = Py_None;
    Py_INCREF(Py_None);
  } else {
    path_obj = PyUnicode_FromString(self->on_disk_path.c_str());
    if (!path_obj) return NULL;
  }

  PyObject* path_repr = PyObject_Repr(path_obj);
  Py_DECREF(path_obj);
  if (!path_repr) return NULL;

  const char* path_c = PyUnicode_AsUTF8(path_repr);
  if (!path_c) {
    Py_DECREF(path_repr);
    return NULL;
  }

  PyObject* out = PyUnicode_FromFormat(
    "Annoy(f=%d, metric='%s', n_items=%ld, n_trees=%ld, on_disk_path=%s)",
    f,
    metric,
    n_items,
    n_trees,
    path_c);
  Py_DECREF(path_repr);
  return out;
}

// ======================= Module / types ===================================

static PyTypeObject py_annoy_type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  // "annoy.Annoy",                            /* tp_name matches the actual exported module/object prints like <class 'annoy.Annoy'> */
  "scikitplot.cexternals._annoy.Annoy",     /* tp_name */
  sizeof(py_annoy),                         /* tp_basicsize */
  0,                                        /* tp_itemsize */
  (destructor)py_an_dealloc,                /* tp_dealloc */
  0,                                        /* tp_vectorcall_offset (was tp_print) */
  0,                                        /* tp_getattr */
  0,                                        /* tp_setattr */
  0,                                        /* tp_as_async (was tp_compare) */
  (reprfunc)py_an_repr,                     /* tp_repr */
  0,                                        /* tp_as_number */
  &py_annoy_as_sequence,                    /* tp_as_sequence → supports len() */
  0,                                        /* tp_as_mapping */
  0,                                        /* tp_hash  */
  0,                                        /* tp_call */
  0,                                        /* tp_str */
  0,                                        /* tp_getattro */
  0,                                        /* tp_setattro */
  0,                                        /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  (char*)kAnnoyTypeDoc,                     /* tp_doc  */
  0,                                        /* tp_traverse */
  0,                                        /* tp_clear */
  0,                                        /* tp_richcompare */
  0,                                        /* tp_weaklistoffset */
  0,                                        /* tp_iter */
  0,                                        /* tp_iternext */
  py_annoy_methods,                         /* tp_methods */
  py_annoy_members,                         /* tp_members */
  py_annoy_getset,                          /* tp_getset */
  0,                                        /* tp_base */
  0,                                        /* tp_dict */
  0,                                        /* tp_descr_get */
  0,                                        /* tp_descr_set */
  0,                                        /* tp_dictoffset */
  (initproc)py_an_init,                     /* tp_init */
  PyType_GenericAlloc,                      /* tp_alloc */
  py_an_new,                                /* tp_new  */
};

static PyMethodDef module_methods[] = {
  {NULL, NULL, 0, NULL}	/* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef annoylibmodule = {
  PyModuleDef_HEAD_INIT,
  "annoylib",          /* m_name: import annoylib */
  ANNOY_MOD_DOC,      /* m_doc: module-level docstring */
  -1,                  /* m_size */
  module_methods,      /* m_methods */
  NULL,                /* m_slots */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL                 /* m_free */
};
#endif

// ---------------------------------------------------------------------
// 2. Helpers: add constants / metadata
// ---------------------------------------------------------------------

// Define these for both Py2 and Py3 so create_module can always call them.
#if PY_MAJOR_VERSION >= 3
static void AddVersionConstants(PyObject* m) {
  // If you have version macros in annoylib.h, expose them here.
  // Example:
  // PyModule_AddIntConstant(m, "ANNOY_VERSION_MAJOR", ANNOY_VERSION_MAJOR);
  // PyModule_AddIntConstant(m, "ANNOY_VERSION_MINOR", ANNOY_VERSION_MINOR);

  // At least publish something stable:
  PyModule_AddStringConstant(m, "__backend__", "cpp");
}
#endif

// Standard NumPy C-API initialization helper.
// #if PY_MAJOR_VERSION >= 3
// static bool InitNumpyIfNeeded() {
//   if (PyArray_API == NULL) {
//     import_array();  // sets an error on failure
//     if (PyErr_Occurred()) {
//       return false;
//     }
//   }
//   return true;
// }
// #endif

// ---------------------------------------------------------------------
// 3. Internal factory: create the module and register all types
// ---------------------------------------------------------------------

static PyObject* create_module(void) {
  // Prepare Annoy type
  if (PyType_Ready(&py_annoy_type) < 0)
    return NULL;

  PyObject* m;

#if PY_MAJOR_VERSION >= 3
  // Initialize NumPy C API (safe to call in both Py2 and Py3)
  // if (!InitNumpyIfNeeded()) return NULL;
  m = PyModule_Create(&annoylibmodule);
#else
  m = Py_InitModule("annoylib", module_methods);
#endif

  if (!m)
    return NULL;

  // Expose `Annoy` class
  Py_INCREF(&py_annoy_type);
  if (PyModule_AddObject(m, "Annoy", (PyObject*)&py_annoy_type) < 0) {
    Py_DECREF(&py_annoy_type);
    Py_DECREF(m);
    return NULL;
  }

#if PY_MAJOR_VERSION >= 3
  // Version/constants
  AddVersionConstants(m);
#endif
  return m;
}

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_annoylib(void) {
  return create_module();
}
#else
PyMODINIT_FUNC initannoylib(void) {
  // (void)create_module();
  PyObject* m = create_module();
  if (!m)
    return;
  // In Python 2, the module object is returned implicitly.
}
#endif

// vim: tabstop=2 shiftwidth=2
