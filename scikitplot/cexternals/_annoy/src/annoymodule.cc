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

// Prefer public headers over internal includes. Python.h already pulls in the
// correct definitions for bytes / unicode helpers on every supported version.
// #include "bytesobject.h"

// TODO: ?Some fields deprecated in Python 3.11+
#include "structmember.h"  // PyMemberDef, T_INT, READONLY

// System / STL ----------------------------------------------------------------
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <algorithm>  // std::transform
#include <cctype>  // std::tolower
#include <cmath>
#include <cstdint>
#include <cstring>  // std::strcmp
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>  // std::ifstream
#include <memory>
#include <new>  // std::bad_alloc
#include <iostream>  // std::cout << R"(...)" << std::endl;
#include <string>  // std::string
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if PY_MAJOR_VERSION >= 3
  #define IS_PY3K
#endif

#ifdef IS_PY3K
  #define PyInt_FromLong PyLong_FromLong
#endif

#if defined(_MSC_VER) && _MSC_VER == 1500
  typedef signed __int32 int32_t;
#else
  #include <stdint.h>
#endif

#ifndef Py_TYPE
  #define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
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
    "C++-powered :class:`~.Annoy` type. For day-to-day work, prefer a\n"
    "higher-level Python wrapper (if your project provides one)::\n"
    "\n"
    "    >>> from annoy import Annoy, AnnoyIndex\n"
    "\n"
    "If your project ships a higher-level wrapper, it may re-export the same\n"
    "type under a different module path (for example, :mod:`~scikitplot.annoy`)::\n"
    "\n"
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

// Annoy's Kiss64Random uses a deterministic default seed.
// This wrapper also exposes set_seed() for reproducible indices.
//
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

// A raw string literal R"( ... )";
static const char kAnnoyTypeDoc[] =
COMPILER_INFO ". " AVX_INFO "."
"\n"
R"ANN(
Approximate Nearest Neighbors index (Annoy) with a small, lazy C-extension wrapper.

::

>>> Annoy(
>>>     f=None,
>>>     metric=None,
>>>     *,
>>>     on_disk_path=None,
>>>     prefault=None,
>>>     schema_version=None,
>>>     seed=None,
>>>     verbose=None,
>>> )

Parameters
----------
f : int or None, optional, default=None
    Vector dimension. If ``0`` or ``None``, dimension may be inferred from the
    first vector passed to ``add_item`` (lazy mode).
    If None, treated as ``0`` (reset to default).
metric : {"angular", "cosine", \
        "euclidean", "l2", "lstsq", \
        "manhattan", "l1", "cityblock", "taxicab", \
        "dot", "@", ".", "dotproduct", "inner", "innerproduct", \
        "hamming"} or None, optional, default=None
    Distance metric (one of 'angular', 'euclidean', 'manhattan', 'dot', 'hamming').
    If omitted and ``f > 0``, defaults to ``'angular'`` (cosine-like).
    If omitted and ``f == 0``, metric may be set later before construction.
    If None, treated as ``'angular'`` (reset to default).
on_disk_path : str or None, optional, default=None
    Path for on-disk build/load. None if not configured.
prefault : bool or None, optional, default=None
    If True, request page-faulting index pages into memory when loading
    (when supported by the underlying platform/backing).
    If None, treated as ``False`` (reset to default).
schema_version : int, optional, default=None
    Reserved for future schema/version tracking. Currently stored on the
    object and reported by :meth:`~.Annoy.info`, but does not change the
    on-disk format.
    If None, treated as ``0`` (reset to default).
seed : int or None, optional, default=None
    Non-negative integer seed. If set before the index is constructed,
    the seed is stored and applied when the C++ index is created.
verbose : int or None, optional, default=None
    Verbosity level. Values are clamped to the range ``[-2, 2]``.
    ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.
    Logging level inspired by gradient-boosting libraries:

    * ``<= 0`` : quiet (warnings only)
    * ``1``    : info (Annoy's ``verbose=True``)
    * ``>= 2`` : debug (currently same as info, reserved for future use)

Attributes
----------
f : int
    Vector dimension. ``0`` means "unknown / lazy".
metric : {'angular', 'euclidean', 'manhattan', 'dot', 'hamming'} or None
    Canonical metric name, or None if not configured yet (lazy).
on_disk_path : str or None
    Path for on-disk build/load. None if not configured.
prefault : bool, default=False
    Stored prefault flag (see :meth:`load`/`:meth:`save` prefault parameters).
schema_version : int, default=0
    Reserved schema/version marker (stored; does not affect on-disk format).

Notes
-----
* Once the underlying C++ index is created, ``f`` and ``metric`` are immutable.
  This keeps the object consistent and avoids undefined behavior.
* The C++ index is created lazily when sufficient information is available:
  when both ``f > 0`` and ``metric`` are known, or when an operation that
  requires the index is first executed.
* If ``f == 0``, the dimensionality is inferred from the first non-empty vector
  passed to :meth:`add_item` and is then fixed for the lifetime of the index.
* If ``metric`` is omitted while ``f > 0``, the current behavior defaults to
  ``'angular'`` and may emit a :class:`FutureWarning`. To avoid warnings and
  future behavior changes, always pass ``metric=...`` explicitly.
* Items must be added *before* calling :meth:`build`. After :meth:`build`, the
  index becomes read-only; to add more items, call :meth:`unbuild`, add items
  again with :meth:`add_item`, then call :meth:`build` again.
* Very large indexes can be built directly on disk with :meth:`on_disk_build`
  and then memory-mapped with :meth:`load`.
* :meth:`info` returns a structured summary (dimension, metric, counts, and
  optional memory usage) suitable for programmatic inspection.
* This wrapper stores user configuration (e.g., seed/verbosity) even before the
  C++ index exists and applies it deterministically upon construction.

Developer Notes:

- Source of truth:
* ``f`` (int) and ``metric_id`` (enum) describe configuration.
* ``ptr`` is NULL when index is not constructed.
- Invariant:
* ``ptr != NULL`` implies ``f > 0`` and ``metric_id != METRIC_UNKNOWN``.

See Also
--------
add_item : Add a vector to the index.
build : Build the forest after adding items.
unbuild : Remove trees to allow adding more items.
get_nns_by_item, get_nns_by_vector : Query nearest neighbours.
save, load : Persist the index to/from disk.
serialize, deserialize : Persist the index to/from bytes.
set_seed : Set the random seed deterministically.
verbose : Set verbosity level.
info : Return a structured summary of the current index.

Examples
--------
>>> from annoy import Annoy, AnnoyIndex

High-level API:

>>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
>>> from scikitplot.annoy import Annoy, AnnoyIndex, Index

The lifecycle follows the examples in ``test.ipynb``:

1. **Construct the index**

>>> import random; random.seed(0)
>>> # from annoy import AnnoyIndex
>>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
>>> from scikitplot.annoy import Annoy, AnnoyIndex, Index

>>> idx = Annoy(f=3, metric="angular")
>>> idx.f, idx.metric
(3, 'angular')

If you pass ``f=0`` the dimension can be inferred on the first
call to :meth:`add_item`.

2. **Add items**

>>> idx.add_item(0, [1.0, 0.0, 0.0])
>>> idx.add_item(1, [0.0, 1.0, 0.0])
>>> idx.add_item(2, [0.0, 0.0, 1.0])
>>> idx.get_n_items()
3

3. **Build the forest**

>>> idx.build(n_trees=10)
>>> idx.get_n_trees()
10
>>> idx.memory_usage()  # byte
543076

After :meth:`build` the index becomes read-only.  You can still
query, save, load and serialize it.

4. **Query neighbours**

By stored item id:

>>> idx.get_nns_by_item(0, 5)
[0, 1, 2, ...]

With distances:

>>> idx.get_nns_by_item(0, 5, include_distances=True)
([0, 1, 2, ...], [0.0, 1.22, 1.26, ...])

Or by an explicit query vector:

>>> idx.get_nns_by_vector([0.1, 0.2, 0.3], 5, include_distances=True)
([103, 71, 160, 573, 672], [...])

5. **Persistence**

To work with memory-mapped indices on disk:

>>> idx.save("annoy_test.annoy")
>>> idx2 = Annoy(f=100, metric="angular")
>>> idx2.load("annoy_test.annoy")
>>> idx2.get_n_items()
1000

Or via raw byte:

>>> buf = idx.serialize()
>>> new_idx = Annoy(f=100, metric="angular")
>>> new_idx.deserialize(buf)
>>> new_idx.get_n_items()
1000

You can release OS resources with :meth:`unload` and drop the
current forest with :meth:`unbuild`.
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

static const char kPrefaultDoc[] =
R"PFDOC(
Default prefault flag stored on the object.

This setting is currently informational and may be used by future versions
as a default for on-disk operations. Today, methods like :meth:`load`,
:meth:`save`, and :meth:`deserialize` accept a per-call ``prefault`` argument.

Returns
-------
bool
    Current prefault flag.

Notes
-----
- This flag does not retroactively change already-loaded mappings.
)PFDOC";

static const char kSchemaVersionDoc[] =
R"SVDOC(
Schema/version marker stored on the object.

This value is reserved for future compatibility work and does not currently
change the Annoy on-disk format.

Returns
-------
int
    Current schema version marker.

Notes
-----
- ``0`` is the default sentinel value.
)SVDOC";

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

  // Optional per-instance attribute dictionary (enables attaching user
  // metadata without changing the C-extension's core schema).
  PyObject* dict;               // instance attribute dictionary (__dict__), NULL unless a user sets custom attributes

  // Optional weakref list (enables weakref.ref(obj)).
  // NOTE: This is separate from the instance __dict__.
  PyObject* weakreflist;        // weakref list head, or NULL

  AnnoyIndexInterface<int32_t, float>* ptr;  // underlying C++ index dynamic_cast<AnnoyAngular*>(ptr)

  int f;                        // 0 means "unknown / lazy" (dimension inferred from first add_item)
  MetricId metric_id;           // METRIC_UNKNOWN means "unknown / lazy"
  // std::string metric;        // empty char "" or NULL

  // --- Optional on-disk path (for on_disk_build / load) ---
  bool on_disk_active;          // true if ptr is currently backed by disk (load() or on_disk_build())
  std::string on_disk_path;     // empty char "" or NULL => none no active on-disk index
  bool prefault;                //
  int schema_version;           //

  // --- Pending runtime configuration (before C++ index exists) ---
  uint64_t pending_seed;        // last seed requested via set_seed()
  int      pending_verbose;     // last verbosity level requested
  bool     has_pending_seed;    // whether user explicitly set a seed
  bool     has_pending_verbose; // whether user explicitly set verbosity
} py_annoy;

// Forward declarations (used for consistent validation in tp_init)
static int py_annoy_set_on_disk_path(py_annoy* self, PyObject* value, void* closure);
static int py_annoy_set_prefault(py_annoy* self, PyObject* value, void* closure);
static int py_annoy_set_schema_version(py_annoy* self, PyObject* value, void* closure);

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

  // PyType_GenericAlloc may GC-track instances immediately when the type has
  // Py_TPFLAGS_HAVE_GC. Untrack during initialization and track once all
  // GC-visible fields (e.g. __dict__) are in a consistent state.
  if (type->tp_flags & Py_TPFLAGS_HAVE_GC) {
    // if (PyObject_GC_IsTracked((PyObject*)self)) { PyObject_GC_UnTrack((PyObject*)self); }
    // PyObject_GC_IsTracked() is only available starting with Python 3.9.
    // For Python 3.8 compatibility we untrack unconditionally.
    // (On CPython this is safe even if the object is not currently tracked.)
    PyObject_GC_UnTrack((PyObject*)self);  // kernel dead
  }

  self->dict = NULL;
  self->weakreflist = NULL;

  self->ptr = NULL;

  self->f = 0;
  self->metric_id = METRIC_UNKNOWN;

  self->on_disk_active = false;
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

  self->prefault = false;
  self->schema_version = 0;

  // Pending configuration: default to “nothing explicitly requested”
  self->pending_seed        = 0ULL;
  self->pending_verbose     = 0;
  self->has_pending_seed    = false;
  self->has_pending_verbose = false;

  // Track this object for cyclic GC only after it is fully initialized.
  //
  // CPython allocation strategies vary: on some builds, tp_alloc may return an
  // instance already tracked by the GC. Tracking twice triggers a fatal
  // assertion in debug builds ("object already tracked by the garbage collector").
  //
  // Rule: if (and only if) the type participates in GC, ensure we end up tracked
  // exactly once at the end of construction.
  if (type->tp_flags & Py_TPFLAGS_HAVE_GC) {
    // if (PyObject_GC_IsTracked((PyObject*)self)) { PyObject_GC_Track((PyObject*)self); }
    // See comment above: PyObject_GC_IsTracked() is not available on Python 3.8.
    // We untracked unconditionally right after allocation, so tracking here is
    // deterministic and happens exactly once.
    PyObject_GC_Track((PyObject*)self);
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
  // Allow re-initialization safely (rare but possible in CPython).
  // We reset the wrapper state deterministically.
  if (self->ptr) {
    delete self->ptr;
    self->ptr = NULL;
  }

  self->f = 0;
  self->metric_id = METRIC_UNKNOWN;

  self->on_disk_active = false;
  self->on_disk_path.clear();

  self->prefault = false;
  self->schema_version = 0;

  // Reset pending configuration as well (true re-init).
  self->pending_seed        = 0ULL;
  self->pending_verbose     = 0;
  self->has_pending_seed    = false;
  self->has_pending_verbose = false;

  // ------------------------------------------------------------------
  // Signature (public, documented):
  //   Annoy(f=None, metric=None, *, on_disk_path=None, prefault=False,
  //         schema_version=0, seed=None, verbose=None)
  //
  // Deterministic rules:
  //   * Only (f, metric) may be passed positionally.
  //   * All other parameters are keyword-only.
  // ------------------------------------------------------------------
  if (!PyTuple_Check(args)) {
    PyErr_SetString(PyExc_TypeError,
      "internal error: args is not a tuple");
    return -1;
  }

  const Py_ssize_t nargs = PyTuple_GET_SIZE(args);
  if (nargs > 2) {
    PyErr_Format(PyExc_TypeError,
      "Annoy() takes at most 2 positional arguments (%zd given)", nargs);
    return -1;
  }

  PyObject* f_obj      = (nargs >= 1) ? PyTuple_GET_ITEM(args, 0) : NULL;
  PyObject* metric_obj = (nargs >= 2) ? PyTuple_GET_ITEM(args, 1) : NULL;

  PyObject*   on_disk_path   = NULL;
  PyObject*   prefault       = NULL;
  PyObject*   schema_version = NULL;
  PyObject*   seed           = NULL;
  PyObject*   verbose        = NULL;

  // parse by PyArg_ParseTuple or PyArg_ParseTupleAndKeywords
  // "O|s" f (required, PyObject), metric (optional, const char*)
  // static const char* kwlist[] = {"f", "metric", NULL};
  // if (!PyArg_ParseTupleAndKeywords(
  //   args, kwargs, "|OO", (char**)kwlist, &f_obj, &metric_obj)) {
  //   return -1;
  // }

  // --------------------------
  // Parse keyword arguments
  // --------------------------
  if (kwargs && kwargs != Py_None) {
    if (!PyDict_Check(kwargs)) {
      PyErr_SetString(PyExc_TypeError,
        "internal error: kwargs is not a dict");
      return -1;
    }

    // Validate keys and detect duplicates with positional arguments.
    PyObject* key = NULL;
    PyObject* value = NULL;
    Py_ssize_t pos = 0;

    while (PyDict_Next(kwargs, &pos, &key, &value)) {
      if (!PyUnicode_Check(key)) {
        PyErr_SetString(PyExc_TypeError,
          "keyword arguments must be strings");
        return -1;
      }

      const char* k = PyUnicode_AsUTF8(key);
      if (!k) return -1;

      const bool is_f            = (std::strcmp(k, "f") == 0);
      const bool is_metric       = (std::strcmp(k, "metric") == 0);
      const bool is_on_disk_path = (std::strcmp(k, "on_disk_path") == 0);
      const bool is_prefault     = (std::strcmp(k, "prefault") == 0);
      const bool is_schema_ver   = (std::strcmp(k, "schema_version") == 0);
      const bool is_seed         = (std::strcmp(k, "seed") == 0);
      const bool is_verbose      = (std::strcmp(k, "verbose") == 0);

      if (!(is_f || is_metric || is_on_disk_path || is_prefault ||
            is_schema_ver || is_seed || is_verbose)) {
        PyErr_Format(PyExc_TypeError,
          "Annoy() got an unexpected keyword argument '%s'", k);
        return -1;
      }

      if (is_f) {
        if (nargs >= 1) {
          PyErr_SetString(PyExc_TypeError,
            "Annoy() got multiple values for argument 'f'");
          return -1;
        }
        f_obj = value;
      } else if (is_metric) {
        if (nargs >= 2) {
          PyErr_SetString(PyExc_TypeError,
            "Annoy() got multiple values for argument 'metric'");
          return -1;
        }
        metric_obj = value;
      } else if (is_on_disk_path) {
        on_disk_path = value;
      } else if (is_prefault) {
        prefault = value;
      } else if (is_schema_ver) {
        schema_version = value;
      } else if (is_seed) {
        seed = value;
      } else if (is_verbose) {
        verbose = value;
      }
    }
  }

  // --------------------------
  // Validate and apply f (dimension)
  // --------------------------
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
    self->f = static_cast<int>(fv);  // (int)fv;
  }

  // --------------------------
  // Validate and apply metric
  // --------------------------
  const char* metric_c = NULL;
  if (metric_obj && metric_obj != Py_None) {
    if (!PyUnicode_Check(metric_obj)) {
      PyErr_SetString(PyExc_TypeError,
        "`metric` must be a string or None");
      return -1;
    }
    metric_c = PyUnicode_AsUTF8(metric_obj);
    if (!metric_c) return -1;

    MetricId id = metric_from_string(metric_c);
    if (id == METRIC_UNKNOWN) {
      PyErr_SetString(PyExc_ValueError,
        "Invalid metric; expected one of: angular, euclidean, manhattan, dot, hamming.");
      return -1;
    }
    self->metric_id = id;
  }

  // Default metric if f>0 and metric omitted (legacy compatibility).
  if (self->f > 0 && self->metric_id == METRIC_UNKNOWN) {
    PyErr_WarnEx(PyExc_FutureWarning,
      "The default argument for metric will be removed in a future version. "
      "Please pass metric='angular' explicitly.", 1);
    self->metric_id = METRIC_ANGULAR;
  }

  // --------------------------
  // Apply on_disk_path (stored; does not activate on-disk mode by itself)
  // --------------------------
  if (on_disk_path) {
    if (py_annoy_set_on_disk_path(self, on_disk_path, NULL) != 0)
      return -1;  // error already set
  }

  // --------------------------
  // Apply prefault (stored; per-call methods also accept prefault)
  // --------------------------
  if (prefault) {
    if (py_annoy_set_prefault(self, prefault, NULL) != 0)
      return -1;
  }

  // --------------------------
  // Apply schema_version (stored marker)
  // --------------------------
  if (schema_version) {
    if (py_annoy_set_schema_version(self, schema_version, NULL) != 0)
      return -1;
  }

  // --------------------------
  // Apply seed / verbose (stored pre-construction; applied on ensure_index)
  // --------------------------
  if (seed && seed != Py_None) {
    // Accept any Python int convertible to [0, 2**64-1]
    unsigned long long seed_arg = PyLong_AsUnsignedLongLong(seed);
    if (PyErr_Occurred()) {
      if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError,
          "seed must be an integer in the range [0, 2**64 - 1]");
      }
      return -1;
    }
    self->pending_seed     = static_cast<uint64_t>(seed_arg);
    self->has_pending_seed = true;
  }

  if (verbose && verbose != Py_None) {
    if (!PyLong_Check(verbose)) {
      PyErr_SetString(PyExc_TypeError,
        "verbose must be an integer (or None)");
      return -1;
    }
    long level = PyLong_AsLong(verbose);
    if (level == -1 && PyErr_Occurred()) return -1;

    // Clamp deterministically to [-2, 2]
    if (level < -2) level = -2;
    if (level >  2) level =  2;

    self->pending_verbose     = static_cast<int>(level);
    self->has_pending_verbose = true;
  }

  // Eagerly construct the underlying C++ index only when both f and metric are known.
  if (self->f > 0 && self->metric_id != METRIC_UNKNOWN) {
    if (!ensure_index(self)) return -1;
  }

  return 0;
}

// tp_dealloc: safe destruction with Py_CLEAR for GC safety
// tp_dealloc: destroy C++ resources and free the Python wrapper
static void py_an_dealloc(py_annoy* self) {
  // if (!self) return;

  // Deallocator for a GC-enabled extension type.
  //
  // Deterministic teardown order:
  //   1) Untrack from cyclic GC (so traversal won't see a half-torn object)
  //   2) Put the object into a "safe" state for potential re-entrancy
  //   3) Release Python references (__dict__)
  //   4) Release C++ resources
  //   5) Destroy placement-new fields
  //   6) Free the Python object memory

  if (Py_TYPE(self)->tp_flags & Py_TPFLAGS_HAVE_GC) {
    // if (PyObject_GC_IsTracked((PyObject*)self)) { PyObject_GC_UnTrack((PyObject*)self); }
    // PyObject_GC_IsTracked() is only available starting with Python 3.9.
    // For Python 3.8 compatibility we untrack unconditionally.
    // (On CPython this is safe even if the object is not currently tracked.)
    PyObject_GC_UnTrack((PyObject*)self);
  }

  // Clear instance dictionary (if enabled).
  Py_CLEAR(self->dict);

  // Clear weak references first (if enabled). This prevents callbacks from
  // observing a partially torn-down object.
  PyObject_ClearWeakRefs((PyObject*)self);

  // Make the wrapper resilient to any finalizers triggered while clearing
  // Python references below.
  //
  // Important: set self->ptr to NULL *before* deleting to prevent any accidental
  // re-entrancy (or future code changes) from double-freeing the same pointer.
  AnnoyIndexInterface<int32_t, float>* ptr = self->ptr;

  // 1) Release OS-backed / heap resources (C++).
  if (ptr) {
    // unload() should be idempotent in Annoy core
    // but we guard anyway in case of future changes.
    try {
      ptr->unload();
    } catch (...) {
      // Never let exceptions escape tp_dealloc
    }
    self->ptr = NULL;
    delete ptr;
  }

  // 2) Destroy placement-new std::string members
  try {
    self->on_disk_path.~basic_string();
  } catch (...) {
    // ignore destruction errors
  }

  // 3) Free the Python object
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// ---------------------------------------------------------------------
// Shared kw-arg helpers
// ---------------------------------------------------------------------

// Resolve prefault in a deterministic way:
// - if prefault_obj is provided and not None: use its truthiness
// - else: use self->prefault
// Returns true on success, false if a Python error is set.
static inline bool resolve_prefault_arg(
  const py_annoy* self,
  PyObject* prefault_obj,
  bool* out_prefault) {
  if (!out_prefault) return false;
  bool prefault = self->prefault;
  if (prefault_obj && prefault_obj != Py_None) {
    const int truth = PyObject_IsTrue(prefault_obj);
    if (truth < 0) return false;
    prefault = (truth != 0);
  }
  *out_prefault = prefault;
  return true;
}

// ===================== Annoy Attr =========================

// The metric is exposed via the get/set table (py_annoy_getset).
// Expose only the core numeric attribute via tp_members.
// Internal / debug-only snapshot (must be READONLY to prevent bypassing setters)
// _f, _metric_id  (and _on_disk_path via getset alias)
static PyMemberDef py_annoy_members[] = {
  {
    (char*)"_schema_version",
    T_INT, offsetof(py_annoy, schema_version),
    READONLY,  // 0
    (char*)"internal: raw schema_version value (read-only). Use .schema_version property instead."
  },

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

  {
    (char*)"_prefault",
    T_BOOL, offsetof(py_annoy, prefault),
    // READONLY is mandatory: otherwise obj._prefault = True bypasses your .prefault setter and breaks sync.
    READONLY,  // 0
    (char*)"internal: raw prefault value (read-only). Use .prefault property instead."
  },

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
    PyErr_SetString(PyExc_ValueError,
      "f cannot be None (use 0 for lazy inference)");
    return -1;
  }
  if (!PyLong_Check(value)) {
    PyErr_SetString(PyExc_TypeError,
      "f must be an integer");
    return -1;
  }
  if (self->ptr) {
    PyErr_SetString(PyExc_AttributeError,
      "Cannot change f after the index has been created.");
    return -1;
  }
  long fv = PyLong_AsLong(value);
  if (fv == -1 && PyErr_Occurred()) return -1;
  if (fv < 0) {
    PyErr_SetString(PyExc_ValueError,
      "f must be non-negative (0 means infer from first vector)");
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
    PyErr_SetString(PyExc_AttributeError,
      "Cannot delete metric attribute");
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
    PyErr_SetString(PyExc_TypeError,
      "metric must be a string (or None)");
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
    PyErr_SetString(PyExc_TypeError,
      "on_disk_path must be a string or None");
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

// Getter for 'prefault'
static PyObject* py_annoy_get_prefault(
  py_annoy* self,
  void*) {
  if (self->prefault) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

// Setter for 'prefault'
static int py_annoy_set_prefault(
  py_annoy* self,
  PyObject* value,
  void*) {
  if (value == NULL) {
    PyErr_SetString(PyExc_AttributeError,
      "Cannot delete prefault attribute");
    return -1;
  }
  // Treat None as "reset to default (False)" for forward compatibility.
  if (value == Py_None) {
    self->prefault = false;
    return 0;
  }
  int truth = PyObject_IsTrue(value);
  if (truth < 0) return -1;  // propagates error
  self->prefault = (truth != 0);
  return 0;
}

// Getter for 'schema_version'
static PyObject* py_annoy_get_schema_version(
  py_annoy* self,
  void*) {
  return PyLong_FromLong((long)self->schema_version);
}

// Setter for 'schema_version'
static int py_annoy_set_schema_version(
  py_annoy* self,
  PyObject* value,
  void*) {
  if (value == NULL) {
    PyErr_SetString(PyExc_AttributeError,
      "Cannot delete schema_version attribute");
    return -1;
  }
  // Allow None to reset to the default sentinel (0).
  if (value == Py_None) {
    self->schema_version = 0;
    return 0;
  }
  if (!PyLong_Check(value)) {
    PyErr_SetString(PyExc_TypeError,
      "schema_version must be an integer (or None)");
    return -1;
  }
  long v = PyLong_AsLong(value);
  if (v == -1 && PyErr_Occurred()) return -1;
  if (v < 0) {
    PyErr_SetString(PyExc_ValueError,
      "schema_version must be non-negative");
    return -1;
  }
  self->schema_version = static_cast<int>(v);
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
    (char*)"prefault",
    (getter)py_annoy_get_prefault,
    (setter)py_annoy_set_prefault,
    (char*)kPrefaultDoc,
    NULL
  },

  {
    (char*)"schema_version",
    (getter)py_annoy_get_schema_version,
    (setter)py_annoy_set_schema_version,
    (char*)kSchemaVersionDoc,
    NULL
  },

  {
    (char*)"_on_disk_path",
    (getter)py_annoy_get_on_disk_path,
    NULL,  // read-only alias of on_disk_path (prevents bypassing validation)
    (char*)"internal: alias of on_disk_path (read-only). Use .on_disk_path to set.",
    NULL
  },

  // Instance dictionary (__dict__).
  //
  // Notes
  // -----
  // - Backed by `tp_dictoffset` (py_annoy::dict).
  // - Implemented as a getset descriptor so the dict can be allocated lazily.
  //
  // See Also
  // --------
  // PyObject_GenericGetDict, PyObject_GenericSetDict
  {
    (char*)"__dict__",
    (getter)PyObject_GenericGetDict,
    (setter)PyObject_GenericSetDict,
    (char*)"Instance attribute dictionary (created lazily).",
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
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
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
  PyObject* prefault_obj = NULL;  // "S|p" int prefault = 0;

  // NOTE: `prefault=None` means "use self.prefault".
  static const char* kwlist[] = {"byte", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "S|O", (char**)kwlist, &byte_object, &prefault_obj)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
    return NULL;
  }

  // Defensive: "S" already enforces a bytes-like object in CPython.
  if (!PyBytes_Check(byte_object)) {
    PyErr_SetString(PyExc_TypeError,
      "Expected `byte` to be bytes");
    return NULL;
  }

  // Deterministic prefault resolution:
  // - if prefault is provided and not None → use it
  // - else → use the stored self.prefault
  bool prefault = false;  // bool prefault = (prefault_flag != 0);
  if (!resolve_prefault_arg(self, prefault_obj, &prefault)) {
    return NULL;  // Python error already set by PyObject_IsTrue
  }

  Py_ssize_t length = PyBytes_Size(byte_object);
  // uint8_t* raw_byte = reinterpret_cast<uint8_t*>(PyBytes_AsString(byte_object));
  uint8_t* raw_byte =
      reinterpret_cast<uint8_t*>(PyBytes_AsString(byte_object));
  vector<uint8_t> v(raw_byte, raw_byte + length);

  ScopedError error;
  if (!self->ptr->deserialize(&v, prefault, &error.err)) {
    PyErr_SetString(PyExc_IOError, error.err ? error.err : (char*)"deserialize failed");
    return NULL;
  }

  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF; // Py_RETURN_TRUE;
}

// Internal helper: get memory usage in bytes via serialize() size.
// Returns true on success, false if a Python error was set.
//
// IMPORTANT:
// - This helper does *not* check whether the index is built.
// - Callers that require a built index must check build status first.
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

// Return whether the current index has a built forest.
//
// Annoy's C++ core reports build status via get_n_trees():
// - 0 trees => not built (or has been unbuilt)
// - >0 trees => built and query-ready
static bool is_index_built(
  const py_annoy* self) {
  return (self && self->ptr && self->ptr->get_n_trees() > 0);
}

// Compute memory usage only if the index is built.
//
// This wrapper is used by user-facing APIs (info(), memory_usage()) so that
// "memory usage" is only reported after :meth:`build` has completed.
//
// Parameters
// ----------
// self : py_annoy*
//     Index wrapper instance.
// out_byte : uint64_t*
//     Output: size in bytes (valid only if *out_available is true).
// out_available : bool*
//     Output: whether memory usage is available (i.e., built).
//
// Returns
// -------
// ok : bool
//     True on success, False if a Python error was set.
static bool get_memory_usage_byte_if_built(
  py_annoy* self,
  uint64_t* out_byte,
  bool* out_available) {
  if (!out_byte || !out_available) return false;

  if (!is_index_built(self)) {
    *out_byte = 0;
    *out_available = false;
    return true;
  }

  *out_available = true;
  return get_memory_usage_byte(self, out_byte);
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
    args, kwargs, "", (char**)kwlist)) {
    return NULL;
  }

  // No initialized index => no memory usage.
  if (!self->ptr)
    Py_RETURN_NONE;

  // Memory usage is only available after :meth:`build`.
  uint64_t byte = 0;
  bool available = false;
  if (!get_memory_usage_byte_if_built(self, &byte, &available)) {
    return NULL;  // error already set
  }
  if (!available) {
    Py_RETURN_NONE;
  }

  return PyLong_FromUnsignedLongLong(
      static_cast<unsigned long long>(byte));
}

// ---------------------------------------------------------------------
// info(include_n_items=True, include_n_trees=True, include_memory=None) -> dict
//
// Returns a structured, JSON-like summary dict.
//
// Design goals:
//   * deterministic key order (insertion order)
//   * no side-effects
//   * allow callers to disable expensive fields (memory sizing)
// ---------------------------------------------------------------------
static PyObject* annoy_build_summary_dict(
  py_annoy* self,
  bool include_n_items,
  bool include_n_trees,
  bool include_memory) {
  PyObject* d = NULL;

  PyObject* py_f = NULL;
  PyObject* py_metric = NULL;
  PyObject* py_path = NULL;
  PyObject* py_prefault = NULL;
  PyObject* py_schema = NULL;
  PyObject* py_seed = NULL;
  PyObject* py_verbose = NULL;

  PyObject* py_items = NULL;
  PyObject* py_trees = NULL;
  PyObject* py_byte = NULL;
  PyObject* py_mib = NULL;

  d = PyDict_New();
  if (!d) goto fail;

  // Base parameters (always present; stable order)
  py_f = PyLong_FromLong((long)self->f);
  if (!py_f) goto fail;
  if (PyDict_SetItemString(d, "f", py_f) < 0) goto fail;

  {
    const char* metric_c = metric_to_cstr(self->metric_id);
    if (!metric_c) {
      py_metric = Py_None;
      Py_INCREF(py_metric);
    } else {
      py_metric = PyUnicode_FromString(metric_c);
      if (!py_metric) goto fail;
    }
    if (PyDict_SetItemString(d, "metric", py_metric) < 0) goto fail;
  }

  if (self->on_disk_path.empty()) {
    py_path = Py_None;
    Py_INCREF(py_path);
  } else {
    py_path = PyUnicode_FromString(self->on_disk_path.c_str());
    if (!py_path) goto fail;
  }
  if (PyDict_SetItemString(d, "on_disk_path", py_path) < 0) goto fail;

  py_prefault = PyBool_FromLong(self->prefault ? 1 : 0);
  if (!py_prefault) goto fail;
  if (PyDict_SetItemString(d, "prefault", py_prefault) < 0) goto fail;

  py_schema = PyLong_FromLong((long)self->schema_version);
  if (!py_schema) goto fail;
  if (PyDict_SetItemString(d, "schema_version", py_schema) < 0) goto fail;

  if (self->has_pending_seed) {
    py_seed = PyLong_FromUnsignedLongLong(
      (unsigned long long)self->pending_seed);
    if (!py_seed) goto fail;
  } else {
    py_seed = Py_None;
    Py_INCREF(py_seed);
  }
  if (PyDict_SetItemString(d, "seed", py_seed) < 0) goto fail;

  if (self->has_pending_verbose) {
    py_verbose = PyLong_FromLong((long)self->pending_verbose);
    if (!py_verbose) goto fail;
  } else {
    py_verbose = Py_None;
    Py_INCREF(py_verbose);
  }
  if (PyDict_SetItemString(d, "verbose", py_verbose) < 0) goto fail;

  // Optional keys (included only when requested)
  if (include_n_items || include_n_trees) {
    int64_t n_items64 = 0;
    int64_t n_trees64 = 0;
    if (self->ptr) {
      n_items64 = static_cast<int64_t>(self->ptr->get_n_items());
      n_trees64 = static_cast<int64_t>(self->ptr->get_n_trees());
    }

    if (include_n_items) {
      py_items = PyLong_FromLongLong(n_items64);
      if (!py_items) goto fail;
      if (PyDict_SetItemString(d, "n_items", py_items) < 0) goto fail;
    }
    if (include_n_trees) {
      py_trees = PyLong_FromLongLong(n_trees64);
      if (!py_trees) goto fail;
      if (PyDict_SetItemString(d, "n_trees", py_trees) < 0) goto fail;
    }
  }

  // include_memory=True may be expensive for large indexes.
  //
  // Memory usage is only reported after :meth:`build` has created the forest.
  // If the index is not built, memory fields are omitted.
  if (include_memory) {
    uint64_t byte_u64 = 0;
    bool available = false;
    if (!get_memory_usage_byte_if_built(self, &byte_u64, &available)) goto fail;

    if (available) {
      const double mib = static_cast<double>(byte_u64) / 1024.0 / 1024.0;

      py_byte = PyLong_FromUnsignedLongLong(byte_u64);
      py_mib  = PyFloat_FromDouble(mib);
      if (!py_byte || !py_mib) goto fail;

      if (PyDict_SetItemString(d, "memory_usage_byte", py_byte) < 0) goto fail;
      if (PyDict_SetItemString(d, "memory_usage_mib", py_mib) < 0) goto fail;
    }
  }

  // Success
  Py_DECREF(py_f);
  Py_XDECREF(py_metric);
  Py_DECREF(py_path);
  Py_DECREF(py_prefault);
  Py_DECREF(py_schema);
  Py_DECREF(py_seed);
  Py_DECREF(py_verbose);
  Py_XDECREF(py_items);
  Py_XDECREF(py_trees);
  Py_XDECREF(py_byte);
  Py_XDECREF(py_mib);
  return d;

fail:
  Py_XDECREF(py_f);
  Py_XDECREF(py_metric);
  Py_XDECREF(py_path);
  Py_XDECREF(py_prefault);
  Py_XDECREF(py_schema);
  Py_XDECREF(py_seed);
  Py_XDECREF(py_verbose);
  Py_XDECREF(py_items);
  Py_XDECREF(py_trees);
  Py_XDECREF(py_byte);
  Py_XDECREF(py_mib);
  Py_XDECREF(d);
  return NULL;
}

static PyObject* py_an_info(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  // Optional controls for expensive / optional fields.
  //
  // Memory usage is only available after :meth:`build`.
  //
  // include_memory semantics:
  // - None (default): include memory fields only if the index is built.
  // - True: include memory fields if built; otherwise omit them.
  // - False: always omit memory fields.
  PyObject* include_n_items_obj = Py_True;
  PyObject* include_n_trees_obj = Py_True;
  PyObject* include_memory_obj  = Py_None; // Py_False
  static const char* kwlist[] = {"include_n_items", "include_n_trees", "include_memory", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "|OOO",
        (char**)kwlist,
        &include_n_items_obj,
        &include_n_trees_obj,
        &include_memory_obj)) {
    return NULL;
  }

  bool include_n_items = true;
  bool include_n_trees = true;

  // None means "use default" for include_n_items/include_n_trees (default=True).
  if (include_n_items_obj && include_n_items_obj != Py_None) {
    const int truth = PyObject_IsTrue(include_n_items_obj);
    if (truth < 0) return NULL;
    include_n_items = (truth != 0);
  }
  if (include_n_trees_obj && include_n_trees_obj != Py_None) {
    const int truth = PyObject_IsTrue(include_n_trees_obj);
    if (truth < 0) return NULL;
    include_n_trees = (truth != 0);
  }
  // include_memory is tri-state: bool | None
  bool include_memory = false;
  if (!include_memory_obj || include_memory_obj == Py_None) {
    include_memory = is_index_built(self);
  } else {
    const int truth = PyObject_IsTrue(include_memory_obj);
    if (truth < 0) return NULL;
    include_memory = (truth != 0);
    // Memory usage is only meaningful after build; if not built, omit.
    if (include_memory && !is_index_built(self)) {
      include_memory = false;
    }
  }

  return annoy_build_summary_dict(self, include_n_items, include_n_trees, include_memory);
}

// Forward declaration for use in method table (defined near __repr__).
static PyObject* py_an_repr_info(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs);

// Forward declaration for use in method table (defined near __repr__).
static PyObject* py_an_repr_html(
  PyObject* obj,
  PyObject* Py_UNUSED(ignored));

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
    PyErr_SetString(PyExc_RuntimeError,
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
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
    return NULL;
  }

  if (n_trees <= 0 && n_trees != -1) {
    PyErr_SetString(PyExc_ValueError,
      "n_trees must be a positive integer or -1 handle internally");
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
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
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
  PyObject* prefault_obj = NULL;  // "s|p" int prefault_flag = 0;

  // NOTE: `prefault=None` means "use self.prefault".
  static const char* kwlist[] = {"fn", "prefault", NULL};

  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "s|O", (char**)kwlist, &filename, &prefault_obj)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized. "
      "Call add_item() + build() (or load()) before save()."
    );
    return NULL;
  }

  // Deterministic prefault resolution:
  // - if prefault is provided and not None → use it
  // - else → use the stored self.prefault
  bool prefault = false;  // bool prefault = (prefault_flag != 0);
  if (!resolve_prefault_arg(self, prefault_obj, &prefault)) {
    return NULL;  // Python error already set by PyObject_IsTrue
  }

  ScopedError error;
  if (!self->ptr->save(filename, prefault, &error.err)) {
    PyErr_SetString(PyExc_IOError,
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
  PyObject* prefault_obj = NULL;  // "s|p" int prefault_flag = 0;

  // NOTE: `prefault=None` means "use self.prefault".
  static const char* kwlist[] = {"fn", "prefault", NULL};

  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "s|O", (char**)kwlist, &filename, &prefault_obj)) {
    return NULL;
  }

  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized.\n"
      "Construct it first, e.g., Annoy(f, metric='angular'), "
      "then call .load(fn).");
    return NULL;
  }

  // Deterministic: raise FileNotFoundError early if the path is missing.
  if (!file_exists(filename)) {
    // file_exists already set FileNotFoundError
    return NULL;
  }

  // Deterministic prefault resolution:
  // - if prefault is provided and not None → use it
  // - else → use the stored self.prefault
  bool prefault = false;  // bool prefault = (prefault_flag != 0);
  if (!resolve_prefault_arg(self, prefault_obj, &prefault)) {
    return NULL;  // Python error already set by PyObject_IsTrue
  }

  ScopedError error;
  if (!self->ptr->load(filename, prefault, &error.err)) {
    PyErr_SetString(PyExc_IOError,
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
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
    return NULL;
  }

  if (n_neighbors <= 0) {
    PyErr_SetString(PyExc_ValueError,
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
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized. "
      "Create it with Annoy(f, metric='angular') and add_item() before querying.");
    return NULL;
  }

  if (n_neighbors <= 0) {
    PyErr_SetString(PyExc_ValueError,
      "`n` (number of neighbors) must be positive");
    return NULL;
  }

  if (self->f <= 0) {
    PyErr_SetString(PyExc_RuntimeError,
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
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
    return NULL;
  }

  // Validate indice against current index state (not building stage)
  if (!check_constraints(self, indice, /*building=*/false))
    return NULL;

  if (self->f <= 0) {
    PyErr_SetString(PyExc_RuntimeError,
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
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
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
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
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
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
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

  // Versioned state for forward compatibility.
  // Contract:
  //   - Readers must ignore unknown keys.
  //   - New writers should only add keys (no semantic change to existing keys).
  PyObject* v = PyLong_FromLong(1);
  if (!v || PyDict_SetItemString(state, "_pickle_version", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // Instance dictionary (__dict__) (optional).
  // We store it explicitly because this type defines a custom __getstate__.
  if (self->dict) {
    v = PyDict_Copy(self->dict);
    if (!v) { Py_DECREF(state); return NULL; }
  } else {
    Py_INCREF(Py_None);
    v = Py_None;
  }
  if (PyDict_SetItemString(state, "__dict__", v) < 0) {
    Py_DECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // f
  v = PyLong_FromLong((long)self->f);
  if (!v || PyDict_SetItemString(state, "f", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // metric: canonical string (empty string means "unknown / lazy")
  const char* metric_c = metric_to_cstr(self->metric_id);
  v = PyUnicode_FromString(metric_c ? metric_c : "");
  if (!v || PyDict_SetItemString(state, "metric", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // metric_id: integer enum for faster restoration / future compatibility
  v = PyLong_FromLong((long)self->metric_id);
  if (!v || PyDict_SetItemString(state, "metric_id", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // on_disk_path (optional metadata)
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

  // prefault / schema_version: stored wrapper configuration
  v = PyBool_FromLong(self->prefault ? 1 : 0);
  if (!v || PyDict_SetItemString(state, "prefault", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  v = PyLong_FromLong((long)self->schema_version);
  if (!v || PyDict_SetItemString(state, "schema_version", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // Pending seed/verbose (internal but deterministic and stable).
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

  // User-facing mirrors for convenience (additive keys).
  // These are redundant with has_pending_* / pending_* but friendlier.
  if (self->has_pending_seed) {
    v = PyLong_FromUnsignedLongLong(
      (unsigned long long)self->pending_seed);
  } else {
    Py_INCREF(Py_None);
    v = Py_None;
  }
  if (!v || PyDict_SetItemString(state, "seed", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  if (self->has_pending_verbose) {
    v = PyLong_FromLong((long)self->pending_verbose);
  } else {
    Py_INCREF(Py_None);
    v = Py_None;
  }
  if (!v || PyDict_SetItemString(state, "verbose", v) < 0) {
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

  // Reset optional instance dictionary as well.
  Py_CLEAR(self->dict);

  // Restore instance dictionary (__dict__) if present.
  PyObject* dict_state = PyDict_GetItemString(state, "__dict__");  // borrowed
  if (dict_state && dict_state != Py_None) {
    if (!PyDict_Check(dict_state)) {
      PyErr_SetString(PyExc_TypeError,
        "`__dict__` in pickle state must be a dict or None");
      return NULL;
    }
    self->dict = PyDict_Copy(dict_state);
    if (!self->dict) return NULL;
  }

  self->f = 0;
  self->metric_id = METRIC_UNKNOWN;

  self->on_disk_active = false;
  self->on_disk_path.clear();

  self->prefault = false;
  self->schema_version = 0;

  self->pending_seed        = 0ULL;
  self->pending_verbose     = 0;
  self->has_pending_seed    = false;
  self->has_pending_verbose = false;

  // --------------------------
  // Core configuration
  // --------------------------

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
      PyErr_SetString(PyExc_ValueError,
        "`metric_id` in pickle state is invalid");
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

  // prefault (optional)
  PyObject* prefault_obj = PyDict_GetItemString(state, "prefault");  // borrowed
  if (prefault_obj && prefault_obj != Py_None) {
    const int truth = PyObject_IsTrue(prefault_obj);
    if (truth < 0) return NULL;
    self->prefault = (truth != 0);
  }

  // schema_version (optional)
  PyObject* schema_obj = PyDict_GetItemString(state, "schema_version");  // borrowed
  if (schema_obj && schema_obj != Py_None) {
    long sv = PyLong_AsLong(schema_obj);
    if (sv == -1 && PyErr_Occurred()) return NULL;
    if (sv < 0) {
      PyErr_SetString(PyExc_ValueError,
        "`schema_version` in pickle state must be non-negative");
      return NULL;
    }
    self->schema_version = (int)sv;
  }

  // --------------------------
  // Pending seed/verbose (optional)
  // --------------------------

  // Primary (internal) keys: has_pending_seed/pending_seed
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

  // Fallback (user-facing) key: seed
  if (!has_seed_obj) {
    PyObject* seed2 = PyDict_GetItemString(state, "seed");  // borrowed
    if (seed2 && seed2 != Py_None) {
      unsigned long long sv = PyLong_AsUnsignedLongLong(seed2);
      if (PyErr_Occurred()) {
        if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
          PyErr_Clear();
          PyErr_SetString(PyExc_ValueError,
            "`seed` in pickle state must be in [0, 2**64 - 1]");
        }
        return NULL;
      }
      self->pending_seed = (uint64_t)sv;
      self->has_pending_seed = true;
    }
  }

  // Primary (internal) keys: has_pending_verbose/pending_verbose
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

  // Fallback (user-facing) key: verbose
  if (!has_verbose_obj) {
    PyObject* verbose2 = PyDict_GetItemString(state, "verbose");  // borrowed
    if (verbose2 && verbose2 != Py_None) {
      long vv = PyLong_AsLong(verbose2);
      if (vv == -1 && PyErr_Occurred()) return NULL;
      self->pending_verbose = (int)vv;
      self->has_pending_verbose = true;
    }
  }

  // --------------------------
  // Payload snapshot (optional)
  // --------------------------

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
    "See Also\n"
    "--------\n"
    "build : Build the forest after adding items.\n"
    "unbuild : Remove trees to allow adding more items.\n"
    "get_nns_by_item, get_nns_by_vector : Query nearest neighbours.\n"
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
    "\n"
    "    If set to ``n_trees=-1``, annoy trees are built dynamically until\n"
    "    the index reaches approximately twice the number of items\n"
    "    ``_n_nodes >= 2 * n_items``.\n"
    "\n"
    "    Guidelines:\n"
    "\n"
    "    * Small datasets (<10k samples): 10-20 trees.\n"
    "    * Medium datasets (10k-1M samples): 20-50 trees.\n"
    "    * Large datasets (>1M samples): 50-100+ trees.\n"
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
    "See Also\n"
    "--------\n"
    "add_item : Add vectors before building.\n"
    "unbuild : Drop trees to add more items.\n"
    "get_nns_by_item, get_nns_by_vector : Query nearest neighbours.\n"
    "save, load : Persist the index to/from disk.\n"
    "\n"
    "References\n"
    "----------\n"
    ".. [1] Erik Bernhardsson, \"Annoy: Approximate Nearest Neighbours in C++/Python\".\n"
    "\n"
    "See Also\n"
    "--------\n"
    "add_item : Add vectors before building.\n"
    "unbuild : Drop trees to add more items.\n"
    "get_nns_by_item, get_nns_by_vector : Query nearest neighbours.\n"
    "save, load : Persist the index to/from disk.\n"
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
    "\n"
    "See Also\n"
    "--------\n"
    "build : Rebuild the forest after adding new items.\n"
    "add_item : Add items (only valid when no trees are built).\n"
  },

  // ------------------------------------------------------------------
  // Persistence: disk + byte + memory usage
  // ------------------------------------------------------------------

  {
    "save",
    (PyCFunction)py_an_save,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "save(fn, prefault=None)\n"
    "\n"
    "Persist the index to a binary file on disk.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "fn : str\n"
    "    Path to the output file. Existing files will be overwritten.\n"
    "prefault : bool or None, optional, default=None\n"
    "    If True, aggressively fault pages into memory during save.\n"
    "    If None, use the stored :attr:`prefault` value.\n"
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
    "\n"
    "Notes\n"
    "-----\n"
    "The output file will be overwritten if it already exists.\n"
    "Use prefault=None to fall back to the stored :attr:`prefault` setting.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "load : Load an index from disk.\n"
    "serialize : Snapshot to bytes for in-memory persistence.\n"
  },

  {
    "load",
    (PyCFunction)py_an_load,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "load(fn, prefault=None)\n"
    "\n"
    "Load (mmap) an index from disk into the current object.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "fn : str\n"
    "    Path to a file previously created by :meth:`save` or\n"
    "    :meth:`on_disk_build`.\n"
    "prefault : bool or None, optional, default=None\n"
    "    If True, fault pages into memory when the file is mapped.\n"
    "    If None, use the stored :attr:`prefault` value.\n"
    "    Primarily useful on some platforms for very large indexes.\n"
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
    "\n"
    "See Also\n"
    "--------\n"
    "save : Save the current index to disk.\n"
    "on_disk_build : Build directly using an on-disk backing file.\n"
    "unload : Release mmap resources.\n"
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
    "\n"
    "See Also\n"
    "--------\n"
    "load : Memory-map an on-disk index into this object.\n"
    "on_disk_build : Configure on-disk build mode.\n"
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
    "\n"
    "See Also\n"
    "--------\n"
    "build : Build trees after adding items (on-disk backed).\n"
    "load : Memory-map the built index.\n"
    "save : Persist the built index to disk.\n"
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
    "deserialize(byte, prefault=None)\n"
    "\n"
    "Restore the index from a serialized byte string.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "byte : bytes\n"
    "    Byte string produced by :meth:`serialize`.\n"
    "prefault : bool or None, optional, default=None\n"
    "    If True, fault pages into memory while restoring.\n"
    "    If None, use the stored :attr:`prefault` value.\n"
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
    "Approximate memory usage of the index in bytes.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "n_bytes : int or None\n"
    "    Approximate number of bytes used by the index. Returns ``None`` if the\n"
    "    index is not initialized or the forest has not been built yet.\n"
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
    "    If True, return a ``(indices, distances)`` tuple. Otherwise return only\n"
    "    the list of indices.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "indices : list[int] | tuple[list[int], list[float]]\n"
    "    If ``include_distances=False``: list of neighbour item ids.\n"
    "    If ``include_distances=True``: ``(indices, distances)``.\n"
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
    "    If True, return a ``(indices, distances)`` tuple. Otherwise return only\n"
    "    the list of indices.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "indices : list[int] | tuple[list[int], list[float]]\n"
    "    If ``include_distances=False``: list of neighbour item ids.\n"
    "    If ``include_distances=True``: ``(indices, distances)``.\n"
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
    "info(include_n_items=True, include_n_trees=True, include_memory=None) -> dict\n"
    "\n"
    "Return a structured summary of the index.\n"
    "\n"
    "This method returns a JSON-like Python dictionary that is easier to\n"
    "inspect programmatically than the legacy multi-line string format.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "include_n_items : bool, optional, default=True\n"
    "    If True, include ``n_items``.\n"
    "include_n_trees : bool, optional, default=True\n"
    "    If True, include ``n_trees``.\n"
    "include_memory : bool or None, optional, default=None\n"
    "    Controls whether memory usage fields are included.\n"
    "    \n"
    "    * ``None``: include memory usage only if the index is built.\n"
    "    * ``True``: include memory usage if available (built).\n"
    "    * ``False``: omit memory usage fields.\n"
    "    \n"
    "    Memory usage is computed after :meth:`build` and may be expensive for\n"
    "    very large indexes.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "info : dict\n"
    "    Dictionary describing the current index state.\n"
    "\n"
    "Notes\n"
    "----\n"
    "- Some keys are optional depending on include_* flags.\n"
    "\n"
    "Keys:\n"
    "\n"
    "* f : int, default=0\n"
    "    Dimensionality of the index.\n"
    "* metric : str, default='angular'\n"
    "    Distance metric name.\n"
    "* on_disk_path : str, default=''\n"
    "    Path used for on-disk build, if configured.\n"
    "* prefault : bool, default=False\n"
    "    If True, aggressively fault pages into memory during save.\n"
    "    Primarily useful on some platforms for very large indexes.\n"
    "* schema_version : int, default=0\n"
    "    Stored schema/version marker on this object (reserved for future use).\n"
    "* seed : int or None, optional, default=None\n"
    "    Non-negative integer seed. If called before the index is constructed,\n"
    "    the seed is stored and applied when the C++ index is created.\n"
    "* verbose : int or None, optional, default=None\n"
    "    Verbosity level. Values are clamped to the range ``[-2, 2]``.\n"
    "    ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.\n"
    "    Logging level inspired by gradient-boosting libraries:\n"
    "\n"
    "    * ``<= 0`` : quiet (warnings only)\n"
    "    * ``1``    : info (Annoy's ``verbose=True``)\n"
    "    * ``>= 2`` : debug (currently same as info, reserved for future use)\n"
    "\n"
    "Optional Keys:\n"
    "\n"
    "* n_items : int\n"
    "    Number of items currently stored.\n"
    "* n_trees : int\n"
    "    Number of built trees in the forest.\n"
    "* memory_usage_byte : int\n"
    "    Approximate memory usage in bytes. Present only when requested and available.\n"
    "* memory_usage_mib : float\n"
    "    Approximate memory usage in MiB. Present only when requested and available.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "serialize : Create a binary snapshot of the index.\n"
    "deserialize : Restore from a binary snapshot.\n"
    "save : Persist the index to disk.\n"
    "load : Load the index from disk.\n"
    "\n"
    "Examples\n"
    "--------\n"
    ">>> info = idx.info()\n"
    ">>> info['f']\n"
    "100\n"
    ">>> info['n_items']\n"
    "1000\n"
  },

  {
    "repr_info",
    (PyCFunction)py_an_repr_info,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "repr_info(include_n_items=True, include_n_trees=True, include_memory=None) -> str\n"
    "\n"
    "Return a dict-like string representation with optional extra fields.\n"
    "\n"
    "Unlike ``__repr__``, this method can include additional fields on demand.\n"
    "Note that ``include_memory=True`` may be expensive for large indexes.\n"
    "Memory is calculated after :meth:`build`.\n"
  },

  // Rich display (Jupyter)
  {
    "_repr_html_",
    (PyCFunction)py_an_repr_html,
    METH_NOARGS,
    (char*)
    "_repr_html_() -> str\n"
    "\n"
    "Return an HTML representation of the Annoy index for Jupyter notebooks.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "html : str\n"
    "    HTML string (safe to embed) describing the current configuration.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This representation is deterministic and side-effect free. It intentionally\n"
    "avoids expensive operations such as serialization or memory-usage estimation.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "info : Return a Python dict with configuration and metadata.\n"
    "__repr__ : Text representation.\n"
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
    "    * ``metric_id`` : int\n"
    "    * ``on_disk_path`` : str | None\n"
    "    * ``prefault`` : bool\n"
    "    * ``schema_version`` : int\n"
    "    * ``has_pending_seed`` : bool\n"
    "    * ``pending_seed`` : int\n"
    "    * ``has_pending_verbose`` : bool\n"
    "    * ``pending_verbose`` : int\n"
    "    * ``seed`` : int | None\n"
    "    * ``verbose`` : int | None\n"
    "    * ``data`` : bytes | None\n"
    "\n"
    "Notes\n"
    "-----\n"
    "If the underlying C++ index is initialized, ``data`` contains a serialized\n"
    "snapshot (see :meth:`serialize`). Otherwise, ``data`` is ``None``.\n"
    "\n"
    "Configuration keys like ``prefault`` and ``schema_version`` are stored on the\n"
    "Python wrapper and restored deterministically. Unknown keys are ignored.\n"
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
    "Users do not need to call this directly.\n"
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

// --------------------- py_an_repr begin ---------------------

// Helper pattern: create a value (new ref), store it, then DECREF it.
// On failure: clean up and return NULL.
#define ANN_REPR_SET_ITEM(KEY, VALUE_EXPR)                                \
do {                                                                      \
  PyObject* _v = (VALUE_EXPR);                                            \
  if (!_v) goto fail;                                                     \
  if (PyDict_SetItemString(d, (KEY), _v) < 0) {                           \
    Py_DECREF(_v);                                                        \
    goto fail;                                                            \
  }                                                                       \
  Py_DECREF(_v);                                                          \
} while (0)
// #undef ANN_REPR_SET_ITEM

// __repr__(self) → "Annoy({ ... })" (dict-like, stable order)
//
// Must be deterministic, side-effect free, and cheap:
// - only render base parameter fields
// - never compute memory usage
//
// We therefore DO NOT call .info() here, because .info() may compute memory
// usage via serialization, which can be expensive for large indexes.
//
// Instead, we emit a small dict in the same *shape* as .info().
// Expensive fields (memory usage) are returned as None when the index is built.
static PyObject* py_an_repr(
  PyObject* obj) {
  py_annoy* self = reinterpret_cast<py_annoy*>(obj);  // (py_annoy*)self_obj;

  PyObject* d = NULL;
  PyObject* d_repr = NULL;
  PyObject* out = NULL;

  d = annoy_build_summary_dict(
    self,
    /*include_n_items=*/false,
    /*include_n_trees=*/false,
    /*include_memory=*/false);
  if (!d) goto fail;

  d_repr = PyObject_Repr(d);
  if (!d_repr) goto fail;

  out = PyUnicode_FromFormat("Annoy(%U)", d_repr);
  if (!out) goto fail;

  Py_DECREF(d_repr);
  Py_DECREF(d);
  return out;

fail:
  Py_XDECREF(out);
  Py_XDECREF(d_repr);
  Py_DECREF(d);
  return NULL;
}

// --------------------- py_an_repr end ---------------------

// --------------------- py_an_repr_info begin ---------------------

// repr_info(include_n_items=True, include_n_trees=True, include_memory=None) -> str
//
// Richer dict-like string representation on demand.
// Unlike __repr__, callers can opt into additional fields.
// include_memory=True may be expensive for large indexes.
static PyObject* py_an_repr_info(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  PyObject* include_n_items_obj = Py_True;
  PyObject* include_n_trees_obj = Py_True;
  PyObject* include_memory_obj  = Py_None; // Py_False
  static const char* kwlist[] = {"include_n_items", "include_n_trees", "include_memory", NULL};

  if (!PyArg_ParseTupleAndKeywords(
        args,
        kwargs,
        "|OOO",
        (char**)kwlist,
        &include_n_items_obj,
        &include_n_trees_obj,
        &include_memory_obj)) {
    return NULL;
  }

  bool include_n_items = true;
  bool include_n_trees = true;

  // None means "use default" for include_n_items/include_n_trees (default=True).
  if (include_n_items_obj && include_n_items_obj != Py_None) {
    const int truth = PyObject_IsTrue(include_n_items_obj);
    if (truth < 0) return NULL;
    include_n_items = (truth != 0);
  }
  if (include_n_trees_obj && include_n_trees_obj != Py_None) {
    const int truth = PyObject_IsTrue(include_n_trees_obj);
    if (truth < 0) return NULL;
    include_n_trees = (truth != 0);
  }
  // include_memory is tri-state: bool | None
  bool include_memory = false;
  if (!include_memory_obj || include_memory_obj == Py_None) {
    include_memory = is_index_built(self);
  } else {
    const int truth = PyObject_IsTrue(include_memory_obj);
    if (truth < 0) return NULL;
    include_memory = (truth != 0);
    // Memory usage is only meaningful after build; if not built, omit.
    if (include_memory && !is_index_built(self)) {
      include_memory = false;
    }
  }

  PyObject* d = NULL;
  PyObject* d_repr = NULL;
  PyObject* out = NULL;

  d = annoy_build_summary_dict(self, include_n_items, include_n_trees, include_memory);
  if (!d) goto fail;

  d_repr = PyObject_Repr(d);
  if (!d_repr) goto fail;

  out = PyUnicode_FromFormat("Annoy(%U)", d_repr);
  if (!out) goto fail;

  Py_DECREF(d_repr);
  Py_DECREF(d);
  return out;

fail:
  Py_XDECREF(out);
  Py_XDECREF(d_repr);
  Py_XDECREF(d);
  return NULL;
}

// --------------------- py_an_repr_info end ---------------------

// ------------------------- HTML repr for notebooks --------------------------
// This is a Jupyter-friendly rich representation inspired by scikit-learn's
// estimator HTML diagram. The repr is implemented as a lightweight, self-contained
// HTML snippet (markup + CSS + minimal JS) and kept strictly side-effect free.
//
// Design goals
// ------------
// - Keep logic identical; only refactor for readability/maintainability.
// - No imports / no extra side effects: we only consult sys.modules.
// - Stable row order (deterministic ordering of fields).
// - Centralize CSS/JS templates and placeholder replacement.
// - Centralize "append a parameter row" patterns to prevent repetition.
//
// Assets
// ------
// CSS and JS are primarily loaded from sibling files in the package:
//
//   scikitplot/cexternals/_annoy/_repr_html/estimator.css
//   scikitplot/cexternals/_annoy/_repr_html/params.css
//   scikitplot/cexternals/_annoy/_repr_html/estimator.js
//
// This mirrors scikit-learn's approach: keep HTML assets as plain files for
// maintainability and reuse. At runtime we resolve these files relative to the
// extension module's __file__ directory. If assets are missing, we fall back to
// embedded minimal CSS/JS so repr remains functional.
//
// const std::string icon = "\xF0\x9F\x97\x97";  // icon.c_str()
// const char* copy = "\xF0\x9F\x97\x97";  // ✔︎ 🗗 (U+1F5D7) UTF-8: F0 9F 97 97

// Each HTML repr invocation must have a unique DOM id.
//
// Rationale
// ---------
// - In notebooks, the same Python object can be displayed multiple times.
// - A per-object DOM id (e.g., based on an address) can therefore be duplicated
//   across outputs, which breaks event wiring.
// - This counter increments under the GIL, making it deterministic within a
//   process. The address of the counter acts as a per-process salt.
//
// ------------------------- CSS/JS asset loading -----------------------------
//
// For maintainability, the notebook HTML repr loads its CSS and JS templates
// from plain text files shipped with the Python package:
//
//   scikitplot/cexternals/_annoy/_repr_html/estimator.css
//   scikitplot/cexternals/_annoy/_repr_html/params.css
//   scikitplot/cexternals/_annoy/_repr_html/estimator.js
//
// Runtime lookup is performed relative to the extension module binary path:
//   <module_dir>/_repr_html/{estimator.css,params.css,estimator.js}
//
// The asset files may contain the placeholder string "__ANNOY_REPR_ID__" which
// is replaced at runtime with the per-repr container id, enabling strict scoping
// without global DOM side effects.

// ------------------------- CSS/JS asset helper -------------------------

// static inline bool annoy_path_is_sep(char c) {
//   return (c == '/') || (c == '\\');
// }

// static std::string annoy_path_dirname(const std::string& path) {
//   if (path.empty()) return std::string();

//   size_t i = path.size();
//   while (i > 0 && annoy_path_is_sep(path[i - 1])) {
//     --i;
//   }
//   while (i > 0 && !annoy_path_is_sep(path[i - 1])) {
//     --i;
//   }
//   while (i > 0 && annoy_path_is_sep(path[i - 1])) {
//     --i;
//   }
//   return path.substr(0, i);
// }

// --------------------- annoy repr_html utilities ---------------------

// Embedded fallbacks: used if the on-disk assets are unavailable.
static const char kAnnoyReprIdPlaceholder[] = "__ANNOY_REPR_ID__";

// NOTE: Keep these templates in one place. They may be replaced by on-disk
// assets in the future, but embedded fallbacks must remain deterministic.
static const char kAnnoyReprCssFallback[] = R"CSS(
#__ANNOY_REPR_ID__ .annoy-box{border:1px solid #d0d7de;border-radius:6px;display:inline-block;min-width:280px;}
#__ANNOY_REPR_ID__ details{margin:0;padding:0;}
#__ANNOY_REPR_ID__ summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;padding:8px 10px;font:12px/1.35 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}
#__ANNOY_REPR_ID__ summary::-webkit-details-marker{display:none;}
#__ANNOY_REPR_ID__ .annoy-title{font-weight:600;}
#__ANNOY_REPR_ID__ .annoy-links{margin-left:auto;display:flex;align-items:center;gap:8px;}
#__ANNOY_REPR_ID__ .annoy-links a{color:#0969da;text-decoration:none;}
#__ANNOY_REPR_ID__ .annoy-links a:hover{text-decoration:underline;}
#__ANNOY_REPR_ID__ .annoy-sep{color:#57606a;}
#__ANNOY_REPR_ID__ .annoy-subtitle{font-weight:600;}
#__ANNOY_REPR_ID__ .annoy-arrow::before{content:'\25B6';display:inline-block;width:14px;}
#__ANNOY_REPR_ID__ details[open] > summary .annoy-arrow::before{content:'\25BC';}
#__ANNOY_REPR_ID__ .annoy-body{padding:0 10px 10px 10px;}
#__ANNOY_REPR_ID__ .annoy-table{border-collapse:collapse;width:100%;font:12px/1.35 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}
#__ANNOY_REPR_ID__ .annoy-table th,.annoy-table td{border-top:1px solid #eaeef2;padding:6px 6px;text-align:left;vertical-align:top;}
#__ANNOY_REPR_ID__ .annoy-table th{font-weight:600;}
#__ANNOY_REPR_ID__ .annoy-td-btn{width:64px;}
#__ANNOY_REPR_ID__ .annoy-copy{font:11px/1 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;border:1px solid #d0d7de;border-radius:6px;padding:3px 6px;background:#f6f8fa;cursor:pointer;}
#__ANNOY_REPR_ID__ .annoy-copy:active{transform:translateY(1px);}
#__ANNOY_REPR_ID__ .annoy-key{white-space:nowrap;}
#__ANNOY_REPR_ID__ .annoy-value{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;}
)CSS";


static const char kAnnoyReprJsFallback[] = R"JS(
(function(){
  var root=document.getElementById('__ANNOY_REPR_ID__');
  if(!root)return;
  var btns=root.querySelectorAll('button.annoy-copy');
  for(var i=0;i<btns.length;i++){
    btns[i].addEventListener('click',function(e){
      e.preventDefault();
      var tr=this.closest('tr'); if(!tr) return;
      var val=tr.querySelector('.annoy-value'); if(!val) return;
      var txt=val.textContent || ''; if(!txt) return;

      function done(btn){
        var old=btn.textContent;
        btn.textContent='✔︎ Copied';
        setTimeout(function(){btn.textContent=old;},800);
      }

      if(navigator.clipboard && navigator.clipboard.writeText){
        navigator.clipboard.writeText(txt).then(done.bind(null,this),function(){done(this);}.bind(this));
      } else {
        var ta=document.createElement('textarea');
        ta.value=txt;
        ta.style.position='fixed';
        ta.style.left='-9999px';
        document.body.appendChild(ta);
        ta.select();
        try{document.execCommand('copy');}catch(_e){}
        document.body.removeChild(ta);
        done(this);
      }
    });
  }
})();
)JS";

// Global repr sequence.
// NOTE: This is intentionally process-global and incrementing, because we want
// a unique id per repr call (not per object). Calls happen under the GIL.
static unsigned long long g_annoy_repr_html_seq = 0;

static void annoy_make_repr_html_id(char* buf, size_t bufsz) {
  const unsigned long long seq = (unsigned long long)(++g_annoy_repr_html_seq);
  const void* salt = (const void*)&g_annoy_repr_html_seq;
  (void)snprintf(buf, bufsz, "annoy-repr-%p-%llu", salt, seq);
}

// Replace all occurrences of kAnnoyReprIdPlaceholder in `tpl` with `id`,
// appending the result into `out`.
static void annoy_append_template_with_id(std::string& out, const char* tpl, const char* id) {
  if (!tpl || !id) return;

  const char* p = tpl;
  const size_t ph_len = strlen(kAnnoyReprIdPlaceholder);

  while (*p) {
    const char* hit = strstr(p, kAnnoyReprIdPlaceholder);
    if (!hit) {
      out.append(p);
      return;
    }
    // Append text up to the placeholder.
    out.append(p, (size_t)(hit - p));
    // Append the replacement id.
    out.append(id);
    // Continue scanning after the placeholder.
    p = hit + ph_len;
  }
}

// --------------------- HTML escaping + row rendering ---------------------

static void annoy_html_escape_append(const char* s, std::string& out) {
  if (!s) return;
  for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
    switch (*p) {
      case '&': out.append("&amp;"); break;
      case '<': out.append("&lt;"); break;
      case '>': out.append("&gt;"); break;
      case '"': out.append("&quot;"); break;
      case '\'': out.append("&#x27;"); break;
      default: out.push_back((char)*p); break;
    }
  }
}

static int annoy_append_pyrepr_html_escaped(PyObject* value, std::string& out) {
  PyObject* r = PyObject_Repr(value);  // new ref
  if (!r) return -1;

  const char* s = PyUnicode_AsUTF8(r);
  if (!s) {
    Py_DECREF(r);
    return -1;
  }
  annoy_html_escape_append(s, out);
  Py_DECREF(r);
  return 0;
}

static int annoy_append_html_row(std::string& html, const char* key, PyObject* value) {
  // NOTE: This function does not steal references. Caller manages DECREF.
  html.append("<tr>");
  html.append("<td class=\"annoy-td-btn\">"
              "<button type=\"button\" class=\"annoy-copy\" title=\"Copy value\" aria-label=\"Copy value\">🗗 Copy</button>"
              "</td>");
  html.append("<td class=\"annoy-key\">");

  annoy_html_escape_append(key, html);

  html.append("</td>");
  html.append("<td class=\"annoy-value\">");

  if (annoy_append_pyrepr_html_escaped(value, html) < 0) return -1;

  html.append("</td>");
  html.append("</tr>");
  return 0;
}

// Centralize the repeated "create -> append row -> decref" pattern.
static int annoy_append_param_row(std::string& html, const char* key, PyObject* v) {
  if (!v) return -1;
  if (annoy_append_html_row(html, key, v) < 0) {
    Py_DECREF(v);
    return -1;
  }
  Py_DECREF(v);
  return 0;
}

static int annoy_append_param_long(std::string& html, const char* key, long x) {
  return annoy_append_param_row(html, key, PyLong_FromLong(x));
}

static int annoy_append_param_bool(std::string& html, const char* key, bool x) {
  return annoy_append_param_row(html, key, PyBool_FromLong(x ? 1 : 0));
}

static int annoy_append_param_none_or_str(std::string& html, const char* key, const std::string& s) {
  if (s.empty()) {
    Py_INCREF(Py_None);
    return annoy_append_param_row(html, key, Py_None);
  }
  return annoy_append_param_row(html, key, PyUnicode_FromString(s.c_str()));
}

static int annoy_append_param_none_or_ull(std::string& html, const char* key, bool present, unsigned long long x) {
  if (!present) {
    Py_INCREF(Py_None);
    return annoy_append_param_row(html, key, Py_None);
  }
  return annoy_append_param_row(html, key, PyLong_FromUnsignedLongLong(x));
}

static int annoy_append_param_none_or_long(std::string& html, const char* key, bool present, long x) {
  if (!present) {
    Py_INCREF(Py_None);
    return annoy_append_param_row(html, key, Py_None);
  }
  return annoy_append_param_row(html, key, PyLong_FromLong(x));
}

// --------------------- docs link derivation (strict, no imports) ---------------------

static int annoy_parse_major_minor_strict(const char* ver, unsigned long* major, unsigned long* minor) {
  // Strict parse:
  // - skip non-digits
  // - parse major digit-run
  // - require '.' then parse minor digit-run
  // - success only if both present
  if (!ver || !major || !minor) return 0;

  const char* p = ver;
  while (*p && (*p < '0' || *p > '9')) ++p;

  unsigned long maj = 0, min = 0;
  int have_major = 0, have_minor = 0;

  while (*p >= '0' && *p <= '9') {
    have_major = 1;
    maj = maj * 10 + (unsigned long)(*p - '0');
    ++p;
  }
  if (*p != '.') return 0;
  ++p;

  while (*p >= '0' && *p <= '9') {
    have_minor = 1;
    min = min * 10 + (unsigned long)(*p - '0');
    ++p;
  }

  if (!have_major || !have_minor) return 0;
  *major = maj;
  *minor = min;
  return 1;
}

struct annoy_docs_links {
  const char* docs_dev;     // never null
  char stable_mm[32];       // empty string if not available
  char docs_stable[256];    // empty string if not available
};

static void annoy_docs_links_init(annoy_docs_links* out) {
  out->docs_dev =
      "https://scikit-plots.github.io/dev/modules/generated/scikitplot.cexternals._annoy.Annoy.html";
  out->stable_mm[0] = '\0';
  out->docs_stable[0] = '\0';
}

static void annoy_docs_links_try_fill_stable(annoy_docs_links* out) {
  // Deterministic + side-effect constrained:
  // - no imports
  // - consult sys.modules only (module must already be loaded)
  PyObject* modules = PyImport_GetModuleDict();  // borrowed
  PyObject* scikitplot_mod = modules ? PyDict_GetItemString(modules, "scikitplot") : NULL;  // borrowed
  if (!scikitplot_mod) return;

  PyObject* ver_obj = PyObject_GetAttrString(scikitplot_mod, "__version__");  // new ref
  if (!ver_obj) return;

  const char* ver = PyUnicode_AsUTF8(ver_obj);
  if (ver && *ver) {
    unsigned long major = 0, minor = 0;
    if (annoy_parse_major_minor_strict(ver, &major, &minor)) {
      (void)snprintf(out->stable_mm, sizeof(out->stable_mm), "%lu.%lu", major, minor);
      (void)snprintf(
          out->docs_stable,
          sizeof(out->docs_stable),
          "https://scikit-plots.github.io/%s/modules/generated/scikitplot.cexternals._annoy.Annoy.html",
          out->stable_mm);
    }
  }
  Py_DECREF(ver_obj);
}

// --------------------- main repr_html entry point ---------------------

// --------------------- py_an_repr_html begin ---------------------

static PyObject* py_an_repr_html(PyObject* obj, PyObject* Py_UNUSED(ignored)) {
  py_annoy* self = reinterpret_cast<py_annoy*>(obj);

  // Unique container id per repr invocation (not per object).
  // This prevents duplicate ids when the same object is displayed multiple times.
  char idbuf[96];
  annoy_make_repr_html_id(idbuf, sizeof(idbuf)); // (void)snprintf(idbuf, sizeof(idbuf), "annoy-repr-%p", (void*)self);

  annoy_docs_links links;
  annoy_docs_links_init(&links);
  annoy_docs_links_try_fill_stable(&links);

  std::string html;
  html.reserve(4096);

  // Outer scoped container.
  html.append("<div id=\"");
  html.append(idbuf);
  html.append("\" class=\"annoy-repr\">");

  // Scoped CSS: embedded fallback today; may be replaced by on-disk assets later.
  html.append("<style>");
  // scikitplot/cexternals/_annoy/_repr_html/estimator.css
  // scikitplot/cexternals/_annoy/_repr_html/params.css
  annoy_append_template_with_id(html, kAnnoyReprCssFallback, idbuf);
  // html.append("#"); html.append(idbuf); html.append(" .annoy-box{border:1px solid #d0d7de;border-radius:6px;display:inline-block;min-width:280px;}");
  // html.append("#"); html.append(idbuf); html.append(" details{margin:0;padding:0;}");
  // html.append("#"); html.append(idbuf); html.append(" summary{cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;padding:8px 10px;font:12px/1.35 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}");
  // html.append("#"); html.append(idbuf); html.append(" summary::-webkit-details-marker{display:none;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-title{font-weight:600;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-links{margin-left:auto;display:flex;align-items:center;gap:8px;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-links a{color:#0969da;text-decoration:none;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-links a:hover{text-decoration:underline;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-sep{color:#57606a;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-subtitle{font-weight:600;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-arrow::before{content:'\\25B6';display:inline-block;width:14px;}");
  // html.append("#"); html.append(idbuf); html.append(" details[open] > summary .annoy-arrow::before{content:'\\25BC';}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-body{padding:0 10px 10px 10px;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-table{border-collapse:collapse;width:100%;font:12px/1.35 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-table th,.annoy-table td{border-top:1px solid #eaeef2;padding:6px 6px;text-align:left;vertical-align:top;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-table th{font-weight:600;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-td-btn{width:64px;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-copy{font:11px/1 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;border:1px solid #d0d7de;border-radius:6px;padding:3px 6px;background:#f6f8fa;cursor:pointer;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-copy:active{transform:translateY(1px);}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-key{white-space:nowrap;}");
  // html.append("#"); html.append(idbuf); html.append(" .annoy-value{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;}");
  html.append("</style>");

  // Header / layout.
  html.append("<div class=\"annoy-box\">");
  html.append("<details class=\"annoy-main\">");
  html.append("<summary>");
  html.append("<span class=\"annoy-arrow\"></span>");
  html.append("<span class=\"annoy-title\">Annoy</span>");

  // Docs links: dev always, stable only when major.minor extracted strictly.
  // Document link ▼ Annoy … dev | X.Y (clickable)
  html.append("<span class=\"annoy-links\">"
              "<a href=\"");
  html.append(links.docs_dev);
  html.append("\" target=\"_blank\" rel=\"noopener noreferrer\" title=\"Annoy docs (dev)\">dev</a>");
  if (links.docs_stable[0] != '\0') {
    html.append("<span class=\"annoy-sep\">|</span>"
                "<a href=\"");
    html.append(links.docs_stable);
    html.append("\" target=\"_blank\" rel=\"noopener noreferrer\" title=\"Annoy docs (installed)\">");
    html.append(links.stable_mm);
    html.append("</a>");
  }
  html.append("</span>");

  html.append("</summary>");
  html.append("<div class=\"annoy-body\">");

  // Parameters section (not expanded by default once the outer block is expanded).
  html.append("<details class=\"annoy-sub\">");
  html.append("<summary><span class=\"annoy-arrow\"></span><span class=\"annoy-subtitle\">Parameters</span></summary>");
  html.append("<table class=\"annoy-table\">");
  html.append("<thead><tr><th></th><th>Parameter</th><th>Value</th></tr></thead><tbody>");

  // Rows MUST be appended in stable order (do not reorder without explicit intent).
  if (annoy_append_param_long(html, "f", (long)self->f) < 0) goto fail;

  // metric
  {
    const char* metric_c = metric_to_cstr(self->metric_id);
    if (!metric_c) {
      Py_INCREF(Py_None);
      if (annoy_append_param_row(html, "metric", Py_None) < 0) goto fail;
    } else {
      if (annoy_append_param_row(html, "metric", PyUnicode_FromString(metric_c)) < 0) goto fail;
    }
  }

  // on_disk_path
  if (annoy_append_param_none_or_str(html, "on_disk_path", self->on_disk_path) < 0) goto fail;

  if (annoy_append_param_bool(html, "prefault", self->prefault != 0) < 0) goto fail;
  if (annoy_append_param_long(html, "schema_version", (long)self->schema_version) < 0) goto fail;

  // seed / verbose (pending values)
  if (annoy_append_param_none_or_ull(html, "seed",
                                    self->has_pending_seed != 0,
                                    (unsigned long long)self->pending_seed) < 0) goto fail;

  if (annoy_append_param_none_or_long(html, "verbose",
                                     self->has_pending_verbose != 0,
                                     (long)self->pending_verbose) < 0) goto fail;

  html.append("</tbody></table>");
  html.append("</details>");  // parameters details
  html.append("</div>");      // body
  html.append("</details>");  // main details
  html.append("</div>");      // box

  // JS: embedded fallback today; may be replaced by on-disk assets later.

  html.append("<script>");
  // scikitplot/cexternals/_annoy/_repr_html/estimator.js
  annoy_append_template_with_id(html, kAnnoyReprJsFallback, idbuf);
  // html.append("(function(){");
  // html.append("var root=document.getElementById('"); html.append(idbuf); html.append("');");
  // html.append("if(!root)return;");
  // html.append("var btns=root.querySelectorAll('button.annoy-copy');");
  // html.append("for(var i=0;i<btns.length;i++){btns[i].addEventListener('click',function(e){");
  // html.append("e.preventDefault();");
  // html.append("var tr=this.closest('tr'); if(!tr) return;");
  // html.append("var val=tr.querySelector('.annoy-value'); if(!val) return;");
  // html.append("var txt=val.textContent || ''; if(!txt) return;");
  // html.append("function done(btn){var old=btn.textContent; btn.textContent='✔︎ Copied'; setTimeout(function(){btn.textContent=old;},800);} ");
  // html.append("if(navigator.clipboard && navigator.clipboard.writeText){navigator.clipboard.writeText(txt).then(done.bind(null,this),function(){done(this);}.bind(this));}");
  // html.append("else{var ta=document.createElement('textarea'); ta.value=txt; ta.style.position='fixed'; ta.style.left='-9999px'; document.body.appendChild(ta); ta.select(); try{document.execCommand('copy');}catch(_e){} document.body.removeChild(ta); done(this);} ");
  // html.append("});}");
  // html.append("})();");
  html.append("</script>");

  html.append("</div>");  // outer container

  return PyUnicode_FromStringAndSize(html.c_str(), (Py_ssize_t)html.size());

fail:
  // Any Python error is already set by failing API calls.
  return NULL;
}

// --------------------- py_an_repr_html end ---------------------

// --------------------- Sequence protocol begin ---------------------

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

// --------------------- Sequence protocol end ---------------------

// --------------------- GC support begin ---------------------

// GC support: allow cycles via instance __dict__.
static int py_an_traverse(
  PyObject* obj,
  visitproc visit,
  void* arg) {
  py_annoy* self = (py_annoy*)obj;
  Py_VISIT(self->dict);
  return 0;
}

static int py_an_clear(
  PyObject* obj) {
  py_annoy* self = (py_annoy*)obj;
  Py_CLEAR(self->dict);
  return 0;
}

// --------------------- GC support end ---------------------

// ======================= Module / types ===================================
// https://docs.python.org/3/c-api/typeobj.html

// NOTE: Positional initialization of PyTypeObject.
//
// CPython has inserted/renamed fields in PyTypeObject across minor versions
// (e.g., 3.8 introduced tp_vectorcall_offset). A positional initializer list
// is therefore brittle: it can compile on one Python version but break (or
// silently mis-initialize) on another.
//
// To keep this extension "Python 3 minor-version consistent", we keep a
// zero-initialized PyTypeObject and fill only the slots we actually use.
static PyTypeObject py_annoy_type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  // "annoy.Annoy",  /* tp_name matches the actual exported module/object prints like <class 'annoy.Annoy'> */
  // "scikitplot.cexternals._annoy.Annoy",                          /* tp_name */
  // sizeof(py_annoy),                                              /* tp_basicsize */
  // 0,                                                             /* tp_itemsize */
  // (destructor)py_an_dealloc,                                     /* tp_dealloc */
  // 0,                                                             /* tp_vectorcall_offset (was tp_print) */
  // 0,                                                             /* tp_getattr */
  // 0,                                                             /* tp_setattr */
  // 0,                                                             /* tp_as_async (was tp_compare) */
  // (reprfunc)py_an_repr,                                          /* tp_repr */
  // 0,                                                             /* tp_as_number */
  // &py_annoy_as_sequence,                                         /* tp_as_sequence → supports len() */
  // 0,                                                             /* tp_as_mapping */
  // 0,                                                             /* tp_hash  */
  // 0,                                                             /* tp_call */
  // 0,                                                             /* tp_str */
  // 0,                                                             /* tp_getattro */
  // 0,                                                             /* tp_setattro */
  // 0,                                                             /* tp_as_buffer */
  // Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC, /* tp_flags */
  // (char*)kAnnoyTypeDoc,                                          /* tp_doc  */
  // (traverseproc)py_an_traverse,                                  /* tp_traverse */
  // (inquiry)py_an_clear,                                          /* tp_clear */
  // 0,                                                             /* tp_richcompare */
  // 0,                                                             /* tp_weaklistoffset */
  // 0,                                                             /* tp_iter */
  // 0,                                                             /* tp_iternext */
  // py_annoy_methods,                                              /* tp_methods */
  // py_annoy_members,                                              /* tp_members */
  // py_annoy_getset,                                               /* tp_getset */
  // 0,                                                             /* tp_base */
  // 0,                                                             /* tp_dict */
  // 0,                                                             /* tp_descr_get */
  // 0,                                                             /* tp_descr_set */
  // offsetof(py_annoy, dict),                                      /* tp_dictoffset */
  // (initproc)py_an_init,                                          /* tp_init */
  // PyType_GenericAlloc,                                           /* tp_alloc */
  // py_an_new,                                                     /* tp_new  */
};

// Initialize the Annoy type in a version-resilient way.
//
// Returns
// -------
// int
//     0 on success, -1 on error (with a Python exception set).
static int init_py_annoy_type() {
  static bool initialized = false;
  if (initialized) {
    return 0;
  }

  // Basic identity / size
  // NOTE: tp_name intentionally keeps the packaging path used by the
  // higher-level wrapper so pickling and repr remain stable.
  py_annoy_type.tp_name = "scikitplot.cexternals._annoy.Annoy";
  py_annoy_type.tp_basicsize = sizeof(py_annoy);
  py_annoy_type.tp_itemsize = 0;

  // Lifecycle
  py_annoy_type.tp_dealloc = (destructor)py_an_dealloc;
  py_annoy_type.tp_new = py_an_new;
  py_annoy_type.tp_init = (initproc)py_an_init;
  py_annoy_type.tp_alloc = PyType_GenericAlloc;
  py_annoy_type.tp_free = PyObject_GC_Del;  // required for Py_TPFLAGS_HAVE_GC

  // Introspection / repr
  py_annoy_type.tp_repr = (reprfunc)py_an_repr;

  // Protocols
  py_annoy_type.tp_as_sequence = &py_annoy_as_sequence;  // supports len(x)

  // Attribute access
  // ----------------
  // We explicitly use CPython's generic attribute handlers. This guarantees that
  // the per-instance dictionary slot (tp_dictoffset -> py_annoy::dict) is used
  // for dynamic attributes (obj.__dict__, setattr/getattr) in a version-stable
  // way.
  //
  // IMPORTANT:
  // - `tp_dictoffset` controls the *instance* dictionary (`__dict__`).
  // - `tp_dict` (PyTypeObject field) is the dictionary of attributes on the
  //   *type object itself* and must not be modified directly with PyDict_* APIs.
  //   See CPython "Type Object Structures" docs.
  py_annoy_type.tp_getattro = PyObject_GenericGetAttr;
  py_annoy_type.tp_setattro = PyObject_GenericSetAttr;

  // Methods / attributes
  py_annoy_type.tp_methods = py_annoy_methods;
  py_annoy_type.tp_members = py_annoy_members;
  py_annoy_type.tp_getset = py_annoy_getset;
  py_annoy_type.tp_dictoffset = offsetof(py_annoy, dict);

  // GC hooks
  py_annoy_type.tp_traverse = (traverseproc)py_an_traverse;
  py_annoy_type.tp_clear = (inquiry)py_an_clear;

  // Weakref support: allow weakref.ref(obj) for this extension type.
  py_annoy_type.tp_weaklistoffset = offsetof(py_annoy, weakreflist);

  // Docs / flags
  py_annoy_type.tp_doc = (char*)kAnnoyTypeDoc;
  py_annoy_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC;

#ifdef Py_TPFLAGS_HAVE_WEAKREFS
  py_annoy_type.tp_flags |= Py_TPFLAGS_HAVE_WEAKREFS;
#endif

  if (PyType_Ready(&py_annoy_type) < 0) {
    return -1;
  }

  initialized = true;
  return 0;
}

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

// Add a type object to the module under an exported name.
//
// Notes
// -----
// - PyModule_AddObject() steals a reference on success and does not steal on
//   failure. This helper centralizes the reference-counting contract.
static int AddTypeAlias(PyObject* m, const char* name, PyTypeObject* type) {
  Py_INCREF(type);
  if (PyModule_AddObject(m, name, (PyObject*)type) < 0) {
    Py_DECREF(type);
    return -1;
  }
  return 0;
}

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
  // Required because this type enables cyclic GC (Py_TPFLAGS_HAVE_GC).
  // py_annoy_type.tp_free = PyObject_GC_Del;
  // if (PyType_Ready(&py_annoy_type) < 0)
  //   return NULL;

  // Prepare Annoy type (version-resilient slot initialization).
  if (init_py_annoy_type() < 0) {
    return NULL;
  }

  PyObject* m;

#if PY_MAJOR_VERSION >= 3
  // Initialize NumPy C API here if this module starts using NumPy.
  // (Currently disabled to keep the backend dependency-free.)
  // if (!InitNumpyIfNeeded()) return NULL;
  m = PyModule_Create(&annoylibmodule);
#else
  m = Py_InitModule("annoylib", module_methods);
#endif

  if (!m)
    return NULL;

  // Cache the extension module directory for HTML repr asset lookup.
  // annoy_set_repr_module_dir(m);

  // Expose `Annoy` class
  // Py_INCREF(&py_annoy_type);
  // if (PyModule_AddObject(m, "Annoy", (PyObject*)&py_annoy_type) < 0) {
  //   Py_DECREF(&py_annoy_type);
  //   Py_DECREF(m);
  //   return NULL;
  // }
  //
  // Expose `Annoy` class and stable aliases.
  // The upstream annoy package exposes `AnnoyIndex`.
  if (AddTypeAlias(m, "Annoy", &py_annoy_type) < 0 ||
      AddTypeAlias(m, "AnnoyIndex", &py_annoy_type) < 0) {
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
