// scikitplot/cexternals/_annoy/src/annoymodule.cc
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
#include <cstdio>  // std::snprintf, std::vsnprintf
#include <cstdlib>  // std::getenv
#include <exception>
#include <fstream>  // std::ifstream
#include <memory>
#include <limits>
#include <new>  // std::bad_alloc
#include <string>  // std::string
#include <stdexcept>
#include <unordered_map>
#include <vector>

// #include <iostream>  // std::cout << R"(...)" << std::endl;

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
    "C++-powered :class:`~scikitplot.cexternals._annoy.Annoy` type.\n"
    "For day-to-day work, prefer a higher-level Python wrapper\n"
    "(if your project provides one)::\n"
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

// ---------------------------------------------------------------------------
// Seed semantics (deterministic, explicit)
// ---------------------------------------------------------------------------
//
// Annoy uses Kiss64Random for all metrics exposed in this module.
// - Kiss64Random::default_seed is deterministic and non-zero.
// - Historically some callers passed seed=0; we normalize that to the default
//   seed to respect the invariant "seed must be != 0" while keeping API stable.
//
// Semantics (deterministic):
//   * seed is None -> "use Annoy's default seed".
//   * seed is int  -> validated to be in [0, 2**64 - 1] and normalized via
//                    normalize_seed_u64() (so seed=0 maps to the default seed with warn).
static inline uint64_t annoy_default_seed_u64() {
  return static_cast<uint64_t>(Kiss64Random::default_seed);
}

static inline uint64_t normalize_seed_u64(uint64_t seed) {
  return static_cast<uint64_t>(Kiss64Random::normalize_seed(seed));
}

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
      snprintf(path, sizeof(path), ".");
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

// Check whether a path exists (no Python exceptions). Used to avoid accidental
// truncation when enabling on-disk build mode via a property assignment.
static inline bool path_exists_noexc(const char* filename) {
  if (!filename || !*filename) return false;
  struct stat st;
  return (stat(filename, &st) == 0);
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
metric : {"angular", "cosine", "euclidean", "l2", "lstsq", "manhattan", "l1", "cityblock", "taxicab", "dot", "@", ".", "dotproduct", "inner", "innerproduct", "hamming"} or None, optional, default=None
    Distance metric (one of 'angular', 'euclidean', 'manhattan', 'dot', 'hamming').
    If omitted and ``f > 0``, defaults to ``'angular'`` (cosine-like).
    If omitted and ``f == 0``, metric may be set later before construction.
    If None, behavior depends on ``f``:

    * If ``f > 0``: defaults to ``'angular'`` (legacy behavior; may emit a
      :class:`FutureWarning`).
    * If ``f == 0``: leaves the metric unset (lazy). You may set
      :attr:`metric` later before construction, or it will default to
      ``'angular'`` on first :meth:`add_item`.
on_disk_path : str or None, optional, default=None
    If provided, configures the path for on-disk building. When the underlying
    index exists, this enables on-disk build mode (equivalent to calling
    :meth:`on_disk_build` with the same filename).

    Safety: Annoy core truncates the target file when enabling on-disk build.
    Therefore, if the path already exists, it is **not** truncated implicitly.
    Call :meth:`on_disk_build` explicitly to overwrite.

    In lazy mode (``f==0`` and/or ``metric is None``), activation occurs once
    the underlying C++ index is created.
prefault : bool or None, optional, default=None
    If True, request page-faulting index pages into memory when loading
    (when supported by the underlying platform/backing).
    If None, treated as ``False`` (reset to default).
schema_version : int, optional, default=None
    Serialization/compatibility strategy marker.

    This does not change the Annoy on-disk format, but it *does* control
    how the index is snapshotted in pickles.

    * ``0`` or ``1``: pickle stores a ``portable-v1`` snapshot (fast restore,
      ABI-checked).
    * ``2``: pickle stores ``canonical-v1`` (portable across ABIs; restores by
      rebuilding deterministically).
    * ``>=3``: pickle stores both portable and canonical (canonical is used as
      a fallback if the ABI check fails).

    If None, treated as ``0`` (reset to default).
seed : int or None, optional, default=None
    Non-negative integer seed. If set before the index is constructed,
    the seed is stored and applied when the C++ index is created.
    Seed value ``0`` is treated as \"use Annoy's deterministic default seed\"
    (a :class:`UserWarning` is emitted when ``0`` is explicitly provided).
verbose : int or None, optional, default=None
    Verbosity level. Values are clamped to the range ``[-2, 2]``.
    ``level >= 1`` enables Annoy's verbose logging; ``level <= 0`` disables it.
    Logging level inspired by gradient-boosting libraries:

    * ``<= 0`` : quiet (warnings only)
    * ``1``    : info (Annoy's ``verbose=True``)
    * ``>= 2`` : debug (currently same as info, reserved for future use)

Attributes
----------
f : int, default=0
    Vector dimension. ``0`` means "unknown / lazy".
metric : {'angular', 'euclidean', 'manhattan', 'dot', 'hamming'}, default="angular"
    Canonical metric name, or None if not configured yet (lazy).
on_disk_path : str or None, optional, default=None
    Configured on-disk build path. Setting this attribute enables on-disk
    build mode (equivalent to :meth:`on_disk_build`), with safety checks
    to avoid implicit truncation of existing files.
prefault : bool, default=False
    Stored prefault flag (see :meth:`load`/`:meth:`save` prefault parameters).
schema_version : int, default=0
    Reserved schema/version marker (stored; does not affect on-disk format).
y : dict | None, optional, default=None
    If provided to fit(X, y), labels are stored here after a successful build.
    You may also set this property manually. When possible, the setter enforces
    that len(y) matches the current number of items (n_items).

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

Notes
-----
* Once the underlying C++ index is created, ``f`` and ``metric`` are immutable.
  This keeps the object consistent and avoids undefined behavior.
* The C++ index is created lazily when sufficient information is available:
  when both ``f > 0`` and ``metric`` are known, or when an operation that
  requires the index is first executed.
* If ``f == 0``, the dimensionality is inferred from the first non-empty vector
  passed to :meth:`add_item` and is then fixed for the lifetime of the index.
* Assigning ``None`` to :attr:`f` is not supported. Use ``0`` for lazy
  inference (this matches ``Annoy(f=None, ...)`` at construction time).
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

>>> idx.build(n_trees=-1)
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
- ``Annoy(f=None, ...)`` is supported at construction time and is treated as ``f=0``.
  After construction, assigning ``None`` to :attr:`f` is not supported; use ``0``.
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

.. seealso::
  * :py:func:`~scipy.spatial.distance.cosine`
  * :py:func:`~scipy.spatial.distance.euclidean`
  * :py:func:`~scipy.spatial.distance.cityblock`
  * :py:func:`~scipy.sparse.coo_array.dot`
  * :py:func:`~scipy.spatial.distance.hamming`

Notes
-----
Changing ``metric`` after the index has been initialized (items added and/or
trees built) is a *structural* change: the forest and all distances depend on
the distance function.

For scikit-learn compatibility, setting a different metric on an already
initialized index will deterministically **reset** the index (drop all items,
trees, and :attr:`y`). You must call :meth:`fit` (or :meth:`add_item` +
:meth:`build`) again before querying.
)METRIC";

static const char kOnDiskPathDoc[] =
R"ODP(
Path used for on-disk build/load/save operations.

Returns
-------
str or None
    Filesystem path used for on-disk operations, or None if not configured.

.. seealso::
  * :meth:`on_disk_build`
  * :meth:`load`
  * :meth:`unload`

Notes
-----
- Assigning a string/PathLike to ``on_disk_path`` configures on-disk build mode
  (equivalent to calling :meth:`on_disk_build` with the same filename).
- Safety: Annoy core truncates the target file when enabling on-disk build.
  Therefore, an existing file is never truncated implicitly when setting this
  attribute. Call :meth:`on_disk_build` explicitly to overwrite.
- Assigning ``None`` (or an empty string) clears the configured path, but only
  when no disk-backed index is currently active.
- Clearing/changing this while an on-disk index is active is disallowed.
  Call :meth:`unload` first.
)ODP";

static const char kPrefaultDoc[] =
R"PFDOC(
Default prefault flag stored on the object.

This setting is used as the default for per-call ``prefault`` arguments when
``prefault`` is omitted or set to ``None`` in methods like :meth:`load` and
:meth:`save`.

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
Serialization/compatibility strategy marker sentinel value.

This does not change the Annoy on-disk format, but it controls how the index
is snapshotted in pickles.

Returns
-------
int
    Current schema version marker.

Notes
-----
- ``0`` or ``1``: pickle stores a ``portable-v1`` snapshot (fast restore, ABI-checked).
- ``2``: pickle stores ``canonical-v1`` (portable; restores by rebuilding deterministically).
- ``>=3``: pickle stores both portable and canonical; canonical is used as a fallback.
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

  AnnoyIndexInterface<int32_t, float>* ptr;  // X matris underlying C++ index dynamic_cast<AnnoyAngular*>(ptr)

  // Optional labels / targets associated with vectors (set by fit or manually).
  // Stored as a Python object (typically 1D array-like) and managed with ref-counting.
  PyObject* y;                  // labels/targets (y / _y), or NULL

  int f;                        // 0 means "unknown / lazy" (dimension inferred from first add_item)
  MetricId metric_id;           // METRIC_UNKNOWN means "unknown / lazy"
  // std::string metric;        // Always has a valid string object, Default constructor "" to check method .empty()

  // --- Optional on-disk path (for on_disk_build / load) ---
  bool on_disk_active;          // true if ptr is currently backed by disk (load() or on_disk_build())
  std::string on_disk_path;     // Always has a valid string object, Default constructor "" to check method .empty(), back to "" (empty string) method .clear()
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
// Forward declaration: used by structural parameter setters (e.g. metric)
// to deterministically drop index state while keeping the wrapper object.
static bool reset_index_state(py_annoy* self, const char* warn_msg);

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

  // If an on-disk path is configured, enable on-disk build mode as soon as the
  // underlying index exists (equivalent to calling on_disk_build(fn)).
  //
  // Safety: Annoy core's on_disk_build() opens the file with O_TRUNC; to avoid
  // accidental data loss, we do not implicitly truncate an existing file.
  if (!self->on_disk_path.empty() && !self->on_disk_active) {
    if (path_exists_noexc(self->on_disk_path.c_str())) {
      if (PyErr_WarnEx(
            PyExc_UserWarning,
            "on_disk_path points to an existing file; refusing to truncate it "
            "implicitly. Call on_disk_build(on_disk_path) explicitly to overwrite.",
            1) < 0) {
        delete self->ptr;
        self->ptr = NULL;
        return false;
    }
    } else {
      ScopedError error;
      if (!self->ptr->on_disk_build(self->on_disk_path.c_str(), &error.err)) {
        // Roll back to a safe state: the index exists but cannot be used.
        delete self->ptr;
        self->ptr = NULL;
        PyErr_SetString(PyExc_IOError,
          error.err ? error.err : (char*)"on_disk_build failed");
        return false;
    }
      self->on_disk_active = true;
    }
  }

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


// Fill an embedding vector from either a dense sequence or a sparse dict.
//
// Supported row formats (deterministic):
// - Dense: sequence of length f (list/tuple/array-like). If allow_missing is
//   true, elements may be None and will be imputed with missing_fill.
// - Sparse: dict {int -> float|None}. Keys must be in [0, f). If allow_missing
//   is false, the dict must contain all keys 0..f-1 and no value may be None.
//   If allow_missing is true, missing keys (and None values) are imputed with
//   missing_fill.
//
// Returns
// -------
// ok : bool
//     True on success, False if a Python exception was set.
static bool fill_embedding_from_row(
  PyObject* row,
  int f,
  bool allow_missing,
  float missing_fill,
  vector<float>* out,
  const char* context) {
  if (!out) return false;

  if (f <= 0) {
    PyErr_SetString(PyExc_RuntimeError,
      "Index dimension `f` is not set");
    return false;
  }
  if (!context) context = "X";

  // Dense path: default initialize to 0.0f.
  out->assign((size_t)f, 0.0f);

  // Sparse dict input: {feature_index -> value}
  if (PyDict_Check(row)) {
    if (allow_missing) {
      out->assign((size_t)f, missing_fill);
    }

    vector<char> seen;
    if (!allow_missing) {
      seen.assign((size_t)f, 0);
    }

    PyObject* key = NULL;
    PyObject* val = NULL;
    Py_ssize_t pos = 0;
    while (PyDict_Next(row, &pos, &key, &val)) {
      if (!PyLong_Check(key)) {
        PyErr_Format(PyExc_TypeError,
          "%s row dict keys must be integers in [0, f)", context);
        return false;
      }
      long long k = PyLong_AsLongLong(key);
      if (k == -1 && PyErr_Occurred()) return false;
      if (k < 0 || k >= (long long)f) {
        PyErr_Format(PyExc_ValueError,
          "%s row dict key %lld is out of range [0, %d)", context, k, f);
        return false;
      }

      float v = 0.0f;
      if (val == Py_None) {
        if (!allow_missing) {
          PyErr_Format(PyExc_ValueError,
            "%s row dict has None at key %lld; set missing_value to impute",
            context, k);
          return false;
        }
        v = missing_fill;
      } else {
        double dv = PyFloat_AsDouble(val);
        if (PyErr_Occurred()) return false;
        v = (float)dv;
      }

      (*out)[(size_t)k] = v;
      if (!allow_missing) {
        seen[(size_t)k] = 1;
      }
    }

    if (!allow_missing) {
      for (int i = 0; i < f; ++i) {
        if (!seen[(size_t)i]) {
          PyErr_Format(PyExc_ValueError,
            "%s row dict is missing key %d; set missing_value to impute",
            context, i);
          return false;
        }
      }
    }

    return true;
  }

  // Dense sequence input
  PyObject* seq = PySequence_Fast(row, "expected a 1D sequence of floats");
  if (!seq) return false;

  const Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
  if (len != f) {
    PyErr_Format(PyExc_ValueError,
                 "embedding length mismatch: expected %d, got %ld",
                 f, (long)len);
    Py_DECREF(seq);
    return false;
  }

  for (Py_ssize_t i = 0; i < len; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(seq, i);

    if (item == Py_None) {
      if (!allow_missing) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError,
          "%s row contains None at position %ld; set missing_value to impute",
          context, (long)i);
        return false;
      }
      (*out)[(size_t)i] = missing_fill;
      continue;
    }

    double val = PyFloat_AsDouble(item);
    if (PyErr_Occurred()) {
      Py_DECREF(seq);
      return false;
    }
    (*out)[(size_t)i] = (float)val;
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
    PyObject_GC_UnTrack((PyObject*)self);
  }

  self->dict = NULL;
  self->weakreflist = NULL;

  self->ptr = NULL;

  self->y = NULL;
  // Clear fitted labels (if any) on re-init.
  Py_CLEAR(self->y);

  self->f = 0;
  self->metric_id = METRIC_UNKNOWN;

  self->on_disk_active = false;
  // Construct placement-new std::string members (py_annoy is a C struct,
  // so tp_alloc does not run C++ constructors).
  try {
    // Placement-new: construct C++ std::string members inside the C struct.
    new (&self->on_disk_path) std::string();  // placement-new: empty path
    // self->on_disk_path.clear(); // Or clear the content Standart metod
    // Explicit destructor call for placement-new std::string member.
    // self->on_disk_path.~basic_string();  // explicit destructor for placement-new member
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

  // Clear fitted labels (if any) on re-init.
  Py_CLEAR(self->y);

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
  //         schema_version=0, seed=None, random_state=None, verbose=None)
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
  PyObject*   random_state   = NULL;
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
      const bool is_random_state = (std::strcmp(k, "random_state") == 0);
      const bool is_verbose      = (std::strcmp(k, "verbose") == 0);

      if (!(is_f || is_metric || is_on_disk_path || is_prefault ||
          is_schema_ver || is_seed || is_random_state || is_verbose)) {
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
    } else if (is_random_state) {
        random_state = value;
    } else if (is_verbose) {
        verbose = value;
    }
    }
  }

  // Disallow setting both aliases.
  if (seed && random_state) {
    PyErr_SetString(PyExc_TypeError,
      "Annoy() got multiple values for argument 'seed' (alias: random_state)"
    );
    return -1;
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
    if (PyErr_WarnEx(PyExc_FutureWarning,
      "The default argument for metric will be removed in a future version. "
      "Please pass metric='angular' explicitly.", 1) < 0) {
      return -1;
    }
    self->metric_id = METRIC_ANGULAR;
  }

  // --------------------------
  // Apply on_disk_path.
  // If provided, this enables on-disk build mode (same as calling
  // on_disk_build(fn)). For safety, an existing file is never truncated
  // implicitly; call on_disk_build(path) explicitly to overwrite.
  // In lazy mode, activation occurs once the underlying index exists
  // (see ensure_index()).
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
  // Apply seed if is not None or NULL (stored pre-construction; applied on ensure_index)
  // --------------------------
  PyObject* seed_like = seed ? seed : random_state;
  if (seed_like && seed_like != Py_None) {
    // Accept any Python int convertible to [0, 2**64 - 1].
    // seed=0 is treated as "use Annoy's deterministic default seed" and emits a warning
    // when explicitly provided (to avoid silent surprises while staying deterministic).
    unsigned long long seed_arg = PyLong_AsUnsignedLongLong(seed_like);
    if (PyErr_Occurred()) {
      if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError,
          "seed must be an integer in the range [0, 2**64 - 1]");
    }
      return -1;
    }

    if (seed_arg == 0ULL) {
      if (PyErr_WarnEx(PyExc_UserWarning,
        "seed=0 uses Annoy's default seed", 1) < 0) {
        return -1;
    }
      // Default is already deterministic; treat as "no override".
      self->pending_seed     = 0ULL;
      self->has_pending_seed = false;
    } else {
      self->pending_seed     = normalize_seed_u64(static_cast<uint64_t>(seed_arg));
      self->has_pending_seed = true;
    }
  }


  // --------------------------
  // Apply verbose if is not None or NULL (stored pre-construction; applied on ensure_index)
  // --------------------------
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

  // Clear weak references first (if enabled). This prevents callbacks from
  // observing a partially torn-down object.
  if (self->weakreflist) {
    PyObject_ClearWeakRefs((PyObject*)self);
  }

  // Clear instance dictionary (if enabled).
  Py_CLEAR(self->dict);

  // Clear fitted labels / targets (if any).
  Py_CLEAR(self->y);

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
    // Placement-new: construct C++ std::string members inside the C struct.
    // new (&self->on_disk_path) std::string();  // placement-new: empty path
    // self->on_disk_path.clear(); // Cpp clear the content Standart metod
    // Explicit destructor call for placement-new std::string member.
    self->on_disk_path.~basic_string();  // explicit destructor for placement-new member
  } catch (...) {
    // Never let exceptions escape tp_dealloc
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

  // {
  //   (char*)"_y",
  //   T_OBJECT_EX, offsetof(py_annoy, y),
  //   // READONLY is mandatory: otherwise obj._y = True bypasses your .y setter and breaks sync.
  //   READONLY,  // 0
  //   (char*)"internal: raw y object (read-only binding). WARNING: dict can still be mutated in-place. Use .y property instead."
  // },

  {NULL}  /* Sentinel */
};


// ===================== Getters/Setters (CORRECT signatures) ===============
// getter: PyObject* (py_annoy*, void*)
// setter: int (py_annoy*, PyObject*, void*)

// Optional sklearn-style fitted attribute: y (labels / targets)
//
// Notes
// -----
// scikit-learn estimators do not typically store `y`, but for Annoy it can be
// useful to attach labels/metadata for downstream retrieval. We store :attr:`y` as a
// Python object and validate basic shape constraints when possible.
//
static PyObject* py_annoy_get_y(
  py_annoy* self,
  void*) {
  if (!self || !self->y) Py_RETURN_NONE;
  Py_INCREF(self->y);
  return self->y;
}


static int py_annoy_set_y(
  py_annoy* self,
  PyObject* value,
  void*) {
  if (!self) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy object is not initialized");
    return -1;
  }
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete attribute y");
    return -1;
  }
  if (value == Py_None) {
    Py_CLEAR(self->y);
    return 0;
  }

  // Accept either:
  // - dict-like mapping {item_id -> label}, or
  // - 1D array-like (sequence) aligned to item ids 0..len-1.
  const bool is_dict = PyDict_Check(value);

  if (!is_dict) {
    // Require an array-like (sequence), but reject str/bytes which are sequences.
    if (!PySequence_Check(value) || PyUnicode_Check(value) || PyBytes_Check(value)) {
      PyErr_SetString(PyExc_TypeError,
        "y must be a dict {item_id -> label}, a 1D array-like (sequence), or None");
      return -1;
    }
    // If the index currently has items, enforce length equality deterministically.
    if (self->ptr) {
      const int n_items = self->ptr->get_n_items();
      if (n_items > 0) {
        const Py_ssize_t n = PySequence_Size(value);
        if (n < 0) return -1;  // error already set
        if ((int64_t)n != (int64_t)n_items) {
          PyErr_Format(PyExc_ValueError,
            "y must have length %d to match current index size (n_items)", n_items);
          return -1;
        }
      }
    }
  } else {
    // For dict input, validate keys when possible (deterministic safety check).
    if (self->ptr) {
      const int n_items = self->ptr->get_n_items();
      PyObject* key = NULL;
      PyObject* val = NULL;
      Py_ssize_t pos = 0;
      while (PyDict_Next(value, &pos, &key, &val)) {
        if (!PyLong_Check(key)) {
          PyErr_SetString(PyExc_TypeError,
            "y dict keys must be integers (item ids)");
          return -1;
        }
        long long kid = PyLong_AsLongLong(key);
        if (kid == -1 && PyErr_Occurred()) return -1;
        if (kid < 0) {
          PyErr_SetString(PyExc_ValueError,
            "y dict keys must be >= 0");
          return -1;
        }
        if (n_items > 0 && kid >= (long long)n_items) {
          PyErr_Format(PyExc_ValueError,
            "y dict key %lld is out of range for current index size (n_items=%d)",
            kid, n_items);
          return -1;
        }
      }
    }
  }

  Py_INCREF(value);
  Py_XDECREF(self->y);
  self->y = value;
  return 0;
}

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
  if (value == NULL) {
    PyErr_SetString(PyExc_AttributeError,
      "Cannot delete f attribute");
    return -1;
  }
  if (value == Py_None) {
    if (self->ptr) {
      PyErr_SetString(PyExc_AttributeError,
        "Cannot unset f after the index has been created. If not loaded from file to call :meth:`unload`");
      return -1;
    }
    self->f = 0;
    return 0;
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

// Read-only sklearn-style fitted attribute: n_features_in_
//
// Notes
// -----
// sklearn's `check_is_fitted` uses trailing-underscore attributes to decide
// whether an estimator has been fitted.
//
// We intentionally raise AttributeError if `f` is unknown so that
// `hasattr(est, "n_features_in_")` behaves like scikit-learn.
static PyObject* py_annoy_get_n_features_in_(
  py_annoy* self,
  void*) {
  if (!self || self->f <= 0) {
    PyErr_SetString(PyExc_AttributeError,
      "n_features_in_ is not available before `f` is set");
    return NULL;
  }
  return PyLong_FromLong((long)self->f);
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
    // For scikit-learn compatibility, allow structural parameter changes at
    // any time by deterministically resetting the current index.
    if (self->ptr) {
      if (!reset_index_state(
            self,
            "Changing metric resets the existing index (drops all items/trees and y). "
            "Refit is required.")) {
        return -1;
      }
    }
    self->metric_id = METRIC_UNKNOWN;
    return 0;
  }
  if (!PyUnicode_Check(value)) {
    PyErr_SetString(PyExc_TypeError,
      "metric must be a string (or None)");
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

  // If the C++ index exists and the metric is changing, this is a structural
  // change. Reset deterministically to avoid incorrect query results.
  if (self->ptr && self->metric_id != id) {
    if (!reset_index_state(
          self,
          "Changing metric resets the existing index (drops all items/trees and y). "
          "Refit is required.")) {
      return -1;
    }
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
  // Semantics (deterministic):
  //   * None (or deletion) clears the configured on-disk path (if no disk-backed
  //     index is currently active).
  //   * A string / PathLike sets the path and enables on-disk build mode as soon
  //     as the underlying index exists (equivalent to calling on_disk_build(fn)).
  //
  // Safety: Annoy core's on_disk_build() truncates existing files. Therefore, we
  // do not implicitly enable on-disk build if the target path already exists;
  // call on_disk_build(path) explicitly to overwrite.
  if (!value || value == Py_None) {
    if (self->on_disk_active) {
      PyErr_SetString(PyExc_AttributeError,
        "Cannot clear on_disk_path while an on-disk index is active. Call unload() first.");
      return -1;
    }
    self->on_disk_path.clear();
    return 0;
  }

  if (self->on_disk_active) {
    PyErr_SetString(PyExc_AttributeError,
      "Cannot change on_disk_path while an on-disk index is active. Call unload() first.");
    return -1;
  }

  // Accept PathLike objects (pathlib.Path, os.DirEntry, etc.)
  PyObject* fs = PyOS_FSPath(value);  // New reference (or NULL on error)
  if (!fs) return -1;

  std::string new_path;

  if (PyUnicode_Check(fs)) {
    Py_ssize_t n = 0;
    const char* s = PyUnicode_AsUTF8AndSize(fs, &n);
    if (!s) { Py_DECREF(fs); return -1; }
    if (n > 0 && std::memchr(s, '\0', (size_t)n) != NULL) {
      Py_DECREF(fs);
      PyErr_SetString(PyExc_ValueError, "on_disk_path must not contain NUL bytes");
      return -1;
    }
    new_path.assign(s, (size_t)n);
  } else if (PyBytes_Check(fs)) {
    const char* s = PyBytes_AS_STRING(fs);
    const Py_ssize_t n = PyBytes_GET_SIZE(fs);
    if (n > 0 && std::memchr(s, '\0', (size_t)n) != NULL) {
      Py_DECREF(fs);
      PyErr_SetString(PyExc_ValueError, "on_disk_path must not contain NUL bytes");
      return -1;
    }
    new_path.assign(s, (size_t)n);
  } else {
    Py_DECREF(fs);
    PyErr_SetString(PyExc_TypeError, "on_disk_path must be a path-like object or None");
    return -1;
  }

  Py_DECREF(fs);

  // Treat empty path as "clear" (same as None), for deterministic behavior.
  if (new_path.empty()) {
    self->on_disk_path.clear();
    return 0;
  }

  const std::string old_path = self->on_disk_path;
  self->on_disk_path = new_path;

  // If the index already exists, enable on-disk build mode immediately so the
  // backing file is created right away (Annoy core behavior).
  if (self->ptr && !self->on_disk_active) {
    if (path_exists_noexc(self->on_disk_path.c_str())) {
      // Refuse to truncate implicitly. Keep the path for introspection and for
      // an explicit on_disk_build() call by the user.
      if (PyErr_WarnEx(
            PyExc_UserWarning,
            "on_disk_path points to an existing file; refusing to truncate it "
            "implicitly. Call on_disk_build(on_disk_path) explicitly to overwrite.",
            1) < 0) {
        self->on_disk_path = old_path;
        return -1;
    }
      return 0;
    }

    ScopedError error;
    if (!self->ptr->on_disk_build(self->on_disk_path.c_str(), &error.err)) {
      // Roll back attribute value (best-effort).
      self->on_disk_path = old_path;
      PyErr_SetString(PyExc_IOError,
        error.err ? error.err : (char*)"on_disk_build failed");
      return -1;
    }
    self->on_disk_active = true;
  }

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

// ------------------------------------------------------------------
// sklearn-style parameter attributes: seed / random_state / verbose
// ------------------------------------------------------------------
//
// These are exposed as attributes (not only methods) to satisfy scikit-learn's
// estimator API expectation that all __init__ parameters are stored on the
// instance under the same name.
//
// Notes
// -----
// - `seed` and `random_state` are synonyms.
// - `verbose` stores an integer level in [-2, 2] or None (unset).
// - For backward compatibility, the callable setter is available via
//   :meth:`set_verbose` (and :meth:`set_verbosity`), not via ``verbose(...)``.
//
// See Also
// --------
// get_params, set_params : Estimator parameter API.
// set_seed : Callable seed setter with identical semantics.
static PyObject* py_annoy_get_seed(
  py_annoy* self,
  void*) {
  if (!self || !self->has_pending_seed) Py_RETURN_NONE;
  return PyLong_FromUnsignedLongLong(
    (unsigned long long)self->pending_seed);
}

static int py_annoy_set_seed(
  py_annoy* self,
  PyObject* value,
  void*) {
  if (value == NULL) {
    PyErr_SetString(PyExc_AttributeError,
      "Cannot delete seed attribute");
    return -1;
  }

  // None: clear override (core default seed is already deterministic).
  if (value == Py_None) {
    self->pending_seed     = 0ULL;
    self->has_pending_seed = false;
    if (self->ptr) self->ptr->set_seed(0ULL);
    return 0;
  }

  // Must be an int in [0, 2**64-1].
  unsigned long long seed_arg = PyLong_AsUnsignedLongLong(value);
  if (PyErr_Occurred()) {
    if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
      PyErr_Clear();
      PyErr_SetString(PyExc_ValueError,
        "seed must be an integer in the range [0, 2**64 - 1] (or None)");
    }
    return -1;
  }

  // 0: explicit reset + warning.
  if (seed_arg == 0ULL) {
    if (PyErr_WarnEx(PyExc_UserWarning,
      "seed=0 resets to Annoy's default seed", 1) < 0) {
      return -1;
    }
    self->pending_seed     = 0ULL;
    self->has_pending_seed = false;
    if (self->ptr) self->ptr->set_seed(0ULL);
    return 0;
  }

  const uint64_t seed = normalize_seed_u64(static_cast<uint64_t>(seed_arg));
  self->pending_seed     = seed;
  self->has_pending_seed = true;
  if (self->ptr) self->ptr->set_seed(seed);
  return 0;
}

// sklearn alias
static PyObject* py_annoy_get_random_state(
  py_annoy* self,
  void* closure) {
  return py_annoy_get_seed(self, closure);
}

static int py_annoy_set_random_state(
  py_annoy* self,
  PyObject* value,
  void* closure) {
  return py_annoy_set_seed(self, value, closure);
}

static PyObject* py_annoy_get_verbose(
  py_annoy* self,
  void*) {
  if (!self || !self->has_pending_verbose) Py_RETURN_NONE;
  return PyLong_FromLong((long)self->pending_verbose);
}

static int py_annoy_set_verbose(
  py_annoy* self,
  PyObject* value,
  void*) {
  if (value == NULL) {
    PyErr_SetString(PyExc_AttributeError,
      "Cannot delete verbose attribute");
    return -1;
  }
  // None: clear override.
  if (value == Py_None) {
    self->pending_verbose     = 0;
    self->has_pending_verbose = false;
    if (self->ptr) self->ptr->verbose(false);
    return 0;
  }
  if (!PyLong_Check(value)) {
    PyErr_SetString(PyExc_TypeError,
      "verbose must be an integer (or None)");
    return -1;
  }

  long level = PyLong_AsLong(value);
  if (level == -1 && PyErr_Occurred()) return -1;

  // Clamp deterministically to [-2, 2]
  if (level < -2) level = -2;
  if (level >  2) level =  2;

  self->pending_verbose     = static_cast<int>(level);
  self->has_pending_verbose = true;

  if (self->ptr) {
    self->ptr->verbose(level >= 1);
  }
  return 0;
}

// Convenience alias for f (dimension) to better match scikit-learn naming.
static PyObject* py_annoy_get_n_features(
  py_annoy* self,
  void* closure) {
  return py_annoy_get_f(self, closure);
}

static int py_annoy_set_n_features(
  py_annoy* self,
  PyObject* value,
  void* closure) {
  return py_annoy_set_f(self, value, closure);
}

// Read-only alias of n_features_in_ for downstream code that expects a trailing underscore.
static PyObject* py_annoy_get_n_features_(
  py_annoy* self,
  void* closure) {
  return py_annoy_get_n_features_in_(self, closure);
}

// ===================== Get/Set table ======================================

static PyGetSetDef py_annoy_getset[] = {

  {
    (char*)"schema_version",
    (getter)py_annoy_get_schema_version,
    (setter)py_annoy_set_schema_version,
    (char*)kSchemaVersionDoc,
    NULL
  },
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
    // (setter)py_annoy_set_on_disk_path,
    NULL,  // read-only alias of on_disk_path (prevents bypassing validation)
    (char*)"internal: alias of on_disk_path (read-only). Use .on_disk_path to set.",
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
    (char*)"seed",
    (getter)py_annoy_get_seed,
    (setter)py_annoy_set_seed,
    (char*)"Random seed override (scikit-learn compatible). None means use Annoy default seed.",
    NULL
  },

  {
    (char*)"random_state",
    (getter)py_annoy_get_random_state,
    (setter)py_annoy_set_random_state,
    (char*)"Alias of `seed` (scikit-learn convention).",
    NULL
  },

  {
    (char*)"verbose",
    (getter)py_annoy_get_verbose,
    (setter)py_annoy_set_verbose,
    (char*)"Verbosity level in [-2, 2] or None (unset). Callable setter: set_verbose().",
    NULL
  },

  {
    (char*)"n_features",
    (getter)py_annoy_get_n_features,
    (setter)py_annoy_set_n_features,
    (char*)"Alias of `f` (dimension), provided for scikit-learn naming parity.",
    NULL
  },

  {
    (char*)"n_features_",
    (getter)py_annoy_get_n_features_,
    NULL,  // read-only
    (char*)"Read-only alias of `n_features_in_`.",
    NULL
  },

  {
    (char*)"n_features_in_",
    (getter)py_annoy_get_n_features_in_,
    NULL,  // read-only
    (char*)"Number of features seen during fit (scikit-learn compatible). Alias of `f` when available.",
    NULL
  },

  {
    (char*)"y",
    (getter)py_annoy_get_y,
    (setter)py_annoy_set_y,
    (char*)
    "Labels / targets associated with the index items.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "If provided to fit(X, y), labels are stored here after a successful build.\n"
    "You may also set this property manually. When possible, the setter enforces\n"
    "that len(y) matches the current number of items (n_items).\n",
    NULL
  },

  {
    (char*)"_y",
    (getter)py_annoy_get_y,
    // (setter)py_annoy_set_y,
    NULL,  // read-only alias of on_disk_path (prevents bypassing validation)
    (char*)
    "Alias for :attr:`y`.\n",
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
  // Callable setter for the `verbose` parameter.
  //
  // - If level is omitted: default=1 (backwards compatible with the previous verbose(1)).
  // - If level is None: clear the override.
  // - Otherwise: must be an int; clamped to [-2, 2].
  PyObject* level_obj = NULL;
  static const char* kwlist[] = {"level", NULL};

  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|O", (char**)kwlist, &level_obj))
    return NULL;

  // Default: level=1 (historical behavior)
  int level = 1;

  // Clear override
  if (level_obj == Py_None) {
    self->pending_verbose     = 0;
    self->has_pending_verbose = false;
    if (self->ptr) self->ptr->verbose(false);
    PY_RETURN_SELF;
  }

  // Explicit level
  if (level_obj) {
    if (!PyLong_Check(level_obj)) {
      PyErr_SetString(PyExc_TypeError, "level must be an integer or None");
      return NULL;
    }
    long lv = PyLong_AsLong(level_obj);
    if (lv == -1 && PyErr_Occurred()) return NULL;
    level = (int)lv;
  }

  // Clamp deterministically
  if (level < -2) level = -2;
  if (level >  2) level =  2;

  // Always remember the user’s choice
  self->pending_verbose     = level;
  self->has_pending_verbose = true;

  // If index not yet created → just store; will apply when the index is
  // constructed lazily.
  if (!self->ptr) {
    PY_RETURN_SELF; // Py_RETURN_TRUE; // Chaining: a.build(...).save(...).info()
  }

  const bool verbose_flag = (level >= 1);
  self->ptr->verbose(verbose_flag);
  PY_RETURN_SELF; // Py_RETURN_TRUE; // Chaining: a.build(...).save(...).info()
}

// Seed control:
//   set_seed(seed: int | None = None) -> Annoy
//
// Semantics (deterministic):
//   * seed omitted / None -> clear any pending override and reset to Annoy's default seed.
//   * seed == 0           -> same reset, but emit a UserWarning (explicit reset).
//   * seed > 0            -> set an explicit seed.
//
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
  // long seed_arg = static_cast<long>(0ULL); // 0ULL
  // unsigned long long seed_arg = 0ULL;
  PyObject* seed_obj = NULL;
  static const char* kwlist[] = {"seed", NULL};

  // Optional seed argument (int or None).
  //
  // Semantics (deterministic):
  //   * seed omitted / None -> clear any pending override and reset to Annoy's default seed.
  //   * seed == 0           -> same reset, but emit a UserWarning (explicit reset).
  //   * seed > 0            -> set an explicit seed (stored if index not yet constructed).
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|O", (char**)kwlist, &seed_obj)) {
    return NULL;
  }
  // const bool provided =
  //   (PyTuple_GET_SIZE(args) > 0) || (kwargs && PyDict_Size(kwargs) > 0);

  // seed omitted / None: clear override (default is already deterministic).
  if (!seed_obj || seed_obj == Py_None) {
    self->pending_seed     = 0ULL;
    self->has_pending_seed = false;
    if (self->ptr) self->ptr->set_seed(0ULL);  // normalized to core default seed
    PY_RETURN_SELF;
  }

  // Parse integer seed (must be in [0, 2**64 - 1]).
  unsigned long long seed_arg = PyLong_AsUnsignedLongLong(seed_obj);
  if (PyErr_Occurred()) {
    if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
      PyErr_Clear();
      PyErr_SetString(PyExc_ValueError,
        "seed must be an integer in the range [0, 2**64 - 1] (or None)");
    }
    return NULL;  // Py_RETURN_NONE;
  }

  // seed == 0: explicit reset to default + warning.
  if (seed_arg == 0ULL) {
    if (PyErr_WarnEx(PyExc_UserWarning,
      "seed=0 resets to Annoy's default seed", 1) < 0) {
      return NULL;
    }
    self->pending_seed     = 0ULL;
    self->has_pending_seed = false;
    if (self->ptr) self->ptr->set_seed(0ULL);  // normalized to core default seed
    PY_RETURN_SELF;
  }

  const uint64_t seed = normalize_seed_u64(static_cast<uint64_t>(seed_arg));  // seed_arg > 0
  self->pending_seed     = seed;
  self->has_pending_seed = true;

  if (self->ptr) {
    self->ptr->set_seed(seed);  // Apply immediately if constructed
  }

  // Chaining: a.build(...).save(...).info()
  PY_RETURN_SELF;
}

// ------------------------------------------------------------------
// Portable serialization wrapper (sklearn-like compatibility guards)
// ------------------------------------------------------------------
//
// Annoy's native serialize()/deserialize() snapshot the in-memory C++ layout.
// That is fast but *not* portable across different ABIs (endianness, sizeof(size_t),
// compiler struct layout changes, etc.).
//
// For sklearn-like persistence, we wrap the native payload with a small, explicit
// header that:
//   1) declares ABI-critical properties, and
//   2) allows deterministic compatibility checks before deserialization.
//
// This does not magically enable cross-endian loading; instead it makes
// incompatibilities fail *loudly and safely* (instead of corrupting memory).
//
// Header layout (little-endian fields):
//   magic[8]          = "ANNOYSP1"
//   version_u16       = 1
//   endian_u8         = 1 (little) or 2 (big)
//   sizeof_size_t_u8
//   sizeof_S_u8       (Annoy index id type; this wrapper uses int32_t)
//   sizeof_T_u8       (vector scalar type; this wrapper uses float)
//   metric_id_u8
//   reserved_u8       (0)
//   f_u32             (embedding dimension)
//   payload_size_u64  (native payload bytes)
//   payload[payload_size_u64]
//
static const uint8_t ANNOY_PORTABLE_MAGIC[8] = {'A','N','N','O','Y','S','P','1'};
static const uint16_t ANNOY_PORTABLE_VERSION = 1;
static const size_t ANNOY_PORTABLE_HEADER_SIZE = 28;

static inline bool annoy_host_is_little_endian() {
  const uint16_t x = 1;
  return *reinterpret_cast<const uint8_t*>(&x) == 1;
}

static inline void annoy_append_u8(std::vector<uint8_t>& out, uint8_t v) {
  out.push_back(v);
}
static inline void annoy_append_u16_le(std::vector<uint8_t>& out, uint16_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFFu));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
}
static inline void annoy_append_u32_le(std::vector<uint8_t>& out, uint32_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFFu));
  out.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
  out.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
  out.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}
static inline void annoy_append_u64_le(std::vector<uint8_t>& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
  }
}

static inline bool annoy_read_u8(const uint8_t*& p, size_t& n, uint8_t* out) {
  if (n < 1) return false;
  *out = *p;
  p += 1;
  n -= 1;
  return true;
}
static inline bool annoy_read_u16_le(const uint8_t*& p, size_t& n, uint16_t* out) {
  if (n < 2) return false;
  *out = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
  p += 2;
  n -= 2;
  return true;
}
static inline bool annoy_read_u32_le(const uint8_t*& p, size_t& n, uint32_t* out) {
  if (n < 4) return false;
  *out = (static_cast<uint32_t>(p[0]) |
          (static_cast<uint32_t>(p[1]) << 8) |
          (static_cast<uint32_t>(p[2]) << 16) |
          (static_cast<uint32_t>(p[3]) << 24));
  p += 4;
  n -= 4;
  return true;
}
static inline bool annoy_read_u64_le(const uint8_t*& p, size_t& n, uint64_t* out) {
  if (n < 8) return false;
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) {
    v |= (static_cast<uint64_t>(p[i]) << (8 * i));
  }
  *out = v;
  p += 8;
  n -= 8;
  return true;
}

static bool annoy_build_portable_blob(
  const py_annoy* self,
  const std::vector<uint8_t>& native_payload,
  std::vector<uint8_t>* out_blob) {
  if (!self || !out_blob) {
    PyErr_SetString(PyExc_RuntimeError, "internal error: null pointer");
    return false;
  }

  const uint8_t endian = annoy_host_is_little_endian() ? 1 : 2;

  // Hard ABI invariants for this wrapper type.
  const uint8_t sizeof_S = static_cast<uint8_t>(sizeof(int32_t));
  const uint8_t sizeof_T = static_cast<uint8_t>(sizeof(float));
  const uint8_t sizeof_size_t = static_cast<uint8_t>(sizeof(size_t));

  // Deterministic: do not allow silent truncation when casting sizes.
  if (native_payload.size() > static_cast<size_t>(std::numeric_limits<uint64_t>::max())) {
    PyErr_SetString(PyExc_OverflowError, "native payload too large to serialize");
    return false;
  }

  out_blob->clear();
  out_blob->reserve(ANNOY_PORTABLE_HEADER_SIZE + native_payload.size());

  out_blob->insert(out_blob->end(), ANNOY_PORTABLE_MAGIC, ANNOY_PORTABLE_MAGIC + 8);
  annoy_append_u16_le(*out_blob, ANNOY_PORTABLE_VERSION);
  annoy_append_u8(*out_blob, endian);
  annoy_append_u8(*out_blob, sizeof_size_t);
  annoy_append_u8(*out_blob, sizeof_S);
  annoy_append_u8(*out_blob, sizeof_T);
  annoy_append_u8(*out_blob, static_cast<uint8_t>(self->metric_id));
  annoy_append_u8(*out_blob, 0);  // reserved
  annoy_append_u32_le(*out_blob, static_cast<uint32_t>(self->f));
  annoy_append_u64_le(*out_blob, static_cast<uint64_t>(native_payload.size()));

  out_blob->insert(out_blob->end(), native_payload.begin(), native_payload.end());
  return true;
}

// If data begins with the portable header, validate it and return a view of the
// underlying native payload. Otherwise return the original buffer as-is.
static bool annoy_unwrap_portable_blob(
  const py_annoy* self,
  const uint8_t* data,
  size_t size,
  const uint8_t** out_payload,
  size_t* out_payload_size,
  uint32_t* out_f,
  uint8_t* out_metric_id) {
  if (!out_payload || !out_payload_size) {
    PyErr_SetString(PyExc_RuntimeError, "internal error: null output pointer");
    return false;
  }
  *out_payload = data;
  *out_payload_size = size;
  if (out_f) *out_f = 0;
  if (out_metric_id) *out_metric_id = 0;

  if (!data || size < ANNOY_PORTABLE_HEADER_SIZE) {
    return true;  // legacy/native
  }

  if (memcmp(data, ANNOY_PORTABLE_MAGIC, 8) != 0) {
    return true;  // legacy/native
  }

  const uint8_t* p = data + 8;
  size_t n = size - 8;

  uint16_t version = 0;
  uint8_t endian = 0;
  uint8_t sizeof_size_t = 0;
  uint8_t sizeof_S = 0;
  uint8_t sizeof_T = 0;
  uint8_t metric_id = 0;
  uint8_t reserved = 0;
  uint32_t f = 0;
  uint64_t payload_size = 0;

  if (!annoy_read_u16_le(p, n, &version) ||
      !annoy_read_u8(p, n, &endian) ||
      !annoy_read_u8(p, n, &sizeof_size_t) ||
      !annoy_read_u8(p, n, &sizeof_S) ||
      !annoy_read_u8(p, n, &sizeof_T) ||
      !annoy_read_u8(p, n, &metric_id) ||
      !annoy_read_u8(p, n, &reserved) ||
      !annoy_read_u32_le(p, n, &f) ||
      !annoy_read_u64_le(p, n, &payload_size)) {
    PyErr_SetString(PyExc_IOError, "portable header is truncated");
    return false;
  }

  if (out_f) *out_f = f;
  if (out_metric_id) *out_metric_id = metric_id;

  (void)reserved;

  if (version != ANNOY_PORTABLE_VERSION) {
    PyErr_SetString(PyExc_IOError, "unsupported portable blob version");
    return false;
  }

  const uint8_t host_endian = annoy_host_is_little_endian() ? 1 : 2;
  if (endian != host_endian) {
    PyErr_SetString(PyExc_IOError, "cannot deserialize portable blob with different endianness");
    return false;
  }
  if (sizeof_size_t != sizeof(size_t) || sizeof_S != sizeof(int32_t) || sizeof_T != sizeof(float)) {
    PyErr_SetString(PyExc_IOError, "cannot deserialize portable blob with incompatible ABI (sizeof mismatch)");
    return false;
  }

  if (self) {
    if (self->f > 0 && static_cast<uint32_t>(self->f) != f) {
      PyErr_SetString(PyExc_ValueError, "portable blob dimension f does not match this index");
      return false;
    }
    if (self->metric_id != METRIC_UNKNOWN && static_cast<uint8_t>(self->metric_id) != metric_id) {
      PyErr_SetString(PyExc_ValueError, "portable blob metric does not match this index");
      return false;
    }
  }

  // payload_size is declared explicitly; ensure remaining bytes match exactly.
  if (payload_size > static_cast<uint64_t>(n)) {
    PyErr_SetString(PyExc_IOError, "portable blob payload is truncated");
    return false;
  }
  if (payload_size != static_cast<uint64_t>(n)) {
    PyErr_SetString(PyExc_IOError, "portable blob payload size mismatch");
    return false;
  }

  *out_payload = p;
  *out_payload_size = static_cast<size_t>(payload_size);
  return true;
}


// ------------------------------------------------------------------
// Canonical serialization (sklearn-like, cross-ABI)
// ------------------------------------------------------------------
//
// "canonical-v1" is a rebuildable wire format intended to be:
//   * deterministic (byte-for-byte given the same stored vectors + params)
//   * portable across compilers / platforms / ABIs (within IEEE-754 float32)
//   * safe to validate before loading
//
// Design choice (portability-first):
//   - We do NOT store Annoy's in-memory node layout.
//   - Instead we store the item vectors + build parameters, and rebuild the
//     forest on load with n_jobs=1 for deterministic reconstruction.
//
// This mirrors how many sklearn estimators persist as "parameters + learned arrays"
// rather than raw memory snapshots.
//
// Format: little-endian
//   magic[8]          = "ANNOYCN1"
//   version_u16       = 1
//   flags_u16         (bit0 = built)
//   metric_id_u8
//   reserved_u8
//   reserved_u16
//   f_u32
//   n_items_u32
//   n_trees_u32       (valid if built flag set; otherwise 0)
//   has_seed_u8
//   has_verbose_u8
//   reserved_u16
//   seed_u64          (valid if has_seed)
//   verbose_i32       (valid if has_verbose)
//   reserved_u32
//   payload_size_u64  (= n_items * f * sizeof(float))
//   payload[payload_size_u64]  float32 values, row-major by item id
//
static const uint8_t  ANNOY_CANONICAL_MAGIC[8] = {'A','N','N','O','Y','C','N','1'};
static const uint16_t ANNOY_CANONICAL_VERSION  = 1;
static const uint16_t ANNOY_CANONICAL_FLAG_BUILT = 1u << 0;
static const size_t   ANNOY_CANONICAL_HEADER_SIZE = 56;

static inline void annoy_append_i32_le(std::vector<uint8_t>& out, int32_t v) {
  annoy_append_u32_le(out, static_cast<uint32_t>(v));
}
static inline bool annoy_read_i32_le(const uint8_t*& p, size_t& n, int32_t* out) {
  uint32_t u = 0;
  if (!annoy_read_u32_le(p, n, &u)) return false;
  *out = static_cast<int32_t>(u);
  return true;
}

static inline bool annoy_is_canonical_blob(const uint8_t* data, size_t size) {
  return (data && size >= ANNOY_CANONICAL_HEADER_SIZE &&
          memcmp(data, ANNOY_CANONICAL_MAGIC, 8) == 0);
}

static bool annoy_build_canonical_blob(
  const py_annoy* self,
  std::vector<uint8_t>* out_blob) {
  if (!self || !out_blob) {
    PyErr_SetString(PyExc_RuntimeError, "internal error: null pointer");
    return false;
  }
  if (!self->ptr) {
    PyErr_SetString(PyExc_RuntimeError, "Annoy index is not initialized");
    return false;
  }
  if (self->f <= 0) {
    PyErr_SetString(PyExc_RuntimeError, "Index dimension f is not set");
    return false;
  }
  if (self->metric_id == METRIC_UNKNOWN) {
    PyErr_SetString(PyExc_RuntimeError, "Index metric is not set");
    return false;
  }

  // Canonical format requires IEEE-754 float32.
  if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559) {
    PyErr_SetString(PyExc_RuntimeError,
      "canonical serialization requires IEEE-754 32-bit float");
    return false;
  }

  const int32_t n_items_i32 = self->ptr->get_n_items();
  if (n_items_i32 < 0) {
    PyErr_SetString(PyExc_RuntimeError, "invalid n_items");
    return false;
  }

  const uint32_t n_items = static_cast<uint32_t>(n_items_i32);
  const uint32_t f = static_cast<uint32_t>(self->f);
  const uint32_t n_trees = static_cast<uint32_t>(self->ptr->get_n_trees());
  const bool built = (n_trees > 0);

  // Compute payload size with overflow checks.
  uint64_t payload_bytes = 0;
  if (n_items != 0 && f != 0) {
    const uint64_t mul = static_cast<uint64_t>(n_items) * static_cast<uint64_t>(f);
    if (mul > (std::numeric_limits<uint64_t>::max() / 4ULL)) {
      PyErr_SetString(PyExc_OverflowError, "canonical payload too large to serialize");
      return false;
    }
    payload_bytes = mul * 4ULL;
  }

  const uint64_t total_u64 = static_cast<uint64_t>(ANNOY_CANONICAL_HEADER_SIZE) + payload_bytes;
  if (total_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    PyErr_SetString(PyExc_OverflowError, "canonical blob too large to allocate");
    return false;
  }

  out_blob->clear();
  out_blob->reserve(static_cast<size_t>(total_u64));

  // Header
  out_blob->insert(out_blob->end(), ANNOY_CANONICAL_MAGIC, ANNOY_CANONICAL_MAGIC + 8);
  annoy_append_u16_le(*out_blob, ANNOY_CANONICAL_VERSION);
  annoy_append_u16_le(*out_blob, built ? ANNOY_CANONICAL_FLAG_BUILT : 0u);
  annoy_append_u8(*out_blob, static_cast<uint8_t>(self->metric_id));
  annoy_append_u8(*out_blob, 0);  // reserved_u8
  annoy_append_u16_le(*out_blob, 0);  // reserved_u16
  annoy_append_u32_le(*out_blob, f);
  annoy_append_u32_le(*out_blob, n_items);
  annoy_append_u32_le(*out_blob, built ? n_trees : 0u);

  annoy_append_u8(*out_blob, self->has_pending_seed ? 1 : 0);
  annoy_append_u8(*out_blob, self->has_pending_verbose ? 1 : 0);
  annoy_append_u16_le(*out_blob, 0);  // reserved_u16

  annoy_append_u64_le(*out_blob, static_cast<uint64_t>(self->pending_seed));
  annoy_append_i32_le(*out_blob, static_cast<int32_t>(self->pending_verbose));
  annoy_append_u32_le(*out_blob, 0);  // reserved_u32
  annoy_append_u64_le(*out_blob, payload_bytes);

  // Payload: row-major vectors (item id order)
  std::vector<float> vec(static_cast<size_t>(f));
  for (uint32_t i = 0; i < n_items; ++i) {
    self->ptr->get_item(static_cast<int32_t>(i), vec.data());
    for (uint32_t j = 0; j < f; ++j) {
      uint32_t bits = 0;
      static_assert(sizeof(bits) == sizeof(float), "float must be 32-bit");
      memcpy(&bits, &vec[static_cast<size_t>(j)], sizeof(float));
      annoy_append_u32_le(*out_blob, bits);
    }
  }

  return true;
}

static bool annoy_parse_canonical_blob(
  const py_annoy* self,
  const uint8_t* data,
  size_t size,
  uint16_t* out_flags,
  uint8_t* out_metric_id,
  uint32_t* out_f,
  uint32_t* out_n_items,
  uint32_t* out_n_trees,
  uint8_t* out_has_seed,
  uint8_t* out_has_verbose,
  uint64_t* out_seed,
  int32_t* out_verbose,
  const uint8_t** out_payload,
  size_t* out_payload_size) {
  if (!data || size < ANNOY_CANONICAL_HEADER_SIZE) {
    PyErr_SetString(PyExc_IOError, "canonical blob is truncated");
    return false;
  }
  if (memcmp(data, ANNOY_CANONICAL_MAGIC, 8) != 0) {
    PyErr_SetString(PyExc_IOError, "canonical blob magic mismatch");
    return false;
  }

  const uint8_t* p = data + 8;
  size_t n = size - 8;

  uint16_t version = 0;
  uint16_t flags = 0;
  uint8_t metric_id = 0;
  uint8_t reserved_u8 = 0;
  uint16_t reserved_u16a = 0;
  uint32_t f = 0;
  uint32_t n_items = 0;
  uint32_t n_trees = 0;
  uint8_t has_seed = 0;
  uint8_t has_verbose = 0;
  uint16_t reserved_u16b = 0;
  uint64_t seed = 0;
  int32_t verbose = 0;
  uint32_t reserved_u32 = 0;
  uint64_t payload_size_u64 = 0;

  if (!annoy_read_u16_le(p, n, &version) ||
      !annoy_read_u16_le(p, n, &flags) ||
      !annoy_read_u8(p, n, &metric_id) ||
      !annoy_read_u8(p, n, &reserved_u8) ||
      !annoy_read_u16_le(p, n, &reserved_u16a) ||
      !annoy_read_u32_le(p, n, &f) ||
      !annoy_read_u32_le(p, n, &n_items) ||
      !annoy_read_u32_le(p, n, &n_trees) ||
      !annoy_read_u8(p, n, &has_seed) ||
      !annoy_read_u8(p, n, &has_verbose) ||
      !annoy_read_u16_le(p, n, &reserved_u16b) ||
      !annoy_read_u64_le(p, n, &seed) ||
      !annoy_read_i32_le(p, n, &verbose) ||
      !annoy_read_u32_le(p, n, &reserved_u32) ||
      !annoy_read_u64_le(p, n, &payload_size_u64)) {
    PyErr_SetString(PyExc_IOError, "canonical header is truncated");
    return false;
  }

  (void)reserved_u8;
  (void)reserved_u16a;
  (void)reserved_u16b;
  (void)reserved_u32;

  if (version != ANNOY_CANONICAL_VERSION) {
    PyErr_SetString(PyExc_IOError, "unsupported canonical blob version");
    return false;
  }

  // Validate metric + f against the current object if known.
  if (self) {
    if (self->f > 0 && static_cast<uint32_t>(self->f) != f) {
      PyErr_SetString(PyExc_ValueError, "canonical blob dimension f does not match this index");
      return false;
    }
    if (self->metric_id != METRIC_UNKNOWN && static_cast<uint8_t>(self->metric_id) != metric_id) {
      PyErr_SetString(PyExc_ValueError, "canonical blob metric does not match this index");
      return false;
    }
  }

  // Validate payload length matches declared sizes.
  const uint64_t expected = static_cast<uint64_t>(n_items) * static_cast<uint64_t>(f) * 4ULL;
  if (expected != payload_size_u64) {
    PyErr_SetString(PyExc_IOError, "canonical payload size mismatch");
    return false;
  }
  if (payload_size_u64 > static_cast<uint64_t>(n)) {
    PyErr_SetString(PyExc_IOError, "canonical payload is truncated");
    return false;
  }
  if (payload_size_u64 != static_cast<uint64_t>(n)) {
    PyErr_SetString(PyExc_IOError, "canonical blob has trailing bytes");
    return false;
  }

  if (out_flags) *out_flags = flags;
  if (out_metric_id) *out_metric_id = metric_id;
  if (out_f) *out_f = f;
  if (out_n_items) *out_n_items = n_items;
  if (out_n_trees) *out_n_trees = n_trees;
  if (out_has_seed) *out_has_seed = has_seed;
  if (out_has_verbose) *out_has_verbose = has_verbose;
  if (out_seed) *out_seed = seed;
  if (out_verbose) *out_verbose = verbose;

  if (out_payload) *out_payload = p;
  if (out_payload_size) *out_payload_size = static_cast<size_t>(payload_size_u64);
  return true;
}

static bool annoy_restore_from_canonical_blob(
  py_annoy* self,
  const uint8_t* data,
  size_t size) {
  if (!self) {
    PyErr_SetString(PyExc_RuntimeError, "internal error: null self");
    return false;
  }

  // Canonical format requires IEEE-754 float32.
  if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559) {
    PyErr_SetString(PyExc_RuntimeError,
      "canonical deserialization requires IEEE-754 32-bit float");
    return false;
  }

  uint16_t flags = 0;
  uint8_t metric_id_u8 = 0;
  uint32_t f_u32 = 0;
  uint32_t n_items = 0;
  uint32_t n_trees = 0;
  uint8_t has_seed = 0;
  uint8_t has_verbose = 0;
  uint64_t seed = 0;
  int32_t verbose = 0;
  const uint8_t* payload = NULL;
  size_t payload_size = 0;

  if (!annoy_parse_canonical_blob(
        self, data, size,
        &flags, &metric_id_u8, &f_u32, &n_items, &n_trees,
        &has_seed, &has_verbose, &seed, &verbose,
        &payload, &payload_size)) {
    return false;  // Python error already set
  }

  // Replace current index state (deterministic).
  if (self->ptr) {
    delete self->ptr;
    self->ptr = NULL;
  }

  // Set core configuration from the blob if not already fixed.
  if (self->f <= 0) self->f = static_cast<int>(f_u32);
  if (self->metric_id == METRIC_UNKNOWN) self->metric_id = static_cast<MetricId>(metric_id_u8);

  // Pending seed/verbose in state; applied during ensure_index().
  self->has_pending_seed = (has_seed != 0);
  self->pending_seed = normalize_seed_u64(static_cast<uint64_t>(seed));
  self->has_pending_verbose = (has_verbose != 0);
  self->pending_verbose = static_cast<int>(verbose);

  // Construct index (applies pending config).
  if (!ensure_index(self)) {
    return false;
  }

  // Load vectors (heavy work): add items with GIL released.
  bool ok = true;
  ScopedError error;
  std::vector<float> vec(static_cast<size_t>(f_u32));
  const uint8_t* p = payload;
  size_t n = payload_size;

  Py_BEGIN_ALLOW_THREADS;
  for (uint32_t i = 0; i < n_items && ok; ++i) {
    for (uint32_t j = 0; j < f_u32; ++j) {
      if (n < 4) { ok = false; break; }
      uint32_t bits = (static_cast<uint32_t>(p[0]) |
                       (static_cast<uint32_t>(p[1]) << 8) |
                       (static_cast<uint32_t>(p[2]) << 16) |
                       (static_cast<uint32_t>(p[3]) << 24));
      p += 4;
      n -= 4;
      float fv = 0.0f;
      memcpy(&fv, &bits, sizeof(float));
      vec[static_cast<size_t>(j)] = fv;
    }
    if (!ok) break;
    if (!self->ptr->add_item(static_cast<int32_t>(i), vec.data(), &error.err)) {
      ok = false;
      break;
    }
  }
  Py_END_ALLOW_THREADS;

  if (!ok) {
    PyErr_SetString(PyExc_RuntimeError, error.err ? error.err : (char*)"canonical add_item failed");
    delete self->ptr;
    self->ptr = NULL;
    return false;
  }

  // Rebuild if the blob was built.
  if ((flags & ANNOY_CANONICAL_FLAG_BUILT) != 0 && n_trees > 0) {
    bool built_ok = false;
    ScopedError berr;
    Py_BEGIN_ALLOW_THREADS;
    built_ok = self->ptr->build(static_cast<int>(n_trees), /*n_jobs=*/1, &berr.err);
    Py_END_ALLOW_THREADS;
    if (!built_ok) {
      PyErr_SetString(PyExc_RuntimeError, berr.err ? berr.err : (char*)"canonical rebuild failed");
      delete self->ptr;
      self->ptr = NULL;
      return false;
    }
  }

  return true;
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
  // (void)args; (void)kwargs; // if not used
  // if (!self->ptr) {
  //   PyErr_SetString(PyExc_RuntimeError,
  //     "Annoy index is not initialized");
  //   return NULL;
  // }

  // Backwards compatible API:
  //   serialize() -> native bytes (legacy)
  //   serialize(format='portable') -> bytes with a small compatibility header
  PyObject* format_obj = NULL;
  static const char* kwlist[] = {"format", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", (char**)kwlist, &format_obj)) {
    return NULL;
  }

  // Note: canonical blobs carry full configuration (metric, f, items) and can
  // initialize an otherwise-lazy index. We therefore defer the "initialized"
  // check until after canonical auto-detection.

  bool portable = false;
  bool canonical = false;
  if (format_obj && format_obj != Py_None) {
    if (!PyUnicode_Check(format_obj)) {
      PyErr_SetString(PyExc_TypeError, "format must be a str or None");
      return NULL;
    }
    const char* fmt = PyUnicode_AsUTF8(format_obj);
    if (!fmt) return NULL;

    std::string s = trim(std::string(fmt));
    std::transform(s.begin(), s.end(), s.begin(),
      [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    if (s == "portable" || s == "portable_v1" || s == "portable-v1") {
      portable = true;
    } else if (s == "canonical" || s == "canonical_v1" || s == "canonical-v1") {
      canonical = true;
    } else if (s == "native" || s == "legacy" || s == "raw") {
      portable = false;
      canonical = false;
    } else {
      PyErr_SetString(PyExc_ValueError,
        "format must be one of: 'native', 'legacy', 'raw', 'portable', 'canonical'");
      return NULL;
    }
  }

  std::vector<uint8_t> out;

  // Canonical format is rebuildable and portable across ABIs. It avoids
  // Annoy's native in-memory snapshot and instead stores item vectors + params.
  if (canonical) {
    if (!annoy_build_canonical_blob(self, &out)) {
      return NULL;  // Python error already set
    }
  } else {
    if (!self->ptr) {
      PyErr_SetString(PyExc_RuntimeError,
        "Annoy index is not initialized. Call add_item() + build() (or load()) before serialize().");
      return NULL;
    }
    ScopedError error;
    std::vector<uint8_t> native = self->ptr->serialize(&error.err);
    if (native.empty() && error.err) {
      PyErr_SetString(PyExc_RuntimeError, error.err);
      return NULL;
    }

    if (portable) {
      if (!annoy_build_portable_blob(self, native, &out)) {
        return NULL;  // Python error already set
    }
    } else {
      out.swap(native);
    }
  }

  if (out.size() > static_cast<size_t>(PY_SSIZE_T_MAX)) {
    PyErr_SetString(PyExc_OverflowError, "serialized payload too large for Python bytes");
    return NULL;
  }

  return PyBytes_FromStringAndSize(
    reinterpret_cast<const char*>(out.data()),
    static_cast<Py_ssize_t>(out.size()));
}

static PyObject* py_an_deserialize(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  PyObject* byte_object = NULL;
  PyObject* prefault_obj = NULL;  // optional; None => use self.prefault

  // Forward compatible: the implementation auto-detects canonical/portable/native.
  static const char* kwlist[] = {"byte", "prefault", NULL};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "S|O", (char**)kwlist, &byte_object, &prefault_obj)) {
    return NULL;
  }

  // Defensive: "S" already enforces bytes in CPython, but keep an explicit check.
  if (!PyBytes_Check(byte_object)) {
    PyErr_SetString(PyExc_TypeError, "Expected `byte` to be bytes");
    return NULL;
  }

  // Deterministic prefault resolution:
  // - if prefault is provided and not None -> use it
  // - else -> use stored self.prefault
  bool prefault = false;
  if (!resolve_prefault_arg(self, prefault_obj, &prefault)) {
    return NULL;
  }

  char* raw = NULL;
  Py_ssize_t length = 0;
  if (PyBytes_AsStringAndSize(byte_object, &raw, &length) < 0) {
    return NULL;
  }
  if (length < 0) {
    PyErr_SetString(PyExc_ValueError, "invalid bytes length");
    return NULL;
  }

  const uint8_t* data = reinterpret_cast<const uint8_t*>(raw);
  const size_t size = static_cast<size_t>(length);

  // 1) Canonical blobs can fully restore by rebuilding, and can initialize a lazy object.
  if (annoy_is_canonical_blob(data, size)) {
    if (!annoy_restore_from_canonical_blob(self, data, size)) {
      return NULL;
    }
    PY_RETURN_SELF;
  }

  // 2) Portable blobs are ABI-guarded and carry f/metric in the header, so they can
  //    initialize a lazy object as well.
  const uint8_t* native_payload = data;
  size_t native_payload_size = size;

  const bool looks_portable = (data && size >= ANNOY_PORTABLE_HEADER_SIZE &&
                               memcmp(data, ANNOY_PORTABLE_MAGIC, 8) == 0);

  if (looks_portable) {
    uint32_t hdr_f = 0;
    uint8_t hdr_metric = 0;

    // First pass: validate ABI + extract header fields without requiring an initialized index.
    if (!annoy_unwrap_portable_blob(NULL, data, size,
                                  &native_payload, &native_payload_size,
                                  &hdr_f, &hdr_metric)) {
      return NULL;
    }

    if (hdr_f == 0) {
      PyErr_SetString(PyExc_IOError, "portable blob has invalid dimension f=0");
      return NULL;
    }
    if (hdr_metric < (uint8_t)METRIC_UNKNOWN || hdr_metric > (uint8_t)METRIC_HAMMING) {
      PyErr_SetString(PyExc_IOError, "portable blob has invalid metric id");
      return NULL;
    }

    // Initialize lazily if needed (sklearn-like: deserialize can construct the estimator).
    if (!self->ptr) {
      if (self->f <= 0) self->f = static_cast<int>(hdr_f);
      if (self->metric_id == METRIC_UNKNOWN) self->metric_id = static_cast<MetricId>(hdr_metric);
      if (!ensure_index(self)) {
        return NULL;
    }
    }

    // Second pass: validate header against this instance (f/metric match) and re-extract payload view.
    if (!annoy_unwrap_portable_blob(self, data, size,
                                  &native_payload, &native_payload_size,
                                  NULL, NULL)) {
      return NULL;
    }
  } else {
    // 3) Native blobs are raw in-memory snapshots; they require an initialized index.
    if (!self->ptr) {
      PyErr_SetString(PyExc_RuntimeError,
        "Annoy index is not initialized. Native blobs require an initialized index; "
        "construct with f/metric or use serialize(format='portable'|'canonical').");
      return NULL;
    }
  }

  std::vector<uint8_t> v(native_payload, native_payload + native_payload_size);

  ScopedError error;
  if (!self->ptr->deserialize(&v, prefault, &error.err)) {
    PyErr_SetString(PyExc_IOError, error.err ? error.err : (char*)"deserialize failed");
    return NULL;
  }

  PY_RETURN_SELF;
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

// Deterministically drop all index state (items/trees) while keeping the Python
// wrapper object and estimator-style parameters.
//
// IMPORTANT
// ---------
// This helper must only be called while holding the GIL (it may emit warnings
// and clears Python-owned metadata like y).
static bool reset_index_state(
  py_annoy* self,
  const char* warn_msg) {
  if (!self) return true;
  if (!self->ptr) {
    // Even without a live C++ index, clear any stale metadata to keep state
    // consistent.
    self->on_disk_active = false;
    self->on_disk_path.clear();
    Py_CLEAR(self->y);
    return true;
  }

  if (warn_msg) {
    if (PyErr_WarnEx(PyExc_UserWarning, warn_msg, 1) < 0) {
      return false;
    }
  }

  // Best-effort resource release: unload mmap/on-disk state before deleting.
  try { self->ptr->unload(); } catch (...) {}
  delete self->ptr;
  self->ptr = NULL;

  // After a reset the index is no longer backed by any on-disk file.
  self->on_disk_active = false;
  self->on_disk_path.clear();

  // Prevent stale label metadata from being queried against a new index.
  Py_CLEAR(self->y);

  return true;
}

// sklearn protocol hook (optional): `__sklearn_is_fitted__`
//
// This allows scikit-learn utilities to query fitted state without relying on
// heuristics.
static PyObject* py_an_sklearn_is_fitted(
  py_annoy* self,
  PyObject* Py_UNUSED(ignored)) {
  return PyBool_FromLong(is_index_built(self) ? 1 : 0);
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

// ------------------------------------------------------------------
// sklearn-like parameter API (get_params / set_params)
// ------------------------------------------------------------------
//
// These methods make AnnoyIndex behave more like a scikit-learn estimator,
// enabling integration with tools that expect the Estimator API
// (e.g., cloning, parameter grids, pipelines).
//
// We intentionally expose only stable, user-facing parameters.
// Fitted state (built trees, items) remains in the index data itself.
//
// Notes
// -----
// - get_params(deep=...) ignores `deep` because Annoy has no nested estimators.
// - set_params(...) is strict: unknown keys raise ValueError (sklearn behaviour).
//
static PyObject* py_an_get_params(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  PyObject* deep_obj = Py_True;
  static const char* kwlist[] = {"deep", NULL};

  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|O", (char**)kwlist, &deep_obj)) {
    return NULL;
  }

  if (deep_obj != Py_True && deep_obj != Py_False) {
    PyErr_SetString(PyExc_TypeError, "deep must be a bool");
    return NULL;
  }
  // `deep` is accepted for scikit-learn API compatibility.
  // This estimator does not contain nested estimators, so deep has no effect.

  PyObject* d = PyDict_New();
  if (!d) return NULL;

  // f
  PyObject* v = PyLong_FromLong((long)self->f);
  if (!v || PyDict_SetItemString(d, "f", v) < 0) { Py_XDECREF(v); Py_DECREF(d); return NULL; }
  Py_DECREF(v);

  // metric
  const char* metric_c = metric_to_cstr(self->metric_id);
  // sklearn compatibility: unknown / lazy metric should round-trip as None
  // (not an empty string).
  if (metric_c) {
    v = PyUnicode_FromString(metric_c);
  } else {
    Py_INCREF(Py_None);
    v = Py_None;
  }
  if (!v || PyDict_SetItemString(d, "metric", v) < 0) { Py_XDECREF(v); Py_DECREF(d); return NULL; }
  Py_DECREF(v);

  // seed / verbose (None if not explicitly set)
  if (self->has_pending_seed) {
    v = PyLong_FromUnsignedLongLong((unsigned long long)self->pending_seed);
  } else {
    Py_INCREF(Py_None);
    v = Py_None;
  }
  if (!v || PyDict_SetItemString(d, "seed", v) < 0) { Py_XDECREF(v); Py_DECREF(d); return NULL; }
  Py_DECREF(v);

  if (self->has_pending_verbose) {
    v = PyLong_FromLong((long)self->pending_verbose);
  } else {
    Py_INCREF(Py_None);
    v = Py_None;
  }
  if (!v || PyDict_SetItemString(d, "verbose", v) < 0) { Py_XDECREF(v); Py_DECREF(d); return NULL; }
  Py_DECREF(v);

  // prefault / schema_version
  v = PyBool_FromLong(self->prefault ? 1 : 0);
  if (!v || PyDict_SetItemString(d, "prefault", v) < 0) { Py_XDECREF(v); Py_DECREF(d); return NULL; }
  Py_DECREF(v);

  v = PyLong_FromLong((long)self->schema_version);
  if (!v || PyDict_SetItemString(d, "schema_version", v) < 0) { Py_XDECREF(v); Py_DECREF(d); return NULL; }
  Py_DECREF(v);

  // on_disk_path (optional metadata)
  if (!self->on_disk_path.empty()) {
    v = PyUnicode_FromString(self->on_disk_path.c_str());
  } else {
    Py_INCREF(Py_None);
    v = Py_None;
  }
  if (!v || PyDict_SetItemString(d, "on_disk_path", v) < 0) { Py_XDECREF(v); Py_DECREF(d); return NULL; }
  Py_DECREF(v);

  return d;
}

static PyObject* py_an_set_params(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  // scikit-learn allows calling set_params() with no args.
  if (args && PyTuple_GET_SIZE(args) != 0) {
    PyErr_SetString(PyExc_TypeError, "set_params() takes only keyword arguments");
    return NULL;
  }
  if (!kwargs || PyDict_Size(kwargs) == 0) {
    PY_RETURN_SELF;
  }

  // Deterministic alias handling: do not accept both at once.
  if (PyDict_GetItemString(kwargs, "seed") &&
      PyDict_GetItemString(kwargs, "random_state")) {
    PyErr_SetString(PyExc_ValueError,
      "Cannot set both 'seed' and 'random_state' (they are aliases)");
    return NULL;
  }

  PyObject* key = NULL;
  PyObject* value = NULL;
  Py_ssize_t pos = 0;
  while (PyDict_Next(kwargs, &pos, &key, &value)) {
    if (!PyUnicode_Check(key)) {
      PyErr_SetString(PyExc_TypeError, "Parameter names must be strings");
      return NULL;
    }
    const char* k = PyUnicode_AsUTF8(key);
    if (!k) return NULL;

    if (std::strcmp(k, "f") == 0) {
      if (py_annoy_set_f(self, value, NULL) != 0) return NULL;
    } else if (std::strcmp(k, "metric") == 0) {
      if (py_annoy_set_metric(self, value, NULL) != 0) return NULL;
    } else if (std::strcmp(k, "seed") == 0 || std::strcmp(k, "random_state") == 0) {
      if (py_annoy_set_seed(self, value ? value : Py_None, NULL) != 0) return NULL;
    } else if (std::strcmp(k, "verbose") == 0) {
      if (py_annoy_set_verbose(self, value ? value : Py_None, NULL) != 0) return NULL;
    } else if (std::strcmp(k, "prefault") == 0) {
      if (py_annoy_set_prefault(self, value, NULL) != 0) return NULL;
    } else if (std::strcmp(k, "schema_version") == 0) {
      if (py_annoy_set_schema_version(self, value, NULL) != 0) return NULL;
    } else if (std::strcmp(k, "on_disk_path") == 0) {
      if (py_annoy_set_on_disk_path(self, value ? value : Py_None, NULL) != 0) return NULL;
    } else {
      PyErr_Format(PyExc_ValueError,
        "Invalid parameter %R for Annoy. Valid parameters are: "
        "f, metric, seed, random_state, verbose, prefault, schema_version, on_disk_path.",
        key);
      return NULL;
    }
  }

  PY_RETURN_SELF;
}

// sklearn protocol hook: __sklearn_tags__
//
// Starting with scikit-learn 1.6, estimator tags are returned via a `Tags`
// object (not a plain dict). We return conservative defaults:
// - estimator_type: None (not a classifier/regressor/transformer)
// - target_tags.required: False (y is not required)
//
// See Also
// --------
// sklearn.utils.get_tags : Consumes __sklearn_tags__ when available.
static PyObject* py_an_sklearn_tags(
  py_annoy* self,
  PyObject* Py_UNUSED(ignored)) {
  (void)self;

  PyObject* utils_mod = PyImport_ImportModule("sklearn.utils");
  if (!utils_mod) return NULL;

  PyObject* Tags = PyObject_GetAttrString(utils_mod, "Tags");
  PyObject* TargetTags = PyObject_GetAttrString(utils_mod, "TargetTags");
  Py_DECREF(utils_mod);

  if (!Tags || !TargetTags) {
    Py_XDECREF(Tags);
    Py_XDECREF(TargetTags);
    return NULL;
  }

  // TargetTags signature (scikit-learn >= 1.6):
  //   TargetTags(required: bool, one_d_labels: bool = False, ...)
  // Older builds may accept TargetTags() without arguments.
  // Deterministic policy for this estimator: y is NOT required.
  PyObject* target = PyObject_CallObject(TargetTags, NULL);
  if (!target && PyErr_ExceptionMatches(PyExc_TypeError)) {
    PyErr_Clear();

    PyObject* tt_kwargs = PyDict_New();
    if (!tt_kwargs) {
      Py_DECREF(TargetTags);
      Py_DECREF(Tags);
      return NULL;
    }
    if (PyDict_SetItemString(tt_kwargs, "required", Py_False) < 0) {
      Py_DECREF(tt_kwargs);
      Py_DECREF(TargetTags);
      Py_DECREF(Tags);
      return NULL;
    }

    PyObject* tt_args = PyTuple_New(0);
    if (!tt_args) {
      Py_DECREF(tt_kwargs);
      Py_DECREF(TargetTags);
      Py_DECREF(Tags);
      return NULL;
    }

    target = PyObject_Call(TargetTags, tt_args, tt_kwargs);
    Py_DECREF(tt_args);
    Py_DECREF(tt_kwargs);
  }

  Py_DECREF(TargetTags);
  if (!target) {
    Py_DECREF(Tags);
    return NULL;
  }

  // kwargs: Tags(estimator_type=None, target_tags=TargetTags())
  PyObject* kwargs = PyDict_New();
  if (!kwargs) {
    Py_DECREF(target);
    Py_DECREF(Tags);
    return NULL;
  }
  if (PyDict_SetItemString(kwargs, "estimator_type", Py_None) < 0 ||
      PyDict_SetItemString(kwargs, "target_tags", target) < 0) {
    Py_DECREF(kwargs);
    Py_DECREF(target);
    Py_DECREF(Tags);
    return NULL;
  }
  Py_DECREF(target);

  PyObject* args = PyTuple_New(0);
  if (!args) {
    Py_DECREF(kwargs);
    Py_DECREF(Tags);
    return NULL;
  }

  PyObject* tags = PyObject_Call(Tags, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  Py_DECREF(Tags);
  return tags;
}

// sklearn protocol hook: __sklearn_clone__
//
// scikit-learn's `clone` delegates to this hook (since v1.3) if present.
// We implement the canonical behavior: construct a new unfitted object of the
// same class with the same parameters.
//
// See Also
// --------
// sklearn.base.clone : Delegates to __sklearn_clone__ when available.
static PyObject* py_an_sklearn_clone(
  py_annoy* self,
  PyObject* Py_UNUSED(ignored)) {
  // params = self.get_params(deep=False)
  PyObject* kwargs = PyDict_New();
  if (!kwargs) return NULL;
  if (PyDict_SetItemString(kwargs, "deep", Py_False) < 0) {
    Py_DECREF(kwargs);
    return NULL;
  }
  PyObject* empty_args = PyTuple_New(0);
  if (!empty_args) {
    Py_DECREF(kwargs);
    return NULL;
  }
  PyObject* params = py_an_get_params(self, empty_args, kwargs);
  Py_DECREF(empty_args);
  Py_DECREF(kwargs);
  if (!params) return NULL;

  // Critical safety: avoid implicit on-disk side effects during cloning.
  // `on_disk_path` can enable on-disk build mode as soon as the C++ index
  // exists (via ensure_index()), potentially creating/truncating files.
  // Cloning must be side-effect free.
  if (PyDict_DelItemString(params, "on_disk_path") != 0) {
    // Key is always present in get_params(), but be robust to future changes.
    if (PyErr_ExceptionMatches(PyExc_KeyError)) {
      PyErr_Clear();
    } else {
      Py_DECREF(params);
      return NULL;
    }
  }

  // new = type(self)(**params)
  PyObject* cls = (PyObject*)Py_TYPE(self);
  PyObject* ctor_args = PyTuple_New(0);
  if (!ctor_args) {
    Py_DECREF(params);
    return NULL;
  }
  PyObject* out = PyObject_Call(cls, ctor_args, params);
  Py_DECREF(ctor_args);
  Py_DECREF(params);
  return out;
}

// ---------------------------------------------------------------------
// rebuild: deterministic refit-like constructor for metric/on-disk changes
// ---------------------------------------------------------------------
//
// This is an explicit, side-effect-conscious helper that rebuilds a new Annoy
// instance from the *current* index contents.
//
// Why?
// - Annoy's C++ index type is metric-specific (different concrete class).
// - Changing metric (or switching to/from on-disk build) therefore requires a
//   rebuild.
//
// Safety rules (deterministic):
// - The source object's `on_disk_path` is NEVER carried over implicitly.
// - If `on_disk_path` is provided and string-equal to the source's configured
//   path, it is ignored to avoid accidental overwrite/truncation hazards.
//
// Signature:
//   rebuild(metric=None, *, on_disk_path=None, n_trees=None, n_jobs=-1) -> Annoy
//
static bool parse_pathlike_to_string_noexc(
  PyObject* value,
  std::string* out_path) {
  if (!out_path) return false;
  out_path->clear();
  if (!value || value == Py_None) {
    return true;
  }

  PyObject* fs = PyOS_FSPath(value);  // new ref
  if (!fs) return false;

  if (PyUnicode_Check(fs)) {
    Py_ssize_t n = 0;
    const char* s = PyUnicode_AsUTF8AndSize(fs, &n);
    if (!s) { Py_DECREF(fs); return false; }
    if (n > 0 && std::memchr(s, '\0', (size_t)n) != NULL) {
      Py_DECREF(fs);
      PyErr_SetString(PyExc_ValueError, "on_disk_path must not contain NUL bytes");
      return false;
    }
    out_path->assign(s, (size_t)n);
    Py_DECREF(fs);
    return true;
  }

  if (PyBytes_Check(fs)) {
    const char* s = PyBytes_AS_STRING(fs);
    const Py_ssize_t n = PyBytes_GET_SIZE(fs);
    if (n > 0 && std::memchr(s, '\0', (size_t)n) != NULL) {
      Py_DECREF(fs);
      PyErr_SetString(PyExc_ValueError, "on_disk_path must not contain NUL bytes");
      return false;
    }
    out_path->assign(s, (size_t)n);
    Py_DECREF(fs);
    return true;
  }

  Py_DECREF(fs);
  PyErr_SetString(PyExc_TypeError, "on_disk_path must be a path-like object or None");
  return false;
}

static PyObject* py_an_rebuild(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {

  // Positional: at most (metric,)
  Py_ssize_t nargs = args ? PyTuple_GET_SIZE(args) : 0;
  if (nargs > 1) {
    PyErr_SetString(PyExc_TypeError,
      "rebuild() takes at most 1 positional argument (metric)");
    return NULL;
  }

  PyObject* metric_obj = (nargs == 1) ? PyTuple_GET_ITEM(args, 0) : Py_None;  // borrowed
  PyObject* on_disk_path_obj = Py_None;  // borrowed
  PyObject* n_trees_obj = Py_None;       // borrowed
  PyObject* n_jobs_obj = Py_None;        // borrowed

  // Validate kwargs and extract supported keys.
  if (kwargs && kwargs != Py_None) {
    if (!PyDict_Check(kwargs)) {
      PyErr_SetString(PyExc_TypeError, "rebuild() kwargs must be a dict");
      return NULL;
    }

    PyObject* metric_kw = PyDict_GetItemString(kwargs, "metric");  // borrowed
    if (metric_kw) {
      if (nargs == 1 && metric_obj != Py_None) {
        PyErr_SetString(PyExc_TypeError,
          "rebuild() got multiple values for argument 'metric'");
        return NULL;
      }
      metric_obj = metric_kw;
    }

    on_disk_path_obj = PyDict_GetItemString(kwargs, "on_disk_path");
    if (!on_disk_path_obj) on_disk_path_obj = Py_None;

    n_trees_obj = PyDict_GetItemString(kwargs, "n_trees");
    if (!n_trees_obj) n_trees_obj = Py_None;

    n_jobs_obj = PyDict_GetItemString(kwargs, "n_jobs");
    if (!n_jobs_obj) n_jobs_obj = Py_None;

    // Strict: reject unknown keys (sklearn-style).
    PyObject* key = NULL;
    PyObject* value = NULL;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
      if (!PyUnicode_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "rebuild() keyword keys must be str");
        return NULL;
      }
      const char* k = PyUnicode_AsUTF8(key);
      if (!k) return NULL;

      if (std::strcmp(k, "metric") == 0 ||
          std::strcmp(k, "on_disk_path") == 0 ||
          std::strcmp(k, "n_trees") == 0 ||
          std::strcmp(k, "n_jobs") == 0) {
        continue;
      }
      PyErr_Format(PyExc_TypeError,
        "rebuild() got an unexpected keyword argument %R "
        "(allowed: metric, on_disk_path, n_trees, n_jobs)",
        key);
      return NULL;
    }
  }

  // Parse / validate metric (None => reuse current).
  PyObject* metric_out = Py_None;  // borrowed
  if (metric_obj != Py_None) {
    if (!PyUnicode_Check(metric_obj)) {
      PyErr_SetString(PyExc_TypeError,
        "metric must be a string (or None)");
      return NULL;
    }
    const char* s = PyUnicode_AsUTF8(metric_obj);
    if (!s) return NULL;
    MetricId id = metric_from_string(s);
    if (id == METRIC_UNKNOWN) {
      PyErr_SetString(PyExc_ValueError,
        "Invalid metric. Valid options: angular, euclidean, manhattan, dot, hamming.");
      return NULL;
    }
    metric_out = metric_obj;
  }

  // Determine whether to apply a user-provided on_disk_path.
  bool have_user_on_disk_path = (on_disk_path_obj != Py_None);
  std::string user_path;
  if (have_user_on_disk_path) {
    if (!parse_pathlike_to_string_noexc(on_disk_path_obj, &user_path)) {
      return NULL;  // error already set
    }
    if (user_path.empty()) {
      // Treat empty as clear.
      have_user_on_disk_path = false;
    }
  }

  // Parse n_jobs.
  int n_jobs = -1;
  if (n_jobs_obj != Py_None) {
    long v = PyLong_AsLong(n_jobs_obj);
    if (v == -1 && PyErr_Occurred()) return NULL;
    if (v == 0 || v < -1) {
      PyErr_SetString(PyExc_ValueError,
        "n_jobs must be a positive integer or -1");
      return NULL;
    }
    n_jobs = (int)v;
  }

  // Parse n_trees (optional). None => reuse old tree count only if built.
  bool have_user_n_trees = false;
  int n_trees = 0;
  if (n_trees_obj != Py_None) {
    long v = PyLong_AsLong(n_trees_obj);
    if (v == -1 && PyErr_Occurred()) return NULL;
    if (v == 0 || v < -1) {
      PyErr_SetString(PyExc_ValueError,
        "n_trees must be a positive integer or -1");
      return NULL;
    }
    have_user_n_trees = true;
    n_trees = (int)v;
  }

  // Create constructor kwargs from get_params(deep=False).
  PyObject* deep_kwargs = PyDict_New();
  if (!deep_kwargs) return NULL;
  if (PyDict_SetItemString(deep_kwargs, "deep", Py_False) < 0) {
    Py_DECREF(deep_kwargs);
    return NULL;
  }
  PyObject* empty_args = PyTuple_New(0);
  if (!empty_args) {
    Py_DECREF(deep_kwargs);
    return NULL;
  }
  PyObject* params = py_an_get_params(self, empty_args, deep_kwargs);
  Py_DECREF(empty_args);
  Py_DECREF(deep_kwargs);
  if (!params) return NULL;

  // Never carry on_disk_path implicitly.
  if (PyDict_DelItemString(params, "on_disk_path") != 0) {
    if (PyErr_ExceptionMatches(PyExc_KeyError)) {
      PyErr_Clear();
    } else {
      Py_DECREF(params);
      return NULL;
    }
  }

  // Apply metric override (if provided).
  if (metric_out != Py_None) {
    if (PyDict_SetItemString(params, "metric", metric_out) < 0) {
      Py_DECREF(params);
      return NULL;
    }
  }

  // Apply on_disk_path only if explicitly requested and not equal to source.
  if (have_user_on_disk_path) {
    const std::string& old_path = self->on_disk_path;
    if (old_path != user_path) {
      if (PyDict_SetItemString(params, "on_disk_path", on_disk_path_obj) < 0) {
        Py_DECREF(params);
        return NULL;
      }
    }
  }

  // Instantiate new object.
  PyObject* cls = (PyObject*)Py_TYPE(self);
  PyObject* ctor_args = PyTuple_New(0);
  if (!ctor_args) {
    Py_DECREF(params);
    return NULL;
  }
  PyObject* out_obj = PyObject_Call(cls, ctor_args, params);
  Py_DECREF(ctor_args);
  Py_DECREF(params);
  if (!out_obj) return NULL;

  py_annoy* out = (py_annoy*)out_obj;

  // If the source has no underlying index yet, we are done (unfitted rebuild).
  if (!self->ptr) {
    return out_obj;
  }

  // Ensure the target index exists before bulk copy.
  if (!ensure_index(out)) {
    Py_DECREF(out_obj);
    return NULL;
  }

  const int32_t n_items = self->ptr->get_n_items();
  const int old_trees = self->ptr->get_n_trees();

  // Copy items deterministically by item id (0..n_items-1).
  std::vector<float> embedding;
  embedding.resize((size_t)self->f);

  ScopedError error;
  bool ok = true;
  Py_BEGIN_ALLOW_THREADS;
  for (int32_t i = 0; i < n_items; ++i) {
    self->ptr->get_item(i, embedding.data());
    if (!out->ptr->add_item(i, embedding.data(), &error.err)) {
      ok = false;
      break;
    }
  }
  Py_END_ALLOW_THREADS;

  if (!ok) {
    Py_DECREF(out_obj);
    PyErr_SetString(PyExc_RuntimeError, error.err ? error.err : (char*)"add_item failed during rebuild");
    return NULL;
  }

  // Copy labels/targets (y) if present. This is metadata keyed by item id.
  if (self->y) {
    Py_INCREF(self->y);
    Py_XDECREF(out->y);
    out->y = self->y;
  }

  // Rebuild forest deterministically.
  // - If n_trees was provided, honor it.
  // - Else, reuse the source's tree count only if the source was already built.
  int trees_to_build = 0;
  bool do_build = false;
  if (have_user_n_trees) {
    trees_to_build = n_trees;
    do_build = true;
  } else if (old_trees > 0) {
    trees_to_build = old_trees;
    do_build = true;
  }

  if (do_build) {
    bool built_ok = false;
    ScopedError build_error;
    Py_BEGIN_ALLOW_THREADS;
    built_ok = out->ptr->build(trees_to_build, n_jobs, &build_error.err);
    Py_END_ALLOW_THREADS;
    if (!built_ok) {
      Py_DECREF(out_obj);
      PyErr_SetString(PyExc_RuntimeError,
        build_error.err ? build_error.err : (char*)"build failed during rebuild");
      return NULL;
    }
  }

  return out_obj;
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

// ------------------------------------------------------------------
// fit: scikit-learn style entry point
// ------------------------------------------------------------------
//
// Deterministic semantics:
//
// - fit(X=None, y=None, *, n_trees=-1, n_jobs=-1, reset=True, start_index=None)
//   * If X is None and y is None: build the forest using previously-added items
//     (equivalent to calling build(n_trees, n_jobs)).
//   * If X is provided: add all rows in X via add_item and then build.
//       - reset=True (default): clear existing items (fresh index) before adding.
//       - reset=False: append items. If the index is currently built, we will
//         unbuild() first and emit a warning (Annoy cannot add to a built index).
//   * If y is provided alongside X: store labels to :attr:`y` (and alias `_y`) after a
//     successful build. If y is None, `y` is cleared to avoid stale metadata.
//
// Notes
// -----
// - X must be a 2D array-like of shape (n_samples, n_features).
// - Only X and y may be passed positionally; all other parameters are keyword-only.
// - Item ids are assigned deterministically: start_index + row_index.
//   If start_index is None: defaults to 0 when reset=True, otherwise defaults to
//   current n_items when reset=False.
//
static PyObject* py_an_fit(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  // Parse positional (X, y) only.
  Py_ssize_t nargs = args ? PyTuple_GET_SIZE(args) : 0;
  if (nargs > 2) {
    PyErr_SetString(PyExc_TypeError,
      "fit() takes at most 2 positional arguments (X, y)");
    return NULL;
  }

  PyObject* X = (nargs >= 1) ? PyTuple_GET_ITEM(args, 0) : Py_None;  // borrowed
  PyObject* y = (nargs >= 2) ? PyTuple_GET_ITEM(args, 1) : Py_None;  // borrowed

  // Defaults
  int n_trees = 10;
  int n_jobs  = -1;  // -1 => "auto" in Annoy core
  PyObject* reset_obj = Py_True;
  PyObject* start_index_obj = Py_None;
  PyObject* missing_value_obj = Py_None;

  // Allow keyword X/y as well (sklearn style), but forbid duplicates.
  if (kwargs && kwargs != Py_None) {
    if (!PyDict_Check(kwargs)) {
      PyErr_SetString(PyExc_TypeError, "fit() kwargs must be a dict");
      return NULL;
    }

    PyObject* x_kw = PyDict_GetItemString(kwargs, "X");  // borrowed
    PyObject* y_kw = PyDict_GetItemString(kwargs, "y");  // borrowed
    if (x_kw) {
      if (nargs >= 1 && X != Py_None) {
        PyErr_SetString(PyExc_TypeError,
          "fit() got multiple values for argument 'X'");
        return NULL;
      }
      X = x_kw;
    }
    if (y_kw) {
      if (nargs >= 2 && y != Py_None) {
        PyErr_SetString(PyExc_TypeError,
          "fit() got multiple values for argument 'y'");
        return NULL;
      }
      y = y_kw;
    }

    // Validate / parse supported keyword parameters.
    PyObject* key = NULL;
    PyObject* value = NULL;
    Py_ssize_t pos = 0;
    while (PyDict_Next(kwargs, &pos, &key, &value)) {
      if (!PyUnicode_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "fit() keyword keys must be str");
        return NULL;
      }
      const char* k = PyUnicode_AsUTF8(key);
      if (!k) return NULL;

      if (std::strcmp(k, "X") == 0 || std::strcmp(k, "y") == 0) {
        continue;  // handled above
      } else if (std::strcmp(k, "n_trees") == 0) {
        // None means "use the default" (handy when parameters come from config files).
        if (value == Py_None) {
          continue;
        }
        long v = PyLong_AsLong(value);
        if (v == -1 && PyErr_Occurred()) return NULL;
        // Align with build(): allow -1 (auto) or positive integers.
        if (v == 0 || v < -1) {
          PyErr_SetString(PyExc_ValueError,
            "n_trees must be a positive integer or -1");
          return NULL;
        }
        n_trees = (int)v;
      } else if (std::strcmp(k, "n_jobs") == 0) {
        if (value == Py_None) {
          continue;
        }
        long v = PyLong_AsLong(value);
        if (v == -1 && PyErr_Occurred()) return NULL;
        if (v == 0 || v < -1) {
          PyErr_SetString(PyExc_ValueError,
            "n_jobs must be a positive integer or -1");
          return NULL;
        }
        n_jobs = (int)v;
      } else if (std::strcmp(k, "reset") == 0) {
        if (value == Py_None) {
          continue;  // None => keep default
        }
        if (value != Py_True && value != Py_False) {
          PyErr_SetString(PyExc_TypeError, "reset must be a bool");
          return NULL;
        }
        reset_obj = value;
      } else if (std::strcmp(k, "start_index") == 0) {
        start_index_obj = value;  // None or int (validated below)
      } else if (std::strcmp(k, "missing_value") == 0) {
        // If not None, this numeric value is used to impute missing entries
        // (None values in dense rows; missing keys / None values in dict rows).
        missing_value_obj = value ? value : Py_None;
      } else {
        PyErr_Format(PyExc_TypeError,
          "fit() got an unexpected keyword argument %R "
          "(allowed: X, y, n_trees, n_jobs, reset, start_index)",
          key);
        return NULL;
      }
    }
  }

  const bool reset = (reset_obj == Py_True);

  const bool have_X = (X != Py_None);
  const bool have_y = (y != Py_None);

  // Mode 1: build-only (manual add_item workflow)
  if (!have_X && !have_y) {
    PyObject* bargs = PyTuple_New(2);
    if (!bargs) return NULL;

    PyObject* t0 = PyLong_FromLong((long)n_trees);
    PyObject* t1 = PyLong_FromLong((long)n_jobs);
    if (!t0 || !t1) {
      Py_XDECREF(t0);
      Py_XDECREF(t1);
      Py_DECREF(bargs);
      return NULL;
    }
    // PyTuple_SET_ITEM steals references.
    PyTuple_SET_ITEM(bargs, 0, t0);
    PyTuple_SET_ITEM(bargs, 1, t1);

    PyObject* r = py_an_build(self, bargs, NULL);
    Py_DECREF(bargs);
    return r;
  }

  // If y is provided, X must also be provided.
  if (!have_X && have_y) {
    PyErr_SetString(PyExc_TypeError, "fit() got y but X is None");
    return NULL;
  }

  // X must be a 2D array-like (sequence of row sequences).
  PyObject* X_seq = PySequence_Fast(X, "X must be a 2D array-like");
  if (!X_seq) return NULL;

  const Py_ssize_t n_samples = PySequence_Fast_GET_SIZE(X_seq);
  if (n_samples <= 0) {
    Py_DECREF(X_seq);
    PyErr_SetString(PyExc_ValueError, "X must have at least one row");
    return NULL;
  }

  // Handle y if provided: require 1D sequence and match n_samples.
  if (have_y) {
    if (!PySequence_Check(y) || PyUnicode_Check(y) || PyBytes_Check(y)) {
      Py_DECREF(X_seq);
      PyErr_SetString(PyExc_TypeError, "y must be a 1D array-like (sequence)");
      return NULL;
    }
    const Py_ssize_t ny = PySequence_Size(y);
    if (ny < 0) { Py_DECREF(X_seq); return NULL; }
    if (ny != n_samples) {
      Py_DECREF(X_seq);
      PyErr_Format(PyExc_ValueError,
        "y has length %zd but X has %zd rows", ny, n_samples);
      return NULL;
    }
  }

  // If reset=True: drop all items/trees (fresh index) deterministically.
  if (reset) {
    if (self->ptr) {
      try { self->ptr->unload(); } catch (...) {}
      delete self->ptr;
      self->ptr = NULL;
    }
    self->on_disk_active = false;  // items are no longer backed by disk
    // Clear any previously stored labels to prevent stale metadata.
    Py_CLEAR(self->y);
  } else {
    // Append mode: if built, unbuild first and warn.
    if (self->ptr && self->ptr->get_n_trees() > 0) {
      if (PyErr_WarnEx(PyExc_UserWarning,
        "Index is built; calling unbuild() to append new items before rebuilding.",
        1) < 0) {
        Py_DECREF(X_seq);
        return NULL;
      }
      ScopedError error;
      if (!self->ptr->unbuild(&error.err)) {
        Py_DECREF(X_seq);
        PyErr_SetString(PyExc_RuntimeError,
          error.err ? error.err : (char*)"unbuild failed");
        return NULL;
      }
    }
    // In append mode, we keep any existing y metadata if the user does not
    // provide new labels for the appended rows.
  }

  // Determine start_index.
  int64_t start_index = 0;
  if (start_index_obj == Py_None) {
    if (!reset && self->ptr) {
      start_index = (int64_t)self->ptr->get_n_items();
    } else {
      start_index = 0;
    }
  } else {
    long long v = PyLong_AsLongLong(start_index_obj);
    if (v == -1 && PyErr_Occurred()) { Py_DECREF(X_seq); return NULL; }
    if (v < 0) {
      Py_DECREF(X_seq);
      PyErr_SetString(PyExc_ValueError, "start_index must be >= 0");
      return NULL;
    }
    start_index = (int64_t)v;
  }

  // Infer f from first row if needed.
  if (self->f <= 0) {
    PyObject* row0 = PySequence_Fast_GET_ITEM(X_seq, 0);  // borrowed
    if (PyDict_Check(row0)) {
      Py_DECREF(X_seq);
      PyErr_SetString(PyExc_ValueError,
        "Cannot infer f from dict rows. "
        "Set f explicitly before fit() when using dict rows.");
      return NULL;
    }
    PyObject* row0_seq = PySequence_Fast(row0, "X rows must be 1D sequences");
    if (!row0_seq) { Py_DECREF(X_seq); return NULL; }
    const Py_ssize_t f = PySequence_Fast_GET_SIZE(row0_seq);
    Py_DECREF(row0_seq);
    if (f <= 0) { Py_DECREF(X_seq); PyErr_SetString(PyExc_ValueError, "X rows must be non-empty"); return NULL; }
    self->f = (int)f;
  }

  // Default metric if still unknown (true lazy mode).
  if (self->metric_id == METRIC_UNKNOWN) {
    self->metric_id = METRIC_ANGULAR;
  }

  // Ensure index exists with current (f, metric, pending seed/verbose).
  if (!ensure_index(self)) { Py_DECREF(X_seq); return NULL; }

  // Optional imputation for missing values inside X.
  bool allow_missing = false;
  float missing_fill = 0.0f;
  if (missing_value_obj != Py_None) {
    double mv = PyFloat_AsDouble(missing_value_obj);
    if (PyErr_Occurred()) { Py_DECREF(X_seq); return NULL; }
    allow_missing = true;
    missing_fill = (float)mv;
  }

  // Number of items before adding rows from X (used for y merging).
  const int n_items_before = self->ptr ? self->ptr->get_n_items() : 0;

  // Validate start_index + n_samples fits in int32 range.
  if (start_index > (int64_t)INT32_MAX) {
    Py_DECREF(X_seq);
    PyErr_SetString(PyExc_OverflowError, "start_index exceeds int32 range");
    return NULL;
  }
  if (start_index + (int64_t)n_samples - 1 > (int64_t)INT32_MAX) {
    Py_DECREF(X_seq);
    PyErr_SetString(PyExc_OverflowError, "Item ids exceed int32 range");
    return NULL;
  }

  // Add items row-by-row.
  std::vector<float> embedding;
  embedding.reserve((size_t)self->f);

  for (Py_ssize_t i = 0; i < n_samples; ++i) {
    PyObject* row = PySequence_Fast_GET_ITEM(X_seq, i);  // borrowed

    if (!fill_embedding_from_row(
          row, self->f, allow_missing, missing_fill, &embedding, "X")) {
      Py_DECREF(X_seq);
      return NULL;  // exception already set
    }

    const int32_t item_id = (int32_t)(start_index + (int64_t)i);

    // Disallow adding after build (should be prevented earlier, but keep explicit).
    if (self->ptr->get_n_trees() > 0) {
      Py_DECREF(X_seq);
      PyErr_SetString(PyExc_RuntimeError,
        "Index is built; cannot add items. Call unbuild() or fit(reset=True).");
      return NULL;
    }

    ScopedError error;
    if (!self->ptr->add_item(item_id, embedding.data(), &error.err)) {
      Py_DECREF(X_seq);
      PyErr_SetString(PyExc_RuntimeError,
        error.err ? error.err : (char*)"add_item failed");
      return NULL;
    }
  }

  Py_DECREF(X_seq);

  // Build the forest (release GIL for heavy work).
  ScopedError error;
  bool ok = false;
  Py_BEGIN_ALLOW_THREADS;
  ok = self->ptr->build(n_trees, n_jobs, &error.err);
  Py_END_ALLOW_THREADS;

  if (!ok) {
    PyErr_SetString(PyExc_RuntimeError,
      error.err ? error.err : (char*)"build failed");
    return NULL;
  }



// Store y only after successful build.
//
// We store labels as a dict mapping {item_id -> label} so that:
// - gaps are representable,
// - append mode can merge without ambiguity,
// - callers can retrieve labels by the same ids used in add_item.
if (have_y) {
  PyObject* y_seq = PySequence_Fast(y, "y must be a 1D array-like (sequence)");
  if (!y_seq) return NULL;

  PyObject* y_dict = NULL;

  if (!reset && self->y) {
    if (PyDict_Check(self->y)) {
      // Append mode: extend existing mapping in-place.
      y_dict = self->y;
      Py_INCREF(y_dict);
    } else if (PySequence_Check(self->y) && !PyUnicode_Check(self->y) && !PyBytes_Check(self->y)) {
      // Deterministic upgrade path: if an existing sequence aligns to current
      // item ids (length == n_items_before), convert it to a dict mapping.
      const Py_ssize_t n_old = PySequence_Size(self->y);
      if (n_old < 0) { Py_DECREF(y_seq); return NULL; }
      if ((int64_t)n_old == (int64_t)n_items_before) {
        PyObject* old_seq = PySequence_Fast(self->y, "y must be a sequence");
        if (!old_seq) { Py_DECREF(y_seq); return NULL; }

        y_dict = PyDict_New();
        if (!y_dict) { Py_DECREF(old_seq); Py_DECREF(y_seq); return NULL; }

        for (Py_ssize_t i = 0; i < n_old; ++i) {
          PyObject* key = PyLong_FromSsize_t(i);
          if (!key) { Py_DECREF(old_seq); Py_DECREF(y_seq); Py_DECREF(y_dict); return NULL; }
          PyObject* label = PySequence_Fast_GET_ITEM(old_seq, i);  // borrowed
          if (PyDict_SetItem(y_dict, key, label) < 0) {
            Py_DECREF(key);
            Py_DECREF(old_seq);
            Py_DECREF(y_seq);
            Py_DECREF(y_dict);
            return NULL;
          }
          Py_DECREF(key);
        }
        Py_DECREF(old_seq);
      }
    }
  }

  if (!y_dict) {
    // Reset mode or no usable prior labels: create a fresh mapping.
    y_dict = PyDict_New();
    if (!y_dict) { Py_DECREF(y_seq); return NULL; }
  }

  for (Py_ssize_t i = 0; i < n_samples; ++i) {
    const int32_t item_id = (int32_t)(start_index + (int64_t)i);
    PyObject* key = PyLong_FromLong((long)item_id);
    if (!key) { Py_DECREF(y_seq); Py_DECREF(y_dict); return NULL; }

    PyObject* label = PySequence_Fast_GET_ITEM(y_seq, i);  // borrowed
    // PyDict_SetItem INCREFs key and value as needed.
    if (PyDict_SetItem(y_dict, key, label) < 0) {
      Py_DECREF(key);
      Py_DECREF(y_seq);
      Py_DECREF(y_dict);
      return NULL;
    }
    Py_DECREF(key);
  }

  Py_DECREF(y_seq);

  Py_XDECREF(self->y);
  self->y = y_dict;  // already INCREF'ed above
} else if (reset) {
  // Reset mode without labels: remove any prior labels to avoid stale metadata.
  Py_CLEAR(self->y);
}

PY_RETURN_SELF;

}

// ------------------------------------------------------------------
// sklearn-style transformer API: transform / fit_transform
// ------------------------------------------------------------------

// Lookup helper for  metadata (optional). Always returns a new reference.
static PyObject* annoy_lookup_y(
  py_annoy* self,
  int32_t item_id,
  PyObject* y_fill_value) {
  if (!y_fill_value) y_fill_value = Py_None;

  if (!self || !self->y || self->y == Py_None) {
    Py_INCREF(y_fill_value);
    return y_fill_value;
  }

  // Dict mapping: {item_id -> label}
  if (PyDict_Check(self->y)) {
#if PY_VERSION_HEX >= 0x030A0000
    PyObject* key = PyLong_FromLong((long)item_id);
    if (!key) return NULL;
    PyObject* v = PyDict_GetItemWithError(self->y, key);  // borrowed
    Py_DECREF(key);
    if (v) {
      Py_INCREF(v);
      return v;
    }
    if (PyErr_Occurred()) return NULL;
    Py_INCREF(y_fill_value);
    return y_fill_value;
#else
    PyObject* key = PyLong_FromLong((long)item_id);
    if (!key) return NULL;
    PyObject* v = PyDict_GetItem(self->y, key);  // borrowed (no error reporting)
    Py_DECREF(key);
    if (v) {
      Py_INCREF(v);
      return v;
    }
    Py_INCREF(y_fill_value);
    return y_fill_value;
#endif
  }

  // Sequence: index by item_id.
  Py_ssize_t n = PySequence_Size(self->y);
  if (n < 0) return NULL;
  if (item_id < 0 || (Py_ssize_t)item_id >= n) {
    Py_INCREF(y_fill_value);
    return y_fill_value;
  }
  PyObject* v = PySequence_GetItem(self->y, (Py_ssize_t)item_id);  // new ref
  if (!v) return NULL;
  return v;
}

static PyObject* py_an_transform(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  // NOTE: Keep all local declarations that participate in the `fail:` cleanup
  // path *before* any `goto fail;` statements.
  //
  // In C++, it is ill-formed to `goto` forward over a declaration that has an
  // initializer (even for simple pointer initializations like `= NULL`).
  // Some toolchains will error with:
  //   "jump to label 'fail' crosses initialization of ..."
  //
  // Therefore, any locals that are referenced after `fail:` should be declared
  // at the top of the function (or in a scope that is never bypassed by `goto`).
  PyObject* out = NULL;
  PyObject* X = NULL;

  int n_neighbors = 5;
  int search_k = -1;
  int include_distances = 0;
  int return_labels = 0;

  PyObject* y_fill_value = Py_None;
  PyObject* input_type_obj = NULL;     // optional
  PyObject* missing_value_obj = Py_None;

  static const char* kwlist[] = {
    "X",
    "n_neighbors",
    "search_k",
    "include_distances",
    "return_labels",
    "y_fill_value",
    "input_type",
    "missing_value",
    NULL
  };

  if (!PyArg_ParseTupleAndKeywords(
        args, kwargs,
        "O|iippOOO",
        (char**)kwlist,
        &X,
        &n_neighbors,
        &search_k,
        &include_distances,
        &return_labels,
        &y_fill_value,
        &input_type_obj,
        &missing_value_obj)) {
    return NULL;
  }

  if (!self || !self->ptr) {
    PyErr_SetString(PyExc_RuntimeError,
      "Annoy index is not initialized");
    return NULL;
  }
  if (!is_index_built(self)) {
    PyErr_SetString(PyExc_RuntimeError,
      "Index is not built; call build() or fit() before transform().");
    return NULL;
  }
  if (n_neighbors <= 0) {
    PyErr_SetString(PyExc_ValueError, "n_neighbors must be a positive integer");
    return NULL;
  }
  if (search_k < -1) {
    PyErr_SetString(PyExc_ValueError, "search_k must be >= -1");
    return NULL;
  }

  // Optional imputation for missing values inside X.
  bool allow_missing = false;
  float missing_fill = 0.0f;
  if (missing_value_obj != Py_None) {
    double mv = PyFloat_AsDouble(missing_value_obj);
    if (PyErr_Occurred()) return NULL;
    allow_missing = true;
    missing_fill = (float)mv;
  }

  // input_type: "vector" (default) or "item"
  std::string input_type = "vector";
  if (input_type_obj && input_type_obj != Py_None) {
    if (!PyUnicode_Check(input_type_obj) && !PyBytes_Check(input_type_obj)) {
      PyErr_SetString(PyExc_TypeError,
        "input_type must be a string ('vector' or 'item')");
      return NULL;
    }
    const char* s = PyUnicode_Check(input_type_obj)
      ? PyUnicode_AsUTF8(input_type_obj)
      : PyBytes_AsString(input_type_obj);
    if (!s) return NULL;
    input_type.assign(s);
    for (size_t i = 0; i < input_type.size(); ++i) {
      input_type[i] = (char)std::tolower((unsigned char)input_type[i]);
    }
  }

  const bool by_item = (input_type == "item");
  if (!by_item && input_type != "vector") {
    PyErr_SetString(PyExc_ValueError,
      "input_type must be 'vector' or 'item'");
    return NULL;
  }

  PyObject* X_seq = NULL;
  Py_ssize_t n_queries = 0;

  if (by_item) {
    X_seq = PySequence_Fast(X, "X must be a 1D sequence of item ids");
  } else {
    X_seq = PySequence_Fast(X, "X must be a 2D array-like (sequence of rows)");
  }
  if (!X_seq) return NULL;

  n_queries = PySequence_Fast_GET_SIZE(X_seq);
  if (n_queries < 0) { Py_DECREF(X_seq); return NULL; }
  if (n_queries == 0) {
    Py_DECREF(X_seq);
    PyErr_SetString(PyExc_ValueError, "X must be non-empty");
    return NULL;
  }

  PyObject* indices_outer = PyList_New(n_queries);
  PyObject* distances_outer = include_distances ? PyList_New(n_queries) : NULL;
  PyObject* labels_outer = return_labels ? PyList_New(n_queries) : NULL;

  if (!indices_outer || (include_distances && !distances_outer) || (return_labels && !labels_outer)) {
    Py_XDECREF(indices_outer);
    Py_XDECREF(distances_outer);
    Py_XDECREF(labels_outer);
    Py_DECREF(X_seq);
    return NULL;
  }

  std::vector<float> query;
  query.reserve((size_t)self->f);

  std::vector<int32_t> result;
  std::vector<float> distances;

  for (Py_ssize_t i = 0; i < n_queries; ++i) {
    result.clear();
    distances.clear();

    if (by_item) {
      PyObject* obj = PySequence_Fast_GET_ITEM(X_seq, i);  // borrowed
      if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "X must contain integers when input_type='item'");
        goto fail;
      }
      long long kid = PyLong_AsLongLong(obj);
      if (kid == -1 && PyErr_Occurred()) goto fail;
      if (kid < 0 || kid > (long long)INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "Item id out of int32 range");
        goto fail;
      }
      const int32_t item_id = (int32_t)kid;
      if (!check_constraints(self, item_id, /*building=*/false)) goto fail;

      Py_BEGIN_ALLOW_THREADS;
      if (include_distances) {
        self->ptr->get_nns_by_item(item_id, n_neighbors, search_k, &result, &distances);
      } else {
        self->ptr->get_nns_by_item(item_id, n_neighbors, search_k, &result, NULL);
      }
      Py_END_ALLOW_THREADS;
    } else {
      PyObject* row = PySequence_Fast_GET_ITEM(X_seq, i);  // borrowed

      if (!fill_embedding_from_row(
            row, self->f, allow_missing, missing_fill, &query, "X")) {
        goto fail;  // exception already set
      }

      Py_BEGIN_ALLOW_THREADS;
      if (include_distances) {
        self->ptr->get_nns_by_vector(query.data(), n_neighbors, search_k, &result, &distances);
      } else {
        self->ptr->get_nns_by_vector(query.data(), n_neighbors, search_k, &result, NULL);
      }
      Py_END_ALLOW_THREADS;
    }

    if (include_distances && distances.size() != result.size()) {
      PyErr_SetString(PyExc_RuntimeError,
        "Internal error: distances size mismatch");
      goto fail;
    }

    // Build Python row objects.
    PyObject* row_ids = PyList_New((Py_ssize_t)result.size());
    if (!row_ids) goto fail;

    PyObject* row_dists = NULL;
    if (include_distances) {
      row_dists = PyList_New((Py_ssize_t)distances.size());
      if (!row_dists) { Py_DECREF(row_ids); goto fail; }
    }

    PyObject* row_labels = NULL;
    if (return_labels) {
      row_labels = PyList_New((Py_ssize_t)result.size());
      if (!row_labels) {
        Py_DECREF(row_ids);
        Py_XDECREF(row_dists);
        goto fail;
      }
    }

    for (Py_ssize_t j = 0; j < (Py_ssize_t)result.size(); ++j) {
      PyObject* pid = PyLong_FromLong((long)result[(size_t)j]);
      if (!pid) {
        Py_DECREF(row_ids);
        Py_XDECREF(row_dists);
        Py_XDECREF(row_labels);
        goto fail;
      }
      PyList_SET_ITEM(row_ids, j, pid);  // steals ref

      if (include_distances) {
        PyObject* pd = PyFloat_FromDouble((double)distances[(size_t)j]);
        if (!pd) {
          Py_DECREF(row_ids);
          Py_DECREF(row_dists);
          Py_XDECREF(row_labels);
          goto fail;
        }
        PyList_SET_ITEM(row_dists, j, pd);
      }

      if (return_labels) {
        PyObject* lbl = annoy_lookup_y(self, result[(size_t)j], y_fill_value);
        if (!lbl) {
          Py_DECREF(row_ids);
          Py_XDECREF(row_dists);
          Py_DECREF(row_labels);
          goto fail;
        }
        PyList_SET_ITEM(row_labels, j, lbl);  // steals ref
      }
    }

    PyList_SET_ITEM(indices_outer, i, row_ids);

    if (include_distances) {
      PyList_SET_ITEM(distances_outer, i, row_dists);
    }

    if (return_labels) {
      PyList_SET_ITEM(labels_outer, i, row_labels);
    }
  }

  Py_DECREF(X_seq);

  if (!include_distances && !return_labels) {
    return indices_outer;
  }

  if (include_distances && !return_labels) {
    out = PyTuple_Pack(2, indices_outer, distances_outer);
  } else if (!include_distances && return_labels) {
    out = PyTuple_Pack(2, indices_outer, labels_outer);
  } else {
    out = PyTuple_Pack(3, indices_outer, distances_outer, labels_outer);
  }

  Py_DECREF(indices_outer);
  Py_XDECREF(distances_outer);
  Py_XDECREF(labels_outer);

  return out;

fail:
  Py_XDECREF(indices_outer);
  Py_XDECREF(distances_outer);
  Py_XDECREF(labels_outer);
  Py_XDECREF(X_seq);
  return NULL;
}

static PyObject* py_an_fit_transform(
  py_annoy* self,
  PyObject* args,
  PyObject* kwargs) {
  PyObject* X = NULL;
  PyObject* y = Py_None;

  int n_trees = 10;
  int n_jobs = -1;
  int reset = 1;
  PyObject* start_index_obj = Py_None;
  PyObject* missing_value_obj = Py_None;

  int n_neighbors = 5;
  int search_k = -1;
  int include_distances = 0;
  int return_labels = 0;
  PyObject* y_fill_value = Py_None;

  static const char* kwlist[] = {
    "X",
    "y",
    "n_trees",
    "n_jobs",
    "reset",
    "start_index",
    "missing_value",
    "n_neighbors",
    "search_k",
    "include_distances",
    "return_labels",
    "y_fill_value",
    NULL
  };

  if (!PyArg_ParseTupleAndKeywords(
        args, kwargs,
        "O|OiipOOiippO",
        (char**)kwlist,
        &X,
        &y,
        &n_trees,
        &n_jobs,
        &reset,
        &start_index_obj,
        &missing_value_obj,
        &n_neighbors,
        &search_k,
        &include_distances,
        &return_labels,
        &y_fill_value)) {
    return NULL;
  }

  // Delegate to fit() to keep logic centralized and deterministic.
  PyObject* fit_args = PyTuple_Pack(2, X, y ? y : Py_None);
  if (!fit_args) return NULL;

  PyObject* fit_kwargs = PyDict_New();
  if (!fit_kwargs) { Py_DECREF(fit_args); return NULL; }

  PyObject* v = NULL;

  v = PyLong_FromLong((long)n_trees);
  if (!v || PyDict_SetItemString(fit_kwargs, "n_trees", v) < 0) { Py_XDECREF(v); Py_DECREF(fit_args); Py_DECREF(fit_kwargs); return NULL; }
  Py_DECREF(v);

  v = PyLong_FromLong((long)n_jobs);
  if (!v || PyDict_SetItemString(fit_kwargs, "n_jobs", v) < 0) { Py_XDECREF(v); Py_DECREF(fit_args); Py_DECREF(fit_kwargs); return NULL; }
  Py_DECREF(v);

  v = reset ? Py_True : Py_False;
  if (PyDict_SetItemString(fit_kwargs, "reset", v) < 0) { Py_DECREF(fit_args); Py_DECREF(fit_kwargs); return NULL; }

  if (PyDict_SetItemString(fit_kwargs, "start_index", start_index_obj ? start_index_obj : Py_None) < 0) {
    Py_DECREF(fit_args); Py_DECREF(fit_kwargs); return NULL;
  }

  if (PyDict_SetItemString(fit_kwargs, "missing_value", missing_value_obj ? missing_value_obj : Py_None) < 0) {
    Py_DECREF(fit_args); Py_DECREF(fit_kwargs); return NULL;
  }

  PyObject* fitted = py_an_fit(self, fit_args, fit_kwargs);
  Py_DECREF(fit_args);
  Py_DECREF(fit_kwargs);
  if (!fitted) return NULL;
  Py_DECREF(fitted);  // fit() returns self

  // Now transform(X) deterministically.
  PyObject* tr_args = PyTuple_Pack(1, X);
  if (!tr_args) return NULL;

  PyObject* tr_kwargs = PyDict_New();
  if (!tr_kwargs) { Py_DECREF(tr_args); return NULL; }

  v = PyLong_FromLong((long)n_neighbors);
  if (!v || PyDict_SetItemString(tr_kwargs, "n_neighbors", v) < 0) { Py_XDECREF(v); Py_DECREF(tr_args); Py_DECREF(tr_kwargs); return NULL; }
  Py_DECREF(v);

  v = PyLong_FromLong((long)search_k);
  if (!v || PyDict_SetItemString(tr_kwargs, "search_k", v) < 0) { Py_XDECREF(v); Py_DECREF(tr_args); Py_DECREF(tr_kwargs); return NULL; }
  Py_DECREF(v);

  v = include_distances ? Py_True : Py_False;
  if (PyDict_SetItemString(tr_kwargs, "include_distances", v) < 0) { Py_DECREF(tr_args); Py_DECREF(tr_kwargs); return NULL; }

  v = return_labels ? Py_True : Py_False;
  if (PyDict_SetItemString(tr_kwargs, "return_labels", v) < 0) { Py_DECREF(tr_args); Py_DECREF(tr_kwargs); return NULL; }

  if (PyDict_SetItemString(tr_kwargs, "y_fill_value", y_fill_value ? y_fill_value : Py_None) < 0) {
    Py_DECREF(tr_args); Py_DECREF(tr_kwargs); return NULL;
  }

  if (PyDict_SetItemString(tr_kwargs, "missing_value", missing_value_obj ? missing_value_obj : Py_None) < 0) {
    Py_DECREF(tr_args); Py_DECREF(tr_kwargs); return NULL;
  }

  PyObject* out = py_an_transform(self, tr_args, tr_kwargs);
  Py_DECREF(tr_args);
  Py_DECREF(tr_kwargs);
  return out;
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

  // char* error = NULL;
  // if (!self->ptr->save(filename, prefault, &error)) {
  //   PyErr_SetString(PyExc_IOError, error); free(error);
  //   return NULL;
  // }
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

  // Querying requires a built forest (robust, doc-consistent behavior).
  if (!is_index_built(self)) {
    PyErr_SetString(PyExc_RuntimeError,
      "Index is not built; call build() or fit() before querying.");
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

  // Querying requires a built forest (robust, doc-consistent behavior).
  if (!is_index_built(self)) {
    PyErr_SetString(PyExc_RuntimeError,
      "Index is not built; call build() or fit() before querying.");
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

  // Build metadata (sklearn-like): record compile/build info for warnings on load.
  v = PyUnicode_FromString(COMPILER_INFO ". " AVX_INFO);
  if (!v || PyDict_SetItemString(state, "_backend_build", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  v = PyLong_FromLong(annoy_host_is_little_endian() ? 1 : 2);
  if (!v || PyDict_SetItemString(state, "_backend_endian", v) < 0) {
    Py_XDECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // ABI metadata (sklearn-like): record fundamental sizes and key build flags so
  // loaders can emit deterministic warnings when environments differ.
  //
  // Notes
  // -----
  // * Portable snapshots already enforce ABI checks; this is extra, user-facing
  //   context.
  // * Canonical snapshots rebuild deterministically and are safe across ABIs.
  PyObject* abi = PyDict_New();
  if (!abi) { Py_DECREF(state); return NULL; }

  PyObject* tmp = PyLong_FromLong((long)sizeof(size_t));
  if (!tmp || PyDict_SetItemString(abi, "sizeof_size_t", tmp) < 0) { Py_XDECREF(tmp); Py_DECREF(abi); Py_DECREF(state); return NULL; }
  Py_DECREF(tmp);
  tmp = PyLong_FromLong((long)sizeof(void*));
  if (!tmp || PyDict_SetItemString(abi, "sizeof_void_p", tmp) < 0) { Py_XDECREF(tmp); Py_DECREF(abi); Py_DECREF(state); return NULL; }
  Py_DECREF(tmp);
  tmp = PyLong_FromLong((long)sizeof(long));
  if (!tmp || PyDict_SetItemString(abi, "sizeof_long", tmp) < 0) { Py_XDECREF(tmp); Py_DECREF(abi); Py_DECREF(state); return NULL; }
  Py_DECREF(tmp);
  tmp = PyLong_FromLong((long)sizeof(int));
  if (!tmp || PyDict_SetItemString(abi, "sizeof_int", tmp) < 0) { Py_XDECREF(tmp); Py_DECREF(abi); Py_DECREF(state); return NULL; }
  Py_DECREF(tmp);
  tmp = PyLong_FromLong((long)sizeof(float));
  if (!tmp || PyDict_SetItemString(abi, "sizeof_float", tmp) < 0) { Py_XDECREF(tmp); Py_DECREF(abi); Py_DECREF(state); return NULL; }
  Py_DECREF(tmp);
  tmp = PyLong_FromLong((long)sizeof(double));
  if (!tmp || PyDict_SetItemString(abi, "sizeof_double", tmp) < 0) { Py_XDECREF(tmp); Py_DECREF(abi); Py_DECREF(state); return NULL; }
  Py_DECREF(tmp);

#ifdef ANNOYLIB_MULTITHREADED_BUILD
  tmp = Py_True;
#else
  tmp = Py_False;
#endif
  Py_INCREF(tmp);
  if (PyDict_SetItemString(abi, "multithreaded_build", tmp) < 0) { Py_DECREF(tmp); Py_DECREF(abi); Py_DECREF(state); return NULL; }
  Py_DECREF(tmp);

  PyObject* vi = PySys_GetObject((char*)"version_info");  // borrowed
  if (vi) {
    tmp = PyObject_Repr(vi);
    if (!tmp || PyDict_SetItemString(abi, "python_version_info", tmp) < 0) { Py_XDECREF(tmp); Py_DECREF(abi); Py_DECREF(state); return NULL; }
    Py_DECREF(tmp);
  }

  if (PyDict_SetItemString(state, "_backend_abi", abi) < 0) {
    Py_DECREF(abi);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(abi);

  // Index payload snapshot (or None if lazy/uninitialized).
  //
  // schema_version strategy (deterministic, user-controlled):
  //   * 0/1 : store portable-v1 snapshot in `data` (fast restore, ABI-checked)
  //   * 2   : store canonical-v1 blob in `data` (portable, rebuild-on-load)
  //   * >=3 : store portable-v1 in `data` and canonical-v1 in `data_canonical`
  //          (portable fallback if the ABI check fails)
  const bool store_portable  = (self->schema_version <= 1) || (self->schema_version >= 3);
  const bool store_canonical = (self->schema_version >= 2);
  const char* primary_format = store_portable ? "portable-v1" : "canonical-v1";

  if (!self->ptr) {
    Py_INCREF(Py_None);
    v = Py_None;
  } else {
    std::vector<uint8_t> blob_primary;

    if (store_portable) {
      ScopedError error;
      std::vector<uint8_t> native = self->ptr->serialize(&error.err);
      if (native.empty() && error.err) {
        PyErr_SetString(PyExc_RuntimeError, error.err);
        Py_DECREF(state);
        return NULL;
    }
      if (!annoy_build_portable_blob(self, native, &blob_primary)) {
        Py_DECREF(state);
        return NULL;
    }
    } else {
      if (!annoy_build_canonical_blob(self, &blob_primary)) {
        Py_DECREF(state);
        return NULL;
    }
    }

    if (blob_primary.size() > static_cast<size_t>(PY_SSIZE_T_MAX)) {
      PyErr_SetString(PyExc_OverflowError, "pickle snapshot too large for Python bytes");
      Py_DECREF(state);
      return NULL;
    }
    v = PyBytes_FromStringAndSize(
      (const char*)blob_primary.data(), (Py_ssize_t)blob_primary.size());
    if (!v) {
      Py_DECREF(state);
      return NULL;
    }

    // Optional canonical fallback (only when we have an initialized index).
    if (store_portable && store_canonical) {
        std::vector<uint8_t> blob_canon;
        if (!annoy_build_canonical_blob(self, &blob_canon)) {
          Py_DECREF(v);
          Py_DECREF(state);
          return NULL;
        }
        if (blob_canon.size() > static_cast<size_t>(PY_SSIZE_T_MAX)) {
          PyErr_SetString(PyExc_OverflowError, "canonical pickle snapshot too large for Python bytes");
          Py_DECREF(v);
          Py_DECREF(state);
          return NULL;
        }
        PyObject* v2 = PyBytes_FromStringAndSize(
            (const char*)blob_canon.data(), (Py_ssize_t)blob_canon.size());
        if (!v2) {
          Py_DECREF(v);
          Py_DECREF(state);
          return NULL;
        }
        if (PyDict_SetItemString(state, "data_canonical", v2) < 0) {
          Py_DECREF(v2);
          Py_DECREF(v);
          Py_DECREF(state);
          return NULL;
        }
        Py_DECREF(v2);

        PyObject* vfmt2 = PyUnicode_FromString("canonical-v1");
        if (!vfmt2 || PyDict_SetItemString(state, "data_canonical_format", vfmt2) < 0) {
          Py_XDECREF(vfmt2);
          Py_DECREF(v);
          Py_DECREF(state);
          return NULL;
        }
        Py_DECREF(vfmt2);
    }
    }

  if (PyDict_SetItemString(state, "data", v) < 0) {
    Py_DECREF(v);
    Py_DECREF(state);
    return NULL;
  }
  Py_DECREF(v);

  // Snapshot format marker (additive; readers must ignore unknown keys)
  v = PyUnicode_FromString(primary_format);
  if (!v || PyDict_SetItemString(state, "data_format", v) < 0) {
    Py_XDECREF(v);
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

  // Clear fitted labels (if any) on re-init.
  Py_CLEAR(self->y);

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
    // Normalize: seed=0 is treated as the deterministic default seed.
    self->pending_seed = normalize_seed_u64((uint64_t)sv);
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
      // Normalize: seed=0 is treated as the deterministic default seed.
      self->pending_seed = normalize_seed_u64((uint64_t)sv);
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

  // Sklearn-like warning on build mismatch (non-fatal).
  PyObject* build_obj = PyDict_GetItemString(state, "_backend_build");  // borrowed
  if (build_obj && PyUnicode_Check(build_obj)) {
    const char* prev_build = PyUnicode_AsUTF8(build_obj);
    if (prev_build) {
      const char* cur_build = COMPILER_INFO ". " AVX_INFO;
      if (std::strcmp(prev_build, cur_build) != 0) {
        PyErr_WarnFormat(
          PyExc_UserWarning, 1,
          "This Annoy index was pickled with build '%s' but is being loaded with build '%s'. "
          "If you see failures, rebuild the index in this environment.",
          prev_build, cur_build);
        // Ignore warning errors (keep loading); if warning becomes an error due to filters,
        // propagate it to keep behaviour explicit.
        if (PyErr_Occurred()) return NULL;
    }
    }
  }

  // Sklearn-like warning on endian mismatch (non-fatal).
  PyObject* endian_obj = PyDict_GetItemString(state, "_backend_endian");  // borrowed
  if (endian_obj && endian_obj != Py_None) {
    long prev_endian = PyLong_AsLong(endian_obj);
    if (prev_endian == -1 && PyErr_Occurred()) return NULL;
    const long cur_endian = annoy_host_is_little_endian() ? 1 : 2;
    if (prev_endian != cur_endian) {
      PyErr_WarnFormat(
        PyExc_UserWarning, 1,
        "This Annoy index was pickled on a %s-endian machine but is being loaded on a %s-endian machine. "
        "Portable snapshots are ABI-checked; canonical snapshots will rebuild. If you see failures, rebuild in this environment.",
        (prev_endian == 1 ? "little" : "big"),
        (cur_endian == 1 ? "little" : "big"));
      if (PyErr_Occurred()) return NULL;
    }
  }

  // Sklearn-like warning on ABI size mismatch (non-fatal).
  PyObject* abi_obj = PyDict_GetItemString(state, "_backend_abi");  // borrowed
  if (abi_obj && abi_obj != Py_None && PyDict_Check(abi_obj)) {
    struct AbiKey { const char* key; long cur; };
    const AbiKey keys[] = {
      {"sizeof_size_t", (long)sizeof(size_t)},
      {"sizeof_void_p", (long)sizeof(void*)},
      {"sizeof_long",   (long)sizeof(long)},
      {"sizeof_int",    (long)sizeof(int)},
      {"sizeof_float",  (long)sizeof(float)},
      {"sizeof_double", (long)sizeof(double)},
    };
    bool mismatch = false;
    for (const auto& k : keys) {
      PyObject* v = PyDict_GetItemString(abi_obj, k.key);  // borrowed
      if (!v || v == Py_None) continue;
      long prev = PyLong_AsLong(v);
      if (prev == -1 && PyErr_Occurred()) return NULL;
      if (prev != k.cur) { mismatch = true; break; }
    }
    if (mismatch) {
      PyErr_WarnFormat(
        PyExc_UserWarning, 1,
        "This Annoy index was pickled with a different ABI (type sizes differ). "
        "Portable snapshots will raise a deterministic error; canonical snapshots will rebuild. "
        "If you see failures, rebuild the index in this environment.");
      if (PyErr_Occurred()) return NULL;
    }
  }


  PyObject* data = PyDict_GetItemString(state, "data");  // borrowed
  if (data && data != Py_None) {
    if (!PyBytes_Check(data)) {
      PyErr_SetString(PyExc_TypeError,
        "`data` in pickle state must be bytes or None");
      return NULL;
    }

    // Determine declared format (optional).
    bool declared_canonical = false;
    PyObject* fmt_obj = PyDict_GetItemString(state, "data_format");  // borrowed
    if (fmt_obj && fmt_obj != Py_None) {
      if (!PyUnicode_Check(fmt_obj)) {
        PyErr_SetString(PyExc_TypeError,
          "`data_format` in pickle state must be str or None");
        return NULL;
    }
      const char* fmt = PyUnicode_AsUTF8(fmt_obj);
      if (!fmt) return NULL;
      declared_canonical = (std::strcmp(fmt, "canonical-v1") == 0);
    }

    char* buf = NULL;
    Py_ssize_t n = 0;
    if (PyBytes_AsStringAndSize(data, &buf, &n) < 0) {
      return NULL;
    }
    const uint8_t* payload = reinterpret_cast<const uint8_t*>(buf);
    size_t payload_size = static_cast<size_t>(n);

    // Canonical blobs can fully restore by rebuilding (portable across ABIs).
    if (declared_canonical || annoy_is_canonical_blob(payload, payload_size)) {
      if (!annoy_restore_from_canonical_blob(self, payload, payload_size)) {
        // Compatibility: if canonical load fails but we have a backing .annoy file, try it.
        if (!self->on_disk_path.empty()) {
          const char* path = self->on_disk_path.c_str();
          if (!file_exists(path)) {
            return NULL;
          }
          if (self->ptr) { delete self->ptr; self->ptr = NULL; }
          if (!ensure_index(self)) return NULL;

          ScopedError lerr;
          if (!self->ptr->load(path, false, &lerr.err)) {
            PyErr_Format(PyExc_IOError,
                         "canonical restore failed; fallback load('%s') also failed (%s)",
                         path,
                         lerr.err ? lerr.err : "load failed");
            delete self->ptr; self->ptr = NULL;
            return NULL;
          }
          self->on_disk_active = true;
        } else {
          return NULL;  // preserve canonical error
        }
    }
      // Success
      Py_RETURN_NONE;
    }

    // Native / portable snapshots require known configuration.
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


    const uint8_t* native_payload = payload;
    size_t native_payload_size = payload_size;

    // Auto-detect portable wrapper (ABI-guarded)
    if (!annoy_unwrap_portable_blob(self, native_payload, native_payload_size,
                                    &native_payload, &native_payload_size,
                                    NULL, NULL)) {
      // If we stored a canonical fallback, try it now.
      PyObject* data_canon = PyDict_GetItemString(state, "data_canonical");  // borrowed
      if (data_canon && data_canon != Py_None && PyBytes_Check(data_canon)) {
        PyObject *exc=NULL, *val=NULL, *tb=NULL;
        PyErr_Fetch(&exc, &val, &tb);  // preserve portable error
        PyErr_Clear();

        char* cbuf = NULL;
        Py_ssize_t cn = 0;
        if (PyBytes_AsStringAndSize(data_canon, &cbuf, &cn) == 0) {
          const uint8_t* cpay = reinterpret_cast<const uint8_t*>(cbuf);
          size_t csz = static_cast<size_t>(cn);
          if (annoy_is_canonical_blob(cpay, csz) &&
              annoy_restore_from_canonical_blob(self, cpay, csz)) {
            Py_XDECREF(exc); Py_XDECREF(val); Py_XDECREF(tb);
            PyErr_WarnEx(PyExc_UserWarning,
              "Portable snapshot was incompatible; restored from canonical fallback by rebuilding.",
              1);
            if (PyErr_Occurred()) return NULL;
            Py_RETURN_NONE;
          }
        }

        // Canonical fallback failed; restore original error.
        PyErr_Restore(exc, val, tb);
    }
      delete self->ptr; self->ptr = NULL;
      return NULL;
    }

    std::vector<uint8_t> v(native_payload, native_payload + native_payload_size);

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
    "See Also\n"
    "--------\n"
    "build : Build the forest after adding items.\n"
    "unbuild : Remove trees to allow adding more items.\n"
    "get_nns_by_item, get_nns_by_vector : Query nearest neighbours.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "Items must be added *before* calling :meth:`build`. After building\n"
    "the forest, further calls to :meth:`add_item` are not supported.\n"
    "\n"
    "Examples\n"
    "--------\n"
    ">>> import random\n"
    ">>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex\n"
    "...\n"
    ">>> f=100\n"
    ">>> n=1000\n"
    ">>> idx = AnnoyIndex(f, metric='l2')\n"
    "...\n"
    ">>> for i in range(n):\n"
    "...    v = [random.gauss(0, 1) for _ in range(f)]\n"
    "...    idx.add_item(i, v)\n"
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
    "    If set to ``n_trees=-1``, trees are built dynamically until\n"
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
    "See Also\n"
    "--------\n"
    "add_item : Add vectors before building.\n"
    "unbuild : Drop trees to add more items.\n"
    "get_nns_by_item, get_nns_by_vector : Query nearest neighbours.\n"
    "save, load : Persist the index to/from disk.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "After :meth:`build` completes, the index becomes read-only for queries.\n"
    "To add more items, call :meth:`unbuild`, add items, and then rebuild.\n"
    "\n"
    "References\n"
    "----------\n"
    ".. [1] Erik Bernhardsson, \"Annoy: Approximate Nearest Neighbours in C++/Python\".\n"
    "\n"
    "Examples\n"
    "--------\n"
    ">>> import random\n"
    ">>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex\n"
    "...\n"
    ">>> f=100\n"
    ">>> n=1000\n"
    ">>> idx = AnnoyIndex(f, metric='l2')\n"
    "...\n"
    ">>> for i in range(n):\n"
    "...    v = [random.gauss(0, 1) for _ in range(f)]\n"
    "...    idx.add_item(i, v)\n"
    ">>> idx.build(10)\n"
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
    "    Byte string produced by :meth:`serialize`. Both native (legacy)\n"
    "    blobs and portable blobs (created with ``serialize(format='portable')``)\n"
    "    are accepted; portable and canonical blobs are auto-detected.\n"
    "    Canonical blobs restore by rebuilding the index deterministically.\n"
    "prefault : bool or None, optional, default=None\n"
    "    Accepted for API symmetry with :meth:`load`. If None, the stored\n"
    "    Ignored for canonical blobs.\n"
    "    :attr:`prefault` value is used.\n"
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
    "\n"
    "Notes\n"
    "-----\n"
    "Portable blobs add a small header (version, ABI sizes, endianness, metric, f)\n"
    "to ensure incompatible binaries fail loudly and safely. They are not a\n"
    "cross-architecture wire format; the payload remains Annoy's native snapshot.\n"
  },

  {
    "fit",
    (PyCFunction)py_an_fit,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "fit(X=None, y=None, *, n_trees=-1, n_jobs=-1, reset=True, start_index=None, missing_value=None)\n"
    "\n"
    "Fit the Annoy index (scikit-learn compatible).\n"
    "\n"
    "This method supports two deterministic workflows:\n"
    "\n"
    "1) Manual add/build:\n"
    "   If X is None and y is None, fit() builds the forest using items\n"
    "   previously added via add_item().\n"
    "\n"
    "2) Array-like X:\n"
    "   If X is provided (2D array-like), fit() optionally resets or appends,\n"
    "   adds all rows as items, then builds the forest.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "X : array-like of shape (n_samples, n_features), default=None\n"
    "    Vectors to add to the index. If None (and y is None), fit() only builds.\n"
    "y : array-like of shape (n_samples,), default=None\n"
    "    Optional labels associated with X. Stored as :attr:`y` after successful build.\n"
    "n_trees : int, default=-1\n"
    "    Number of trees to build. Use -1 for Annoy's internal default.\n"
    "n_jobs : int, default=-1\n"
    "    Number of threads to use during build (-1 means \"auto\").\n"
    "reset : bool, default=True\n"
    "    If True, clear existing items before adding X. If False, append.\n"
    "start_index : int or None, default=None\n"
    "    Item id for the first row of X. If None, uses 0 when reset=True,\n"
    "    otherwise uses current n_items when reset=False.\n"
    "missing_value : float or None, default=None\n"
    "    If not None, imputes missing entries in X.\n"
    "\n"
    "    - Dense rows: replaces None elements with missing_value.\n"
    "    - Dict rows: fills missing keys (and None values) with missing_value.\n"
    "\n"
    "    If None, missing entries raise an error (strict mode).\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "add_item : Add one item at a time.\n"
    "build : Build the forest after manual calls to add_item.\n"
    "unbuild : Remove trees so items can be appended.\n"
    "y : Stored labels :attr:`y` (if provided).\n"
    "get_params, set_params : Estimator parameter API.\n"
    "\n"
    "Examples\n"
    "--------\n"
    ">>> import random\n"
    ">>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex\n"
    "...\n"
    ">>> n, f = 10_000, 1_000\n"
    ">>> X = [[random.gauss(0, 1) for _ in range(f)] for _ in range(n)]\n"
    ">>> q = [[random.gauss(0, 1) for _ in range(f)]]\n"
    "...\n"
    ">>> for m in ['angular', 'l1', 'l2', '.', 'hamming']:\n"
    "...     idx = AnnoyIndex().set_params(metric=m).fit(X)\n"
    "...     print(m, idx.transform(q))\n"
    "...\n"
    ">>> idx = AnnoyIndex().fit(X)\n"
    ">>> for m in ['angular', 'l1', 'l2', '.', 'hamming']:\n"
    "...     idx_m = base.rebuild(metric=m)  # rebuild-from-index\n"
    "...     print(m, idx_m.transform(q))  # no .fit(X) here\n"
  },

  {
    "fit_transform",
    (PyCFunction)py_an_fit_transform,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "fit_transform(X, y=None, *, n_trees=-1, n_jobs=-1, reset=True, start_index=None,\n"
    "              missing_value=None, n_neighbors=5, search_k=-1, include_distances=False,\n"
    "              return_labels=False, y_fill_value=None)\n"
    "\n"
    "Fit the index and transform X in a single deterministic call.\n"
    "\n"
    "This is equivalent to:\n"
    "    self.fit(X, y=y, n_trees=..., n_jobs=..., reset=..., start_index=..., missing_value=...)\n"
    "    self.transform(X, n_neighbors=..., search_k=..., include_distances=..., return_labels=...,\n"
    "    y_fill_value=..., missing_value=...)\n"
    "\n"
    "See Also\n"
    "--------\n"
    "fit : Build the index.\n"
    "transform : Query the built index.\n"
    "\n"
    "Examples\n"
    "--------\n"
    ">>> import random\n"
    ">>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex\n"
    "...\n"
    ">>> n, f = 10_000, 1_000\n"
    ">>> X = [[random.gauss(0, 1) for _ in range(f)] for _ in range(n)]\n"
    ">>> q = [[random.gauss(0, 1) for _ in range(f)]]\n"
    "...\n"
    ">>> for m in ['angular', 'l1', 'l2', '.', 'hamming']:\n"
    "...     print(m, AnnoyIndex().set_params(metric=m).fit_transform(q))\n"
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
    "get_params",
    (PyCFunction)py_an_get_params,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "get_params(deep=True) -> dict\n"
    "\n"
    "Return estimator-style parameters (scikit-learn compatibility).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "deep : bool, optional, default=True\n"
    "    Included for scikit-learn API compatibility. Ignored because Annoy\n"
    "    does not contain nested estimators.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "params : dict\n"
    "    Dictionary of stable, user-facing parameters.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "set_params : Set estimator-style parameters.\n"
    "schema_version : Controls pickle / snapshot strategy.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This is intended to make Annoy behave like a scikit-learn estimator for\n"
    "tools such as :func:`sklearn.base.clone` and parameter grids.\n"
  },

  {
    "__sklearn_is_fitted__",
    (PyCFunction)py_an_sklearn_is_fitted,
    METH_NOARGS,
    (char*)
    "__sklearn_is_fitted__() -> bool\n"
    "\n"
    "Return whether this estimator is fitted (scikit-learn protocol hook).\n"
    "\n"
    "Returns\n"
    "-------\n"
    "is_fitted : bool\n"
    "    True iff the index has been built (n_trees > 0).\n"
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
    "See Also\n"
    "--------\n"
    "serialize : Create a binary snapshot of the index.\n"
    "deserialize : Restore from a binary snapshot.\n"
    "save : Persist the index to disk.\n"
    "load : Load the index from disk.\n"
    "\n"
    "Notes\n"
    "-----\n"
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
    "Examples\n"
    "--------\n"
    ">>> info = idx.info()\n"
    ">>> info['f']\n"
    "100\n"
    ">>> info['n_items']\n"
    "1000\n"
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
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Raises\n"
    "------\n"
    "IOError\n"
    "    If the file cannot be opened or mapped.\n"
    "RuntimeError\n"
    "    If the index is not initialized or the file is incompatible.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "save : Save the current index to disk.\n"
    "on_disk_build : Build directly using an on-disk backing file.\n"
    "unload : Release mmap resources.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "The in-memory index must have been constructed with the same dimension\n"
    "and metric as the on-disk file.\n"
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
    "See Also\n"
    "--------\n"
    "build : Build trees after adding items (on-disk backed).\n"
    "load : Memory-map the built index.\n"
    "save : Persist the built index to disk.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This mode is useful for very large datasets that do not fit\n"
    "comfortably in RAM during construction.\n"
  },

   {
    "rebuild",
    (PyCFunction)py_an_rebuild,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "rebuild(metric=None, *, on_disk_path=None, n_trees=None, n_jobs=-1) -> Annoy\n"
    "\n"
    "Return a new Annoy index rebuilt from the current index contents.\n"
    "\n"
    "This helper is intended for deterministic, explicit rebuilds when changing\n"
    "structural constraints such as the metric (Annoy uses metric-specific C++\n"
    "index types). The source index is not mutated.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "metric : {'angular', 'euclidean', 'manhattan', 'dot', 'hamming'} or None, optional\n"
    "    Metric for the new index. If None, reuse the current metric.\n"
    "on_disk_path : path-like or None, optional\n"
    "    Optional on-disk build path for the new index.\n"
    "\n"
    "    Safety: the source object's on_disk_path is never carried over implicitly.\n"
    "    If on_disk_path is provided and is string-equal to the source's configured\n"
    "    path, it is ignored to avoid accidental overwrite/truncation hazards.\n"
    "n_trees : int or None, optional\n"
    "    If provided, build the new index with this number of trees (or -1 for\n"
    "    Annoy's internal auto mode). If None, reuse the source's tree count only\n"
    "    when the source index is already built; otherwise do not build.\n"
    "n_jobs : int, optional, default=-1\n"
    "    Number of threads to use while building (-1 means \"auto\").\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    A new Annoy instance containing the same items (and :attr:`y` metadata if present).\n"
    "\n"
    "See Also\n"
    "--------\n"
    "get_params : Read constructor parameters.\n"
    "set_params : Update estimator parameters (use with `fit(X)` when refitting from data).\n"
    "fit : Build the index from `X` (preferred if you already have `X` available).\n"
    "serialize, deserialize : Persist / restore indexes; canonical restores rebuild deterministically.\n"
    "__sklearn_clone__ : Unfitted clone hook (no fitted state).\n"
    "\n"
    "Notes\n"
    "-----\n"
    "`rebuild(metric=...)` is deterministic and preserves item ids (0..n_items-1).\n"
    "by copying item vectors from the current fitted index into a new instance\n"
    "and rebuilding trees.\n"
    "\n"
    "Use `rebuild()` when you want to change `metric` while *reusing the already-stored\n"
    "vectors* (e.g., you do not want to re-read or re-materialize `X`, or you loaded an\n"
    "index from disk and only have access to its stored vectors).\n"
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
    "See Also\n"
    "--------\n"
    "load : Load an index from disk.\n"
    "serialize : Snapshot to bytes for in-memory persistence.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "The output file will be overwritten if it already exists.\n"
    "Use prefault=None to fall back to the stored :attr:`prefault` setting.\n"
  },

  {
    "serialize",
    (PyCFunction)py_an_serialize,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "serialize(format=None) -> bytes\n"
    "\n"
    "Serialize the built in-memory index into a byte string.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "format : {\"native\", \"portable\", \"canonical\"} or None, optional, default=None\n"
    "    Serialization format.\n"
    "\n"
    "    * \"native\" (legacy): raw Annoy memory snapshot. Fastest, but\n"
    "      only compatible when the ABI matches exactly.\n"
    "    * \"portable\": prepend a small compatibility header (version,\n"
    "      endianness, sizeof checks, metric, f) so deserialization fails\n"
    "      loudly on mismatches.\n"
    "    * \"canonical\": rebuildable wire format storing item vectors + build\n"
    "      parameters. Portable across ABIs (within IEEE-754 float32) and\n"
    "      restores by rebuilding trees deterministically.\n"
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
    "OverflowError\n"
    "    If the serialized payload is too large to fit in a Python bytes object.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "deserialize : Restore an index from a serialized byte string.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "\"Portable\" blobs are the native snapshot with additional compatibility guards.\n"
    "They are not a cross-architecture wire format.\n"
    "\n"
    "\"Canonical\" blobs trade load time for portability: deserialization rebuilds\n"
    "the index with ``n_jobs=1`` for deterministic reconstruction.\n"
  },

  {
    "set_params",
    (PyCFunction)py_an_set_params,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "set_params(**params) -> Annoy\n"
    "\n"
    "Set estimator-style parameters (scikit-learn compatibility).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "**params\n"
    "    Keyword parameters to set. Unknown keys raise ``ValueError``.\n"
    "\n"
    "Returns\n"
    "-------\n"
    ":class:`~.Annoy`\n"
    "    This instance (self), enabling method chaining.\n"
    "\n"
    "Raises\n"
    "------\n"
    "ValueError\n"
    "    If an unknown parameter name is provided.\n"
    "TypeError\n"
    "    If parameter names are not strings or types are invalid.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "get_params : Return estimator-style parameters.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "Changing structural parameters (notably ``metric``) on an already\n"
    "initialized index resets the index deterministically (drops all items,\n"
    "trees, and :attr:`y`). Refit/rebuild is required before querying.\n"
    "\n"
    "This behavior matches scikit-learn expectations: ``set_params`` may be\n"
    "called at any time, but parameter changes that affect learned state\n"
    "invalidate the fitted model.\n"
  },

  // ------------------------------------------------------------------
  // RNG / logging controls
  // ------------------------------------------------------------------

  {
    "set_seed",
    (PyCFunction)py_an_set_seed,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "set_seed(seed=None)\n"
    "\n"
    "Set the random seed used for tree construction.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "seed : int or None, optional, default=None\n"
    "    Non-negative integer seed. If called before the index is constructed,\n"
    "    the seed is stored and applied when the C++ index is created.\n"
    "    Seed value ``0`` resets to Annoy's core default seed (with a :class:`UserWarning`).\n"
    "\n"
    "    * If omitted (or None, NULL), the seed is set to Annoy's default seed.\n"
    "    * If 0, clear any pending override and reset to Annoy's default seed\n"
    "      (a :class:`UserWarning` is emitted).\n"
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
    "transform",
    (PyCFunction)py_an_transform,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "transform(X, *, n_neighbors=5, search_k=-1, include_distances=False, return_labels=False,\n"
    "          y_fill_value=None, input_type='vector', missing_value=None)\n"
    "\n"
    "Transform queries into nearest-neighbor ids (and optional distances / labels).\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "X : array-like\n"
    "    Query inputs. The expected shape/type depends on `input_type`:\n"
    "\n"
    "    - input_type='vector': X must be a 2D array-like of shape (n_queries, f).\n"
    "    - input_type='item':   X must be a 1D sequence of item ids.\n"
    "n_neighbors : int, default=5\n"
    "    Number of neighbors to retrieve for each query.\n"
    "search_k : int, default=-1\n"
    "    Search parameter passed to Annoy (-1 uses Annoy's default).\n"
    "include_distances : bool, default=False\n"
    "    If True, also return per-neighbor distances.\n"
    "return_labels : bool, default=False\n"
    "    If True, also return per-neighbor labels resolved from :attr:`y`.\n"
    "y_fill_value : object, default=None\n"
    "    Value used when :attr:`y` is unset or missing an entry for a neighbor id.\n"
    "input_type : {'vector', 'item'}, default='vector'\n"
    "    Controls how X is interpreted.\n"
    "missing_value : float or None, default=None\n"
    "    If not None, imputes missing entries in X (None values in dense rows;\n"
    "    missing keys / None values in dict rows). If None, missing entries raise.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "indices : list of list of int\n"
    "    Neighbor ids for each query.\n"
    "(indices, distances) : tuple\n"
    "    Returned when include_distances=True.\n"
    "(indices, labels) : tuple\n"
    "    Returned when return_labels=True.\n"
    "(indices, distances, labels) : tuple\n"
    "    Returned when include_distances=True and return_labels=True.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "get_nns_by_vector, get_nns_by_item : Low-level query methods.\n"
    "fit, fit_transform : Estimator-style APIs.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "transform() requires a built index; call fit() or build() first.\n"
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
    "See Also\n"
    "--------\n"
    "build : Rebuild the forest after adding new items.\n"
    "add_item : Add items (only valid when no trees are built).\n"
    "\n"
    "Notes\n"
    "-----\n"
    "After calling :meth:`unbuild`, you must call :meth:`build`\n"
    "again before running nearest-neighbour queries.\n"
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
    "See Also\n"
    "--------\n"
    "load : Memory-map an on-disk index into this object.\n"
    "on_disk_build : Configure on-disk build mode.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This releases OS-level resources associated with the mmap,\n"
    "but keeps the Python object alive.\n"
  },

  {
    "set_verbose",
    (PyCFunction)py_an_verbose,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "set_verbose(level=1)\n"
    "\n"
    "Set the verbosity level (callable setter).\n"
    "\n"
    "This method exists to preserve a callable interface while keeping the\n"
    "parameter name ``verbose`` available as an attribute for scikit-learn\n"
    "compatibility.\n"
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
    "\n"
    "See Also\n"
    "--------\n"
    "set_verbosity : Alias of :meth:`set_verbose`.\n"
    "verbose : Parameter attribute (int | None).\n"
    "get_params, set_params : Estimator parameter API.\n"
  },

  {
    "set_verbosity",
    (PyCFunction)py_an_verbose,
    METH_VARARGS | METH_KEYWORDS,
    (char*)
    "set_verbosity(level=1)\n"
    "\n"
    "Alias of :meth:`set_verbose`.\n"
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
    "See Also\n"
    "--------\n"
    "info : Return a Python dict with configuration and metadata.\n"
    "__repr__ : Text representation.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This representation is deterministic and side-effect free. It intentionally\n"
    "avoids expensive operations such as serialization or memory-usage estimation.\n"
  },

  {
    "__sklearn_tags__",
    (PyCFunction)py_an_sklearn_tags,
    METH_NOARGS,
    (char*)
    "__sklearn_tags__() -> sklearn.utils.Tags\n"
    "\n"
    "Return estimator tags (scikit-learn protocol hook).\n"
    "\n"
    "Returns\n"
    "-------\n"
    "tags : sklearn.utils.Tags\n"
    "    Conservative tags for this estimator.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "sklearn.utils.get_tags : Read estimator tags.\n"
    "\n"
    "Notes\n"
    "-----\n"
    "This method is consulted by scikit-learn utilities such as\n"
    "``sklearn.utils.get_tags``.\n"
  },

  {
    "__sklearn_clone__",
    (PyCFunction)py_an_sklearn_clone,
    METH_NOARGS,
    (char*)
    "__sklearn_clone__() -> Annoy\n"
    "\n"
    "Return an unfitted clone (scikit-learn protocol hook).\n"
    "\n"
    "Returns\n"
    "-------\n"
    "clone : :class:`~.Annoy`\n"
    "    New unfitted instance with identical parameters.\n"
    "\n"
    "See Also\n"
    "--------\n"
    "get_params : Parameters used for cloning.\n"
    "sklearn.base.clone : Delegates to this hook when available.\n"
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

  out = PyUnicode_FromFormat("Annoy(**%U)", d_repr);
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

  out = PyUnicode_FromFormat("Annoy(**%U)", d_repr);
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
#__ANNOY_REPR_ID__ .annoy-td-btn{width:72px;}
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
  // html.append("#"); html.append(idbuf); html.append(" .annoy-td-btn{width:72px;}");
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
  // traverse
  Py_VISIT(self->dict);
  Py_VISIT(self->y);
  return 0;
}

static int py_an_clear(
  PyObject* obj) {
  py_annoy* self = (py_annoy*)obj;
  // clear
  Py_CLEAR(self->dict);
  Py_CLEAR(self->y);
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
  {NULL, NULL, 0, NULL}  /* Sentinel */
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
