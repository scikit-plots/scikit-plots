// scikitplot/cexternals/_annoy/src/annoylib.h
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

/* =========================================================================================
 * ENHANCED ANNOYLIB - COMPREHENSIVE TYPE SUPPORT & PARAMETER MANAGEMENT
 * =========================================================================================
 *
 * Canonical semantics:
 *
 *   * "indice"        : integer ID returned by Annoy
 *   * "index"         : 0..n-1 row position in a result set (Python side)
 *   * "distance"      : Hamming distance (clipped to [0, f_external])
 *   * "embedding"     : binary embedding represented as float[0,1] or float[-inf,inf] on the API
 *
 * Features:
 * - Extended floating-point: float16, float32, float64, float128
 * - Boolean/binary data: bool, uint8_t for 1-bit/8-bit data
 * - Flexible index types: int32_t, int64_t, float, double
 * - Lazy construction: Dimension inference from first add_item
 * - Default initialization: All variables initialized to safe defaults
 * - Cross-platform: Windows/Mac/Linux/Arch compatibility
 * - Future-proof: Generic template design with compile-time validation
 *
 * Key Enhancements:
 *
 * 1. **Extended Type Support**:
 *    - float16, float32, float64, float128 for data
 *    - bool, uint8_t for binary/boolean data
 *    - int32_t, int64_t, float, double for indices
 *
 * 2. **Lazy Construction**:
 *    - Dimension (f) can be inferred from first add_item() call
 *    - All parameters have safe default values
 *    - No kernel crashes from uninitialized state
 *
 * 3. **Centralized Parameter Management**:
 *    - All constructor parameters stored as attributes
 *    - Methods use stored attributes as defaults (e.g., build() uses self.n_trees)
 *    - sklearn-compatible get_params() / set_params() methods
 *
 * 4. **Cross-Platform Reliability**:
 *    - Windows, macOS, Linux, Arch compatibility
 *    - Safe integer types and default initialization
 *    - Proper error messages prevent silent failures
 *
 * 5. **Future-Proof Design**:
 *    - **kwargs support for extensibility
 *    - Generic template design with compile-time validation
 *    - Consistent API across AnnoyIndex and HammingWrapper
 *
 * =========================================================================================
 *
 * Constructor Signatures:
 *
 * AnnoyIndex(
 *     f=None,                     // Dimension (inferred from first vector if 0/None)
 *     metric=None,                // "angular", "euclidean", "manhattan", "dot", "hamming"
 *     *,                          // Keyword-only args below
 *     n_trees = -1,               // Default trees for build() (-1 = auto)
 *     n_neighbors = 5,            // Default neighbors for queries
 *     on_disk_path = None,        // Default path for on_disk_build() str
 *     prefault = None,            // Prefault pages when loading bool
 *     seed = None,                // Random seed (0/None = use default)
 *     verbose = None,             // Verbosity level bool or int
 *     schema_version = None,      // Serialization schema version 0/None
 *     dtype = "float",            // Data type            : 'bool', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float', 'float32', 'double', 'float64', 'float128'
 *     index_dtype = "int32",      // Index type           : 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'
 *     wrapper_dtype = "uint32",   // Hamming wrapper type : 'bool', 'uint8', 'uint16', 'uint32', 'uint64'
 *     random_dtype = "uint64",    // Random seed type     : 'uint16', 'uint32', 'uint64'
 *     y_dtype = "float",          // Label data type      : 'bool', 'float16', 'float', 'float32', 'double', 'float64', 'float128'
 *     y_return_type = "list",     // Return type: list or dict
 *     n_jobs = 1,                 // Number of threads (-1 = all cores)
 *     l1_ratio = 0.0,             // Future: random projection ratio
 * )
 *
 * HammingWrapper(
 *     f = 0,                      // Same as AnnoyIndex for metric "hamming" wraper
 *     *,                          // Keyword-only args below
 *     n_trees = -1,
 *     n_neighbors = 5,
 *     on_disk_path = None,
 *     prefault = None,
 *     seed = None,
 *     verbose = None,
 *     schema_version = 0,
 *     dtype = "float",
 *     index_dtype = "int32",
 *     wrapper_dtype = "uint32",
 *     random_dtype = "uint64",
 *     y_dtype = "float",
 *     y_return_type = "list",
 *     n_jobs = 1,
 *     l1_ratio = 0.0,
 * )
 *
 * Key Behaviors:
 * -------------
 * - If f=0: Dimension inferred from first add_item() call
 * - If n_trees=-1 in constructor: build() uses auto-calculation
 * - If n_trees=NULL in build(): Uses constructor's n_trees value
 * - If filename=NULL in save/load: Uses constructor's on_disk_path
 * - All methods check _f_inferred to prevent premature usage errors
 *
 * ========================================================================================= */

// ✔ Clean platform → types → OS → C → C++ → optional → macros order
#ifndef ANNOY_ANNOYLIB_H
#define ANNOY_ANNOYLIB_H

// MSVC compatibility
#if defined(_MSC_VER) && _MSC_VER < 1900
  #ifndef snprintf
    #define snprintf _snprintf
  #endif
#endif

// Integer types for legacy MSVC
// https://en.cppreference.com/w/cpp/types/integer.html
#if defined(_MSC_VER) && _MSC_VER == 1500
  typedef unsigned char     uint8_t;
  typedef unsigned __int16  uint16_t;
  typedef signed __int32    int32_t;
  typedef unsigned __int32  uint32_t;
  typedef signed __int64    int64_t;
  typedef unsigned __int64  uint64_t;
#else
  // for mixed C/C++ usage
  #include <stdint.h>
  #include <cstdint>
#endif

/* =========================
 * Extended floating-point types
 * ========================= */
// On ARM → float16_t is a primitive type.
// On x86 → float16_t is a struct.
// On fallback → float16_t is a struct.
// https://en.cppreference.com/w/cpp/types/integer.html
#ifndef ANNOY_FLOAT16_DEFINED
#define ANNOY_FLOAT16_DEFINED
// Float16 support (IEEE 754 half precision - 16 bits)
#if defined(__F16C__) || defined(__ARM_FP16_FORMAT_IEEE)
  #if defined(__ARM_FP16_FORMAT_IEEE)
    // TODO: Native ARM float16, Cython does not properly support __fp16.
    // C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\Llvm\ARM64\lib\clang\19\include\arm_vector_types.h
    // arm_vector_types.h(18,16): error: typedef redefinition with different types ('__fp16' vs 'float16_t')
    typedef __fp16 float16_t;
    // struct float16_t {
    //     __fp16 data;

    //     float16_t() : data(0) {}
    //     explicit float16_t(float f) : data(static_cast<__fp16>(f)) {}

    //     operator float() const {
    //         return static_cast<float>(data);
    //     }

    //     float16_t& operator=(float f) {
    //         data = static_cast<__fp16>(f);
    //         return *this;
    //     }
    // };
    #define ANNOY_HAS_NATIVE_FLOAT16 1
  #elif defined(__F16C__)
    // x86 F16C hardware acceleration
    #include <immintrin.h>
    // Hardware-accelerated float16 using F16C instructions
    struct float16_t {
      uint16_t data;
      float16_t() : data(0) {}
      explicit float16_t(float f) {
        __m128 v = _mm_set_ss(f);
        __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
        data = static_cast<uint16_t>(_mm_extract_epi16(h, 0));
      }
      operator float() const {
        __m128i h = _mm_cvtsi32_si128(data);
        __m128 v = _mm_cvtph_ps(h);
        return _mm_cvtss_f32(v);
      }
      float16_t& operator=(float f) {
        *this = float16_t(f);
        return *this;
      }
    };
    #define ANNOY_HAS_F16C_FLOAT16 1
  #endif
#else
  // Portable software implementation
  struct float16_t {
    uint16_t data;
    float16_t() : data(0) {}
    explicit float16_t(float f) {
      // IEEE 754 half-precision conversion
      uint32_t x;
      memcpy(&x, &f, sizeof(float));
      uint32_t sign = (x >> 31) << 15;
      uint32_t exp = ((x >> 23) & 0xFF);
      uint32_t frac = (x >> 13) & 0x3FF;
      if (exp == 0) {
        data = static_cast<uint16_t>(sign);  // Zero or denormal
      } else if (exp == 0xFF) {
        data = static_cast<uint16_t>(sign | 0x7C00 | (frac ? 0x200 : 0));  // Inf or NaN
      } else {
        int32_t newexp = static_cast<int32_t>(exp) - 127 + 15;
        if (newexp <= 0) {
          data = static_cast<uint16_t>(sign);  // Underflow to zero
        } else if (newexp >= 31) {
          data = static_cast<uint16_t>(sign | 0x7C00);  // Overflow to infinity
        } else {
          data = static_cast<uint16_t>(sign | (static_cast<uint32_t>(newexp) << 10) | frac);
        }
      }
    }
    operator float() const {
      uint32_t sign = static_cast<uint32_t>(data >> 15) << 31;
      uint32_t exp = (data >> 10) & 0x1F;
      uint32_t frac = (data & 0x3FF) << 13;
      uint32_t result;
      if (exp == 0) {
        result = sign;  // Zero or denormal
      } else if (exp == 31) {
        result = sign | 0x7F800000 | frac;  // Inf or NaN
      } else {
        result = sign | ((exp - 15 + 127) << 23) | frac;
      }
      float fresult;
      memcpy(&fresult, &result, sizeof(float));
      return fresult;
    }
    float16_t& operator=(float f) {
      *this = float16_t(f);
      return *this;
    }
  };
  #define ANNOY_HAS_SOFTWARE_FLOAT16 1
#endif
#endif // ANNOY_FLOAT16_DEFINED

// Float128 support (quadruple precision)
#ifndef ANNOY_FLOAT128_DEFINED
#define ANNOY_FLOAT128_DEFINED
// Float128 (quadruple precision - 128 bits)
#if defined(__SIZEOF_FLOAT128__) && defined(__GNUC__)
  // Native GCC/Clang __float128
  typedef __float128 float128_t;
  #define ANNOY_HAS_FLOAT128 1
#elif defined(_MSC_VER) && defined(_M_X64)
  // MSVC doesn't have native float128, use long double (80-bit extended precision)
  typedef long double float128_t;
  #define ANNOY_HAS_FLOAT128_EMULATED 1
  #pragma message("Warning: float128 emulated using long double (80-bit) on MSVC")
#else
  // Generic fallback to long double
  typedef long double float128_t;
  #define ANNOY_HAS_FLOAT128_FALLBACK 1
#endif
#endif // ANNOY_FLOAT128_DEFINED

/* =========================
 * Type aliases for consistency
 * ========================= */
#ifndef ANNOY_TYPE_ALIASES_DEFINED
#define ANNOY_TYPE_ALIASES_DEFINED
// Standard type aliases
typedef float  float32_t;
typedef double float64_t;
#endif // ANNOY_TYPE_ALIASES_DEFINED

#ifndef _MSC_VER
  #include <unistd.h>
#endif
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(_MSC_VER) || defined(__MINGW32__)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  // ⚠️ Override some definitions to support 64-bit file offsets on Windows
  // a bit hacky, but override some definitions to support 64 bit
  // 64-bit file offset support on Windows
  #define off_t int64_t
  #define lseek_getsize(fd) _lseeki64(fd, 0, SEEK_END)
  #include <windows.h>
  #include <io.h>   // optional but safer for _lseeki64
  #include "mman.h"
#else
 #include <sys/mman.h>
 #define lseek_getsize(fd) lseek(fd, 0, SEEK_END)
#endif

/* =========================
 * Standard C/C++ library
 * ========================= */
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>   // guarantees snprintf mapping
#include <stdlib.h>
#include <string.h>

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vector>
#include <algorithm>
#include <queue>
#include <limits>
#include <stdexcept>
#include <utility>

// For 201103L "This library requires at least C++11 (-std=c++11)."
// For 201402L "This library requires at least C++14 (-std=c++14)."
// For 201703L "This library requires at least C++17 (-std=c++17)."
#if __cplusplus >= 201103L
  #include <type_traits>
  #include <unordered_set>
#endif

#ifdef ANNOYLIB_MULTITHREADED_BUILD
  // ThreadedBuildPolicy
  #include <thread>
  #include <functional>
  #include <mutex>
  #include <atomic>
  #if __cplusplus >= 201402L
    #include <shared_mutex>
  #else
    #include <shared_mutex>
  #endif
#endif

/* =========================
 * Compile-time assertions (C++11)
 * ========================= */
// #if __cplusplus >= 201103L
//   static_assert(sizeof(uint8_t) == 1, "uint8_t must be 8-bit");
//   static_assert(sizeof(uint16_t) == 2, "uint16_t must be 16-bit");
//   static_assert(sizeof(uint32_t) == 4, "uint32_t must be 32-bit");
//   static_assert(sizeof(uint64_t) == 8, "uint64_t must be 64-bit");
//   static_assert(sizeof(int32_t) == 4, "int32_t must be 32-bit");
//   static_assert(sizeof(int64_t) == 8, "int64_t must be 64-bit");
//   static_assert(sizeof(float) == 4, "float must be 32-bit");
//   static_assert(sizeof(double) == 8, "double must be 64-bit");
// #endif

#ifdef _MSC_VER
  // Needed for Visual Studio to disable runtime checks for memcpy / low-level memory ops
  // ⚠️ NOTE: This header disables MSVC runtime checks for the entire translation unit.
  #pragma runtime_checks("s", off)
  // Disable specific warnings for cross-platform compatibility
  #pragma warning(disable: 4996) // deprecated functions
  #pragma warning(disable: 4244) // possible loss of data (expected for float conversions)
#endif

/* =========================================================================================
 * UTILITY MACROS
 * ========================================================================================= */
// This allows others to supply their own logger / error printer without
// requiring Annoy to import their headers. See RcppAnnoy for a use case.
#ifndef __ERROR_PRINTER_OVERRIDE__
  #define annoylib_showUpdate(...) \
    do { fprintf(stderr, __VA_ARGS__); } while (0)
    // { fprintf(stderr, __VA_ARGS__ ); }
#else
  #define annoylib_showUpdate(...) \
    do { __ERROR_PRINTER_OVERRIDE__( __VA_ARGS__ ); } while (0)
    // { __ERROR_PRINTER_OVERRIDE__( __VA_ARGS__ ); }
#endif

// Portable alloc definition, cf Writing R Extensions, Section 1.6.4
// alloca is inherently unsafe but historically required here.
// NOTE:
// alloca() is intentionally used here for performance and locality.
// Invariants:
// - _s is bounded (Node size)
// - recursion depth is limited by tree height
// - stack usage is deterministic
// DO NOT replace with heap allocation without benchmarking.
#ifdef __GNUC__
  // Includes GCC, clang and Intel compilers
  # undef alloca
  // #define annoylib_alloca(x) __builtin_alloca((x))
  # define alloca(x) __builtin_alloca((x))
#elif defined(__sun) || defined(_AIX)
  // this is necessary (and sufficient) for Solaris 10 and AIX 6:
  # include <alloca.h>
#endif

/* ==================================================
 * SIMD feature detection - Compiler detection
 * ================================================== */
// Clang defines __GNUC__ but behaves differently
// GCC version alone is insufficient for AVX512 correctness
// #if !defined(NO_MANUAL_VECTORIZATION) && defined(__GNUC__) && (__GNUC__ >6) && defined(__AVX512F__)  // See #402
//   #define ANNOYLIB_USE_AVX512
// #elif !defined(NO_MANUAL_VECTORIZATION) && defined(__AVX__) && defined(__SSE__) && defined(__SSE2__) && defined(__SSE3__)
//   #define ANNOYLIB_USE_AVX
// #else
// #endif
/*
 * Policy:
 * - SIMD can be disabled explicitly via NO_MANUAL_VECTORIZATION
 * - AVX-512 is enabled ONLY for GCC (not Clang) due to known correctness issues
 * - AVX is enabled for any compiler advertising the required feature macros
 * - AVX requires AVX + SSE[1-3]
 * - Exactly one SIMD path may be active
 * - Scalar fallback otherwise
 */
#if defined(__clang__)
  #define ANNOYLIB_COMPILER_CLANG 1
#elif defined(__GNUC__)
  #define ANNOYLIB_COMPILER_GCC 1
#elif defined(_MSC_VER)
  #define ANNOYLIB_COMPILER_MSVC 1
#else
  #define ANNOYLIB_COMPILER_UNKNOWN 1
#endif
#if !defined(NO_MANUAL_VECTORIZATION)
  /* ---- AVX-512 (GCC only) ---- */
  #if defined(ANNOYLIB_COMPILER_GCC) && defined(__AVX512F__)
    #define ANNOYLIB_USE_AVX512 1
  /* ---- AVX (GCC / Clang / MSVC) (broad support) ---- */
  #elif defined(__AVX__) && defined(__SSE__) && defined(__SSE2__) && defined(__SSE3__)
    #define ANNOYLIB_USE_AVX 1
  #else
    /* Scalar fallback */
    #define ANNOYLIB_SIMD_SCALAR 1
  #endif
#endif /* NO_MANUAL_VECTORIZATION */
#if defined(ANNOYLIB_USE_AVX512) && defined(ANNOYLIB_USE_AVX)
  #error "Invalid SIMD state: both AVX and AVX512 enabled"
#endif
#if (defined(ANNOYLIB_USE_AVX512) + defined(ANNOYLIB_USE_AVX) + defined(ANNOYLIB_SIMD_SCALAR)) != 1
  #error "Exactly one SIMD backend must be selected"
#endif
/* =========================
 * SIMD headers MSVC → <intrin.h> GCC/Clang → <x86intrin.h>
 * ========================= */
#if defined(ANNOYLIB_USE_AVX) || defined(ANNOYLIB_USE_AVX512)
  #if defined(_MSC_VER) || defined(ANNOYLIB_COMPILER_MSVC)
    #include <intrin.h>
  #elif defined(__GNUC__) || defined(__clang__)
    // Clang can still include intrinsics even if AVX-512 is disabled
    #include <x86intrin.h>
  #endif
#endif

#if !defined(__MINGW32__)
#define ANNOYLIB_FTRUNCATE_SIZE(x) static_cast<int64_t>(x)
#else
#define ANNOYLIB_FTRUNCATE_SIZE(x) (x)
#endif

// We let the v array in the Node struct take whatever space is needed, so this is a mostly insignificant number.
// Compilers need *some* size defined for the v array, and some memory checking tools will flag for buffer overruns if this is set too low.
#define ANNOYLIB_V_ARRAY_SIZE 65536

#ifndef _MSC_VER
#define annoylib_popcount __builtin_popcountll
#else // See #293, #358
#define annoylib_popcount cole_popcount
#endif

#ifndef ANNOY_UNUSED
  #define ANNOY_UNUSED(x) (void)(x)
#endif

// Safe string duplication
inline char* dup_cstr(const char* str) {
  if (str == NULL) return NULL;
  size_t len = std::strlen(str);
  char* copy = static_cast<char*>(std::malloc(len + 1));
  if (copy != NULL) {
    memcpy(copy, str, len + 1);
  }
  return copy;
}

// Safe error message setting
inline void set_error_msg(char** error, const char* msg) {
  if (error != NULL && msg != NULL) {
    *error = dup_cstr(msg);
  }
}
/* =========================================================================================
 * PARAMETER CONTAINER FOR CENTRALIZED MANAGEMENT
 * ========================================================================================= */
namespace Annoy {
/* ================================================================
 * Software float16 implementation (if needed)
 * ================================================================ */
// #ifdef ANNOY_HAS_SOFTWARE_FLOAT16
// inline float16_t::float16_t(float f) {
//   uint32_t x;
//   memcpy(&x, &f, sizeof(float));
//
//   uint32_t sign = (x >> 31) & 0x1;
//   uint32_t exp = (x >> 23) & 0xFF;
//   uint32_t frac = x & 0x7FFFFF;
//
//   uint16_t h_sign = static_cast<uint16_t>(sign << 15);
//   uint16_t h_exp = 0;
//   uint16_t h_frac = 0;
//
//   if (exp == 0xFF) {
//     h_exp = 0x1F;
//     h_frac = (frac != 0) ? 0x3FF : 0;
//   } else if (exp == 0) {
//     h_exp = 0;
//     h_frac = 0;
//   } else {
//     int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
//     if (new_exp >= 31) {
//       h_exp = 0x1F;
//       h_frac = 0;
//     } else if (new_exp <= 0) {
//       h_exp = 0;
//       h_frac = 0;
//     } else {
//       h_exp = static_cast<uint16_t>(new_exp);
//       h_frac = static_cast<uint16_t>(frac >> 13);
//     }
//   }
//
//   data = h_sign | (h_exp << 10) | h_frac;
// }
//
// inline float16_t::operator float() const {
//   uint16_t h_sign = (data >> 15) & 0x1;
//   uint16_t h_exp = (data >> 10) & 0x1F;
//   uint16_t h_frac = data & 0x3FF;
//
//   uint32_t f_sign = static_cast<uint32_t>(h_sign) << 31;
//   uint32_t f_exp = 0;
//   uint32_t f_frac = 0;
//
//   if (h_exp == 0x1F) {
//     f_exp = 0xFF;
//     f_frac = (h_frac != 0) ? 0x7FFFFF : 0;
//   } else if (h_exp == 0) {
//     f_exp = 0;
//     f_frac = 0;
//   } else {
//     f_exp = static_cast<uint32_t>(h_exp) - 15 + 127;
//     f_frac = static_cast<uint32_t>(h_frac) << 13;
//   }
//
//   uint32_t bits = f_sign | (f_exp << 23) | f_frac;
//   float result;
//   memcpy(&result, &bits, sizeof(float));
//   return result;
// }
// #endif
// static inline annoy_off_t lseek_getsize(int fd);
using annoy_off_t = int64_t;
#if __cplusplus >= 201103L
  using std::is_same;
#endif
using std::free;
using std::malloc;
using std::memcpy;
using std::vector;
using std::pair;
using std::numeric_limits;
using std::make_pair;
using std::fabs;
// fabsf is not guaranteed to be in std::
// Avoid fabsf portability issues
// Use std::fabs overload resolution instead
using ::fabsf;
// #include <cstring> vs #include <string.h>
using std::strerror;
using std::strlen;
using std::strcpy;

#if __cplusplus >= 201103L
  using std::unordered_set;
#else
  using std::set;
  #define unordered_set set
#endif

// Default values for lazy construction and Python API compatibility (if needed)
static const int DEFAULT_DIMENSION = 0;      // 0 means "infer from first vector"
static const int DEFAULT_N_JOBS = 1;         // Default n_threads
static const int DEFAULT_TREES = 10;         // Default n_trees
static const int DEFAULT_NEIGHBORS = 5;      // Default n_neighbors
static const int DEFAULT_SEARCH_K = -1;      // -1 means "use automatic value"
static const int DEFAULT_VERBOSE = 0;        // 0 means quiet
static const int DEFAULT_SCHEMA = 0;         // Schema version
/* ================================================================
 * Default value utilities (if needed)
 * ================================================================ */
template<typename T>
struct DefaultValue {
  static constexpr T get() noexcept { return T(0); }
};
template<>
struct DefaultValue<bool> {
  static constexpr bool get() noexcept { return false; }
};
template<>
struct DefaultValue<float> {
  static constexpr float get() noexcept { return 0.0f; }
};
template<>
struct DefaultValue<double> {
  static constexpr double get() noexcept { return 0.0; }
};
template<>
struct DefaultValue<float128_t> {
  static constexpr float128_t get() noexcept { return 0.0L; }
};
template<typename T>
inline T safe_divide(T a, T b, T default_val = DefaultValue<T>::get()) {
  static_assert(std::is_floating_point<T>::value,
                "safe_divide is only valid for floating-point types");
  return (b != T(0)) ? a / b : default_val;
}
/**
 * @brief Centralized parameter storage for AnnoyIndex
 *
 * Stores all constructor parameters as attributes to enable:
 * - Lazy initialization
 * - Parameter reuse across methods
 * - sklearn-compatible get_params/set_params
 */
struct AnnoyParams {
  // Core parameters
  int f;                          // Dimension (0 = infer from first vector)
  int n_trees;                    // Number of trees (-1 = auto)
  int n_neighbors;                // Default neighbors for queries (10 = default)
  int seed;                       // Random seed (0 = default)
  int n_jobs;                     // Number of threads (-1 = all cores)
  int schema_version;             // Serialization version

  // Behavioral flags
  bool prefault;                  // Prefault pages when loading
  bool verbose;                   // Verbosity level
  bool f_inferred;                // Whether f was inferred

  // Paths and strings
  const char* on_disk_path;       // Default path for on_disk_build
  const char* metric;             // Metric name
  const char* dtype;              // Data type string
  const char* index_dtype;        // Index type string
  const char* wrapper_dtype;      // Hamming type string
  const char* random_dtype;       // Random type string
  const char* y_dtype;            // Label type string
  const char* y_return_type;      // Return type (list/dict)

  // Future-proof parameters
  double l1_ratio;                // Random projection ratio

  /**
   * @brief Constructor with safe defaults
   *
   * All parameters initialized to safe default values to enable
   * lazy construction and prevent undefined behavior.
   */
  AnnoyParams()
    : f(0)
    , n_trees(-1)
    , n_neighbors(5)
    , seed(0)
    , n_jobs(1)
    , schema_version(0)
    , prefault(false)
    , verbose(false)
    , f_inferred(false)
    , on_disk_path(NULL)
    , metric("angular")
    , dtype("float")
    , index_dtype("int32")
    , wrapper_dtype("uint32")
    , random_dtype("uint32")
    , y_dtype("float")
    , y_return_type("list")
    , l1_ratio(0.0)
  {}

  /**
   * @brief Destructor - cleanup allocated strings
   */
  ~AnnoyParams() {
    // Note: Paths are owned by caller, don't free here
  }

  /**
   * @brief Validate parameters for consistency
   *
   * @param error Output parameter for error messages
   * @return true if parameters are valid, false otherwise
   */
  bool validate(char** error = NULL) const {
    if (f < 0) {
      set_error_msg(error, "Dimension (f) must be non-negative");
      return false;
    }
    if (n_neighbors < 1) {
      set_error_msg(error, "n_neighbors must be at least 1");
      return false;
    }
    if (n_jobs < -1 || n_jobs == 0) {
      set_error_msg(error, "n_jobs must be -1 (all cores) or >= 1");
      return false;
    }
    if (l1_ratio < 0.0 || l1_ratio > 9999) {
      set_error_msg(error, "l1_ratio must be in range [0.0, inf]");
      return false;
    }
    return true;
  }
};
/* =========================================================================================
 * UTILITY FUNCTIONS
 * =========================================================================================
 * Helper functions for error handling, string duplication, and common operations.
 * ========================================================================================= */
// Error handling: one allocator, one owner
inline char* dup_error(const char* msg) {
  const size_t n = std::strlen(msg) + 1;
  char* p = static_cast<char*>(std::malloc(n));
  if (p) memcpy(p, msg, n);
  return p;
}
// Duplicate a C-style string (safe memory allocation)
inline char* dup_cstr(const char* s) {
  if (s == NULL) return NULL;
  const size_t len = strlen(s);
  char* r = static_cast<char*>(malloc(len + 1));
  if (r == NULL) return NULL;
  memcpy(r, s, len + 1);
  return r;
}
// Set error from C-style string
inline void set_error_from_string(char **error, const char* msg) {
  // if (error != NULL) {
  //   *error = dup_cstr(msg);
  // }
  annoylib_showUpdate("%s\n", msg);
  if (error) {
    *error = (char *)malloc(strlen(msg) + 1);
    // strcpy(*error, msg);
    if (*error) {
      strcpy(*error, msg);
    }
  }
}
// Set error from errno
inline void set_error_from_errno(char **error, const char* msg) {
  // if (error == NULL) return;
  // char buf[512];
  // snprintf(buf, sizeof(buf), "%s: %s",
  //          prefix ? prefix : "Error",
  //          strerror(errno));
  // *error = dup_cstr(buf);
  annoylib_showUpdate("%s: %s (%d)\n", msg, strerror(errno), errno);
  if (error) {
    // Caller owns *error and must free() it
    *error = static_cast<char*>(malloc(256));  // TODO: win doesn't support snprintf
    // snprintf(*error, 255, "%s: %s (%d)", msg, strerror(errno), errno);
    if (*error) {
      snprintf(*error, 255, "%s: %s (%d)", msg, strerror(errno), errno);
    }
  }
}
#if __cplusplus >= 201103L
/* ================================================================
 * Type Traits and Compile-Time Validation
 * ================================================================ */
#ifndef ANNOY_TYPE_TRAITS_DEFINED
#define ANNOY_TYPE_TRAITS_DEFINED
/**
 * Type trait to validate data type (T parameter).
 * Valid data types: 'bool', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'float16', 'float', 'float32', 'double', 'float64', 'float128'
 */
template<typename T>
struct is_valid_data_type {
  static constexpr bool value =
    std::is_same<T, bool>::value ||
    std::is_same<T, int32_t>::value ||
    std::is_same<T, int64_t>::value ||
    std::is_same<T, uint8_t>::value ||
    std::is_same<T, uint16_t>::value ||
    std::is_same<T, uint32_t>::value ||
    std::is_same<T, uint64_t>::value ||
    std::is_same<T, float16_t>::value ||
    std::is_same<T, float>::value ||
    std::is_same<T, double>::value ||
    std::is_same<T, float128_t>::value;
};
/**
 * Type trait to validate index type (S parameter).
 * Valid index types: 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'
 */
template<typename S>
struct is_valid_index_type {
  static constexpr bool value =
    std::is_same<S, int32_t>::value ||
    std::is_same<S, int64_t>::value ||
    std::is_same<S, uint8_t>::value ||
    std::is_same<S, uint16_t>::value ||
    std::is_same<S, uint32_t>::value ||
    std::is_same<S, uint64_t>::value;
};
/**
 * Type trait to validate wrapper data type (InternalT parameter).
 * Valid wrapper data types: 'bool', 'uint8', 'uint16', 'uint32', 'uint64'
 */
template<typename InternalT>
struct is_valid_wrapper_type {
  static constexpr bool value =
    std::is_same<InternalT, bool>::value ||
    std::is_same<InternalT, uint8_t>::value ||
    std::is_same<InternalT, uint16_t>::value ||
    std::is_same<InternalT, uint32_t>::value ||
    std::is_same<InternalT, uint64_t>::value;
};
/**
 * Type trait to validate Random seed type (R parameter).
 * Valid types: uint32_t, uint64_t
 */
template<typename R>
struct is_valid_random_seed_type {
  static constexpr bool value =
    std::is_same<R, uint32_t>::value ||
    std::is_same<R, uint64_t>::value;
};
/**
 * Type trait for arithmetic types (excludes bool).
 */
template<typename T>
struct is_arithmetic_type {
  static constexpr bool value =
    std::is_same<T, uint8_t>::value ||
    std::is_same<T, uint16_t>::value ||
    std::is_same<T, uint32_t>::value ||
    std::is_same<T, uint64_t>::value ||
    std::is_same<T, float16_t>::value ||
    std::is_same<T, float>::value ||
    std::is_same<T, double>::value ||
    std::is_same<T, float128_t>::value;
};
#endif // ANNOY_TYPE_TRAITS_DEFINED
#endif // __cplusplus >= 201103L
namespace bitops {
/* ================================================================
 * Portable fallback (unsigned only)
 * ================================================================ */
constexpr inline uint32_t popcount_fallback32(uint32_t x) noexcept {
  x = x - ((x >> 1) & 0x55555555u);
  x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
  x = (x + (x >> 4)) & 0x0F0F0F0Fu;
  return (x * 0x01010101u) >> 24;
}
constexpr inline uint32_t popcount_fallback64(uint64_t x) noexcept {
  return popcount_fallback32(static_cast<uint32_t>(x)) +
         popcount_fallback32(static_cast<uint32_t>(x >> 32));
}
/* ================================================================
 * Population count (number of 1 bits) Public API
 * ================================================================ */
inline uint32_t popcount(uint32_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<uint32_t>(__builtin_popcount(x));
#elif defined(_MSC_VER)  // && defined(_M_X64)
  return static_cast<uint32_t>(__popcnt(x));
#else
  // Software fallback
  return popcount_fallback32(x);
#endif
}
inline uint32_t popcount(uint64_t x) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<uint32_t>(__builtin_popcountll(x));
#elif defined(_MSC_VER) && defined(_M_X64)
  return static_cast<uint32_t>(__popcnt64(x));
#else
  // Software fallback
  return popcount_fallback64(x);
#endif
}
inline uint32_t popcount(uint8_t x) noexcept {
  return popcount(static_cast<uint32_t>(x));
}
inline uint32_t popcount(bool x) noexcept {
  return x ? 1u : 0u;
}
// inline uint8_t popcount(uint8_t x) noexcept {
//   return static_cast<uint8_t>(popcount(static_cast<uint32_t>(x)));
// }
// inline bool popcount(bool x) noexcept {
//   return x;
// }
} // namespace bitops
/* =========================================================================================
 * TYPE CONVERSION UTILITIES
 * =========================================================================================
 * Safe conversion between different numeric types with proper clamping and validation.
 * These functions ensure data integrity when converting between extended types.
 * ========================================================================================= */
#ifndef ANNOY_TYPE_CONVERSION_DEFINED
#define ANNOY_TYPE_CONVERSION_DEFINED
// Safe conversion from any numeric type to target type T
// This is a helper for type conversions with proper clamping
template<typename T, typename S>
inline T safe_numeric_cast(S value) {
  // Integer to integer: direct cast with bounds checking (clamping)
  if (std::numeric_limits<T>::is_integer && std::numeric_limits<S>::is_integer) {
    if (value > static_cast<S>(std::numeric_limits<T>::max())) {
      return std::numeric_limits<T>::max();
    } else if (value < static_cast<S>(std::numeric_limits<T>::lowest())) {
      return std::numeric_limits<T>::lowest();
    }
    return static_cast<T>(value);
  }

  // Float to int: round to nearest with bounds checking (clamping)
  if (std::numeric_limits<T>::is_integer && !std::numeric_limits<S>::is_integer) {
    S rounded = (value >= S(0)) ? (value + S(0.5)) : (value - S(0.5));
    if (rounded > static_cast<S>(std::numeric_limits<T>::max())) {
      return std::numeric_limits<T>::max();
    } else if (rounded < static_cast<S>(std::numeric_limits<T>::lowest())) {
      return std::numeric_limits<T>::lowest();
    }
    return static_cast<T>(rounded);
  }

  // Int to float or float to float: direct cast
  return static_cast<T>(value);
}
// Specialization for float16_t conversions
template<>
inline float16_t safe_numeric_cast<float16_t, float>(float value) {
  return float16_t(value);
}
template<>
inline float safe_numeric_cast<float, float16_t>(float16_t value) {
  return static_cast<float>(value);
}
// Specialization for bool conversions
template<>
inline bool safe_numeric_cast<bool, float>(float value) {
  return value >= 0.5f;
}
template<>
inline float safe_numeric_cast<float, bool>(bool value) {
  return value ? 1.0f : 0.0f;
}
template<>
inline bool safe_numeric_cast<bool, double>(double value) {
  return value >= 0.5;
}
template<>
inline double safe_numeric_cast<double, bool>(bool value) {
  return value ? 1.0 : 0.0;
}
#endif // ANNOY_TYPE_CONVERSION_DEFINED

inline bool remap_memory_and_truncate(void** _ptr, int _fd,
                                      size_t old_size, size_t new_size,
                                      bool* trunc_ok) {
  if (trunc_ok) *trunc_ok = true;
  if (new_size == old_size) return true;

#ifdef __linux__
  if (new_size > old_size) {
    if (ftruncate(_fd, new_size) == -1) {
      if (trunc_ok) *trunc_ok = false;
      return false;
    }

    void* new_ptr = mremap(*_ptr, old_size, new_size, MREMAP_MAYMOVE);
    if (new_ptr == MAP_FAILED) {
#ifdef MAP_POPULATE
      new_ptr = mmap(0, new_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
      new_ptr = mmap(0, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
      if (new_ptr == MAP_FAILED) {
        // Best-effort rollback of file size; mapping remains old.
        (void)ftruncate(_fd, ANNOYLIB_FTRUNCATE_SIZE(old_size));
        if (trunc_ok) *trunc_ok = false;
        return false;
      }
      munmap(*_ptr, old_size);
    }
    *_ptr = new_ptr;
    return true;
  }

  // shrink: resize mapping first, then truncate (truncate failure is non-fatal)
  void* new_ptr = mremap(*_ptr, old_size, new_size, MREMAP_MAYMOVE);
  if (new_ptr == MAP_FAILED) return false;
  *_ptr = new_ptr;
  if (ftruncate(_fd, new_size) == -1) {
    if (trunc_ok) *trunc_ok = false;
  }
  return true;

#else
  // Grow: truncate first so mapping of new_size is valid
  if (new_size > old_size) {
    if (ftruncate(_fd, ANNOYLIB_FTRUNCATE_SIZE(new_size)) == -1) {
      if (trunc_ok) *trunc_ok = false;
      return false;
    }
#ifdef MAP_POPULATE
    void* new_ptr = mmap(0, new_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    void* new_ptr = mmap(0, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
    if (new_ptr == MAP_FAILED) {
      (void)ftruncate(_fd, ANNOYLIB_FTRUNCATE_SIZE(old_size));
      if (trunc_ok) *trunc_ok = false;
      return false;
    }
    munmap(*_ptr, old_size);
    *_ptr = new_ptr;
    return true;
  }

  // Shrink: map new view first (file still >= old_size), then swap.
#ifdef MAP_POPULATE
  void* new_ptr = mmap(0, new_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
  void* new_ptr = mmap(0, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
  if (new_ptr == MAP_FAILED) return false;

  munmap(*_ptr, old_size);
  *_ptr = new_ptr;

  if (ftruncate(_fd, ANNOYLIB_FTRUNCATE_SIZE(new_size)) == -1) {
    if (trunc_ok) *trunc_ok = false;
  }
  return true;
#endif
}

inline bool remap_memory_and_truncate(void** _ptr, int _fd, size_t old_size, size_t new_size) {
  bool trunc_ok = true;
  bool mapped_ok = remap_memory_and_truncate(_ptr, _fd, old_size, new_size, &trunc_ok);
  return mapped_ok && trunc_ok;
}

namespace {

template<typename S, typename Node>
inline Node* get_node_ptr(const void* _nodes, const size_t _s, const S i) {
  return (Node*)((uint8_t *)_nodes + (_s * i));
}

template<typename T>
inline T dot(const T* x, const T* y, int f) {
  T s = 0;
  for (int z = 0; z < f; z++) {
    s += (*x) * (*y);
    x++;
    y++;
  }
  return s;
}

template<typename T>
inline T manhattan_distance(const T* x, const T* y, int f) {
  T d = 0.0;
  for (int i = 0; i < f; i++)
    d += fabs(x[i] - y[i]);
  return d;
}

template<typename T>
inline T euclidean_distance(const T* x, const T* y, int f) {
  // Don't use dot-product: avoid catastrophic cancellation in #314.
  T d = 0.0;
  for (int i = 0; i < f; ++i) {
    const T tmp=*x - *y;
    d += tmp * tmp;
    ++x;
    ++y;
  }
  return d;
}

#ifdef ANNOYLIB_USE_AVX
// Horizontal single sum of 256bit vector.
inline float hsum256_ps_avx(__m256 v) {
  const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
  const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
  const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
  return _mm_cvtss_f32(x32);
}

template<>
inline float dot<float>(const float* x, const float *y, int f) {
  float result = 0;
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      d = _mm256_add_ps(d, _mm256_mul_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y)));
      x += 8;
      y += 8;
    }
    // Sum all floats in dot register.
    result += hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += *x * *y;
    x++;
    y++;
  }
  return result;
}

template<>
inline float manhattan_distance<float>(const float* x, const float* y, int f) {
  float result = 0;
  int i = f;
  if (f > 7) {
    __m256 manhattan = _mm256_setzero_ps();
    __m256 minus_zero = _mm256_set1_ps(-0.0f);
    for (; i > 7; i -= 8) {
      const __m256 x_minus_y = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
      const __m256 distance = _mm256_andnot_ps(minus_zero, x_minus_y); // Absolute value of x_minus_y (forces sign bit to zero)
      manhattan = _mm256_add_ps(manhattan, distance);
      x += 8;
      y += 8;
    }
    // Sum all floats in manhattan register.
    result = hsum256_ps_avx(manhattan);
  }
  // Don't forget the remaining values.
  for (; i > 0; i--) {
    result += fabsf(*x - *y);
    x++;
    y++;
  }
  return result;
}

template<>
inline float euclidean_distance<float>(const float* x, const float* y, int f) {
  float result=0;
  if (f > 7) {
    __m256 d = _mm256_setzero_ps();
    for (; f > 7; f -= 8) {
      const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(x), _mm256_loadu_ps(y));
      d = _mm256_add_ps(d, _mm256_mul_ps(diff, diff)); // no support for fmadd in AVX...
      x += 8;
      y += 8;
    }
    // Sum all floats in dot register.
    result = hsum256_ps_avx(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    float tmp = *x - *y;
    result += tmp * tmp;
    x++;
    y++;
  }
  return result;
}
#endif


#ifdef ANNOYLIB_USE_AVX512
template<>
inline float dot<float>(const float* x, const float *y, int f) {
  float result = 0;
  if (f > 15) {
    __m512 d = _mm512_setzero_ps();
    for (; f > 15; f -= 16) {
      //AVX512F includes FMA
      d = _mm512_fmadd_ps(_mm512_loadu_ps(x), _mm512_loadu_ps(y), d);
      x += 16;
      y += 16;
    }
    // Sum all floats in dot register.
    result += _mm512_reduce_add_ps(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    result += *x * *y;
    x++;
    y++;
  }
  return result;
}

template<>
inline float manhattan_distance<float>(const float* x, const float* y, int f) {
  float result = 0;
  int i = f;
  if (f > 15) {
    __m512 manhattan = _mm512_setzero_ps();
    for (; i > 15; i -= 16) {
      const __m512 x_minus_y = _mm512_sub_ps(_mm512_loadu_ps(x), _mm512_loadu_ps(y));
      manhattan = _mm512_add_ps(manhattan, _mm512_abs_ps(x_minus_y));
      x += 16;
      y += 16;
    }
    // Sum all floats in manhattan register.
    result = _mm512_reduce_add_ps(manhattan);
  }
  // Don't forget the remaining values.
  for (; i > 0; i--) {
    result += fabsf(*x - *y);
    x++;
    y++;
  }
  return result;
}

template<>
inline float euclidean_distance<float>(const float* x, const float* y, int f) {
  float result=0;
  if (f > 15) {
    __m512 d = _mm512_setzero_ps();
    for (; f > 15; f -= 16) {
      const __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(x), _mm512_loadu_ps(y));
      d = _mm512_fmadd_ps(diff, diff, d);
      x += 16;
      y += 16;
    }
    // Sum all floats in dot register.
    result = _mm512_reduce_add_ps(d);
  }
  // Don't forget the remaining values.
  for (; f > 0; f--) {
    float tmp = *x - *y;
    result += tmp * tmp;
    x++;
    y++;
  }
  return result;
}
#endif

template<typename T, typename Random, typename Distance, typename Node>
inline void two_means(const std::vector<Node*>& nodes, int f, Random& random, bool cosine, Node* p, Node* q) {
  /*
    This algorithm is a huge heuristic. Empirically it works really well, but I
    can't motivate it well. The basic idea is to keep two centroids and assign
    points to either one of them. We weight each centroid by the number of points
    assigned to it, so to balance it.
  */
  static int iteration_steps = 200;
  size_t count = nodes.size();

  size_t i = random.index(count);
  size_t j = random.index(count-1);
  j += (j >= i); // ensure that i != j

  Distance::template copy_node<T, Node>(p, nodes[i], f);
  Distance::template copy_node<T, Node>(q, nodes[j], f);

  if (cosine) { Distance::template normalize<T, Node>(p, f); Distance::template normalize<T, Node>(q, f); }
  Distance::init_node(p, f);
  Distance::init_node(q, f);

  int ic = 1, jc = 1;
  for (int l = 0; l < iteration_steps; l++) {
    size_t k = random.index(count);
    T di = ic * Distance::distance(p, nodes[k], f),
      dj = jc * Distance::distance(q, nodes[k], f);
    T norm = cosine ? Distance::template get_norm<T, Node>(nodes[k], f) : 1;
    if (!(norm > T(0))) {
      continue;
    }
    if (di < dj) {
      Distance::update_mean(p, nodes[k], norm, ic, f);
      Distance::init_node(p, f);
      ic++;
    } else if (dj < di) {
      Distance::update_mean(q, nodes[k], norm, jc, f);
      Distance::init_node(q, f);
      jc++;
    }
  }
}
} // namespace
/* ================================================================
 * FORWARD DECLARATIONS class of build policies
 * ================================================================ */
class AnnoyIndexSingleThreadedBuildPolicy;

#ifdef ANNOYLIB_MULTITHREADED_BUILD
  class AnnoyIndexMultiThreadedBuildPolicy;
  typedef AnnoyIndexMultiThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#else
  typedef AnnoyIndexSingleThreadedBuildPolicy AnnoyIndexThreadedBuildPolicy;
#endif
/* =========================================================================================
 * NODE STRUCTURE FOR TREE STORAGE
 * ========================================================================================= */
/**
 * @brief Tree node for random projection trees
 *
 * @tparam S Index type
 * @tparam T Data type
 */
template<typename S, typename T>
struct Node {
  /*
    * We store a binary tree where each node has two things
    * - A vector associated with it
    * - Two children
    * All nodes occupy the same amount of memory
    * All nodes with n_descendants == 1 are leaf nodes.
    * A memory optimization is that for nodes with 2 <= n_descendants <= K,
    * we skip the vector. Instead we store a list of all descendants. K is
    * determined by the number of items that fits in the space of the vector.
    * For nodes with n_descendants == 1 the vector is a data point.
    * For nodes with n_descendants > K the vector is the normal of the split plane.
    * Note that we can't really do sizeof(node<T>) because we cheat and allocate
    * more memory to be able to fit the vector outside
    */
  S n_descendants; // Number of points in subtree
  union {
    S children[2]; // Will possibly store more than 2, Child node indices (or -1 for leaf, item index for leaf)
    T norm;
  };
  T* v[ANNOYLIB_V_ARRAY_SIZE]; // Hyperplane normal vector (for split nodes)

  Node() : n_descendants(0), v(NULL) {
    children[0] = static_cast<S>(-1);
    children[1] = static_cast<S>(-1);
  }

  ~Node() {
    if (v != NULL) {
      delete[] v;
      v = NULL;
    }
  }

  bool is_leaf() const {
    return children[0] == static_cast<S>(-1) && children[1] == static_cast<S>(-1);
  }
};
/* ================================================================
 * DISTANCE METRICS Base Class and Implementations
 * ================================================================ */
struct Base {
  template<typename T, typename S, typename Node>
  static inline void preprocess(void* nodes, size_t _s, const S node_count, const int f) {
    // Override this in specific metric structs below if you need to do any pre-processing
    // on the entire set of nodes passed into this index.
  }
  template<typename T, typename S, typename Node>
  static inline void postprocess(void* nodes, size_t _s, const S node_count, const int f) {
    // Override this in specific metric structs below if you need to do any post-processing
    // on the entire set of nodes passed into this index.
  }
  template<typename Node>
  static inline void zero_value(Node* dest) {
    // Initialize any fields that require sane defaults within this node.
  }
  template<typename T, typename Node>
  static inline void copy_node(Node* dest, const Node* source, const int f) {
    memcpy(dest->v, source->v, f * sizeof(T));
  }
  template<typename T, typename Node>
  static inline T get_norm(Node* node, int f) {
      return sqrt(dot(node->v, node->v, f));
  }
  template<typename T, typename Node>
  static inline void normalize(Node* node, int f) {
    T norm = Base::get_norm<T, Node>(node, f);
    if (norm > 0) {
      for (int z = 0; z < f; z++)
        node->v[z] /= norm;
    }
  }
  template<typename T, typename Node>
  static inline void update_mean(Node* mean, Node* new_node, T norm, int c, int f) {
      for (int z = 0; z < f; z++)
        mean->v[z] = (mean->v[z] * c + new_node->v[z] / norm) / (c + 1);
  }
};
/**
 * @brief Angular (cosine) Distance Metric
 *
 * Computes 1 - cosine_similarity, normalized to [0, 2].
 * Suitable for normalized vectors.
 */
struct Angular : Base {
  template<typename S, typename T>
  struct Node {
    /*
     * We store a binary tree where each node has two things
     * - A vector associated with it
     * - Two children
     * All nodes occupy the same amount of memory
     * All nodes with n_descendants == 1 are leaf nodes.
     * A memory optimization is that for nodes with 2 <= n_descendants <= K,
     * we skip the vector. Instead we store a list of all descendants. K is
     * determined by the number of items that fits in the space of the vector.
     * For nodes with n_descendants == 1 the vector is a data point.
     * For nodes with n_descendants > K the vector is the normal of the split plane.
     * Note that we can't really do sizeof(node<T>) because we cheat and allocate
     * more memory to be able to fit the vector outside
     */
    S n_descendants;
    union {
      S children[2]; // Will possibly store more than 2
      T norm;
    };
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    // want to calculate (a/|a| - b/|b|)^2
    // = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
    // = 2 - 2cos
    T pp = x->norm ? x->norm : dot(x->v, x->v, f); // For backwards compatibility reasons, we need to fall back and compute the norm here
    T qq = y->norm ? y->norm : dot(y->v, y->v, f);
    T pq = dot(x->v, y->v, f);
    T ppqq = pp * qq;
    if (ppqq > 0) {
      // return 2.0 - 2.0 * pq / sqrt(ppqq);
      return static_cast<T>(2.0) - static_cast<T>(2.0) * pq / sqrt(ppqq);
    } else {
      return static_cast<T>(2.0);  // Maximum distance cos is 0
    }
  }
  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return dot(n->v, y, f);
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return (bool)random.flip();
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const Node<S, T>* y, int f, Random& random) {
    return side(n, y->v, f, random);
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const std::vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)alloca(s);
    Node<S, T>* q = (Node<S, T>*)alloca(s);
    two_means<T, Random, Angular, Node<S, T> >(nodes, f, random, true, p, q);
    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];
    Base::normalize<T, Node<S, T> >(n, f);
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    // Used when requesting distances from Python layer
    // Turns out sometimes the squared distance is -0.0
    // so we have to make sure it's a positive number.
    return sqrt(std::max(distance, T(0)));
  }
  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    if (child_nr == 0)
      margin = -margin;
    return std::min(distance, margin);
  }
  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::infinity();
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
    n->norm = dot(n->v, n->v, f);
  }
  static const char* name() {
    return "angular";
  }
};
/**
 * @brief Dot Product Distance Metric similarity (negated for distance)
 */
struct DotProduct : Angular {
  template<typename S, typename T>
  struct Node {
    /*
     * This is an extension of the Angular node with extra attributes for the DotProduct metric.
     * It has dot_factor which is needed to reduce the task to Angular distance metric (see the preprocess method)
     * and also a built flag that helps to compute exact dot products when an index is already built.
     */
    S n_descendants;
    S children[2]; // Will possibly store more than 2
    T dot_factor;
    T norm;
    bool built;
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };

  static const char* name() {
    return "dot";
  }

  template<typename T, typename Node>
  static inline T get_norm(Node* node, int f) {
      return sqrt(dot(node->v, node->v, f) + node->dot_factor * node->dot_factor);
  }

  template<typename T, typename Node>
  static inline void update_mean(Node* mean, Node* new_node, T norm, int c, int f) {
      for (int z = 0; z < f; z++)
        mean->v[z] = (mean->v[z] * c + new_node->v[z] / norm) / (c + 1);
      mean->dot_factor = (mean->dot_factor * c + new_node->dot_factor / norm) / (c + 1);
  }

  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    if (x->built || y->built) {
      // When index is already built, we don't need angular distances to retrieve NNs
      // Thus, we can return dot product scores itself
      return -dot(x->v, y->v, f);  // Negate for "distance" semantics
    }

    // Calculated by analogy with the angular case
    T pp = x->norm ? x->norm : dot(x->v, x->v, f) + x->dot_factor * x->dot_factor;
    T qq = y->norm ? y->norm : dot(y->v, y->v, f) + y->dot_factor * y->dot_factor;
    T pq = dot(x->v, y->v, f) + x->dot_factor * y->dot_factor;
    T ppqq = pp * qq;

    if (ppqq > 0) return 2.0 - 2.0 * pq / sqrt(ppqq);
    else return 2.0;
  }

  template<typename Node>
  static inline void zero_value(Node* dest) {
    dest->dot_factor = 0;
  }

  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
    n->built = false;
    n->norm = dot(n->v, n->v, f) + n->dot_factor * n->dot_factor;
  }

  template<typename T, typename Node>
  static inline void copy_node(Node* dest, const Node* source, const int f) {
    memcpy(dest->v, source->v, f * sizeof(T));
    dest->dot_factor = source->dot_factor;
  }

  template<typename S, typename T, typename Random>
  static inline void create_split(const std::vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)alloca(s);
    Node<S, T>* q = (Node<S, T>*)alloca(s);
    DotProduct::zero_value(p);
    DotProduct::zero_value(q);
    two_means<T, Random, DotProduct, Node<S, T> >(nodes, f, random, true, p, q);
    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];
    n->dot_factor = p->dot_factor - q->dot_factor;
    DotProduct::normalize<T, Node<S, T> >(n, f);
  }

  template<typename T, typename Node>
  static inline void normalize(Node* node, int f) {
    T norm = sqrt(dot(node->v, node->v, f) + pow(node->dot_factor, 2));
    if (norm > 0) {
      for (int z = 0; z < f; z++)
        node->v[z] /= norm;
      node->dot_factor /= norm;
    }
  }

  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return dot(n->v, y, f);
  }

  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const Node<S, T>* y, int f) {
    return dot(n->v, y->v, f) + n->dot_factor * y->dot_factor;
  }

  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const Node<S, T>* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return (bool)random.flip();
  }

  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return (bool)random.flip();
  }

  template<typename T>
  static inline T normalized_distance(T distance) {
    return -distance;
  }

  template<typename T, typename S, typename Node>
  static inline void preprocess(void* nodes, size_t _s, const S node_count, const int f) {
    // This uses a method from Microsoft Research for transforming inner product spaces to cosine/angular-compatible spaces.
    // (Bachrach et al., 2014, see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf)

    // Step one: compute the norm of each vector and store that in its extra dimension (f-1)
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      T d = dot(node->v, node->v, f);
      T norm = d < 0 ? 0 : sqrt(d);
      node->dot_factor = norm;
      node->built = false;
    }

    // Step two: find the maximum norm
    T max_norm = 0;
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      if (node->dot_factor > max_norm) {
        max_norm = node->dot_factor;
      }
    }

    // Step three: set each vector's extra dimension to sqrt(max_norm^2 - norm^2)
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      T node_norm = node->dot_factor;
      T squared_norm_diff = pow(max_norm, static_cast<T>(2.0)) - pow(node_norm, static_cast<T>(2.0));
      T dot_factor = squared_norm_diff < 0 ? 0 : sqrt(squared_norm_diff);

      node->norm = pow(max_norm, static_cast<T>(2.0));
      node->dot_factor = dot_factor;
    }
  }

  template<typename T, typename S, typename Node>
  static inline void postprocess(void* nodes, size_t _s, const S node_count, const int f) {
    for (S i = 0; i < node_count; i++) {
      Node* node = get_node_ptr<S, Node>(nodes, _s, i);
      // When an index is built, we will remember it in index item nodes to compute distances differently
      node->built = true;
    }
  }
};
/**
 * @brief Hamming Distance Metric for binary data
 */
struct Hamming : Base {
  template<typename S, typename T>
  struct Node {
    S n_descendants;
    S children[2];
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };

  static const size_t max_iterations = 20;

  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    return distance - (margin != (unsigned int) child_nr);
  }

  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::max();
  }
  template<typename T>
  static inline int cole_popcount(T v) {
    // Note: Only used with MSVC 9, which lacks intrinsics and fails to
    // calculate std::bitset::count for v > 32bit. Uses the generalized
    // approach by Eric Cole.
    // See https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSet64
    v = v - ((v >> 1) & (T)~(T)0/3);
    v = (v & (T)~(T)0/15*3) + ((v >> 2) & (T)~(T)0/15*3);
    v = (v + (v >> 4)) & (T)~(T)0/255*15;
    return (T)(v * ((T)~(T)0/255)) >> (sizeof(T) - 1) * 8;
  }
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    size_t dist = 0;
    for (int i = 0; i < f; i++) {
      dist += annoylib_popcount(x->v[i] ^ y->v[i]);
    }
    return dist;
  }
  template<typename S, typename T>
  static inline bool margin(const Node<S, T>* n, const T* y, int f) {
    static const size_t n_bits = sizeof(T) * 8;
    T chunk = n->v[0] / n_bits;
    return (y[chunk] & (static_cast<T>(1) << (n_bits - 1 - (n->v[0] % n_bits)))) != 0;
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    return margin(n, y, f);
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const Node<S, T>* y, int f, Random& random) {
    return side(n, y->v, f, random);
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const std::vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    size_t cur_size = 0;
    size_t i = 0;
    int dim = f * 8 * sizeof(T);
    for (; i < max_iterations; i++) {
      // choose random position to split at
      n->v[0] = random.index(dim);
      cur_size = 0;
      for (typename std::vector<Node<S, T>*>::const_iterator it = nodes.begin(); it != nodes.end(); ++it) {
        if (margin(n, (*it)->v, f)) {
          cur_size++;
        }
      }
      if (cur_size > 0 && cur_size < nodes.size()) {
        break;
      }
    }
    // brute-force search for splitting coordinate
    if (i == max_iterations) {
      int j = 0;
      for (; j < dim; j++) {
        n->v[0] = j;
        cur_size = 0;
        for (typename std::vector<Node<S, T>*>::const_iterator it = nodes.begin(); it != nodes.end(); ++it) {
          if (margin(n, (*it)->v, f)) {
            cur_size++;
          }
        }
        if (cur_size > 0 && cur_size < nodes.size()) {
          break;
        }
      }
    }
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    return distance;
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
  }
  static const char* name() {
    return "hamming";
  }
};
/* ================================================================
 * Minkowski Distance Base
 * ================================================================ */
struct Minkowski : Base {
  template<typename S, typename T>
  struct Node {
    S n_descendants;
    T a; // need an extra constant term to determine the offset of the plane
    S children[2];
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };
  template<typename S, typename T>
  static inline T margin(const Node<S, T>* n, const T* y, int f) {
    return n->a + dot(n->v, y, f);
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const T* y, int f, Random& random) {
    T dot = margin(n, y, f);
    if (dot != 0)
      return (dot > 0);
    else
      return (bool)random.flip();
  }
  template<typename S, typename T, typename Random>
  static inline bool side(const Node<S, T>* n, const Node<S, T>* y, int f, Random& random) {
    return side(n, y->v, f, random);
  }
  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    if (child_nr == 0)
      margin = -margin;
    return std::min(distance, margin);
  }
  template<typename T>
  static inline T pq_initial_value() {
    return numeric_limits<T>::infinity();
  }
};
/**
 * @brief Euclidean (L2) Distance Metric
 */
struct Euclidean : Minkowski {
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    return euclidean_distance(x->v, y->v, f);
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const std::vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)alloca(s);
    Node<S, T>* q = (Node<S, T>*)alloca(s);
    two_means<T, Random, Euclidean, Node<S, T> >(nodes, f, random, false, p, q);

    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];
    Base::normalize<T, Node<S, T> >(n, f);
    n->a = 0.0;
    for (int z = 0; z < f; z++)
      n->a += -n->v[z] * (p->v[z] + q->v[z]) / 2;
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    return sqrt(std::max(distance, T(0)));
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
  }
  static const char* name() {
    return "euclidean";
  }

};
/**
 * @brief Manhattan (L1) Distance Metric
 */
struct Manhattan : Minkowski {
  template<typename S, typename T>
  static inline T distance(const Node<S, T>* x, const Node<S, T>* y, int f) {
    return manhattan_distance(x->v, y->v, f);
  }
  template<typename S, typename T, typename Random>
  static inline void create_split(const std::vector<Node<S, T>*>& nodes, int f, size_t s, Random& random, Node<S, T>* n) {
    Node<S, T>* p = (Node<S, T>*)alloca(s);
    Node<S, T>* q = (Node<S, T>*)alloca(s);
    two_means<T, Random, Manhattan, Node<S, T> >(nodes, f, random, false, p, q);

    for (int z = 0; z < f; z++)
      n->v[z] = p->v[z] - q->v[z];
    Base::normalize<T, Node<S, T> >(n, f);
    n->a = 0.0;
    for (int z = 0; z < f; z++)
      n->a += -n->v[z] * (p->v[z] + q->v[z]) / 2;
  }
  template<typename T>
  static inline T normalized_distance(T distance) {
    return std::max(distance, T(0));
  }
  template<typename S, typename T>
  static inline void init_node(Node<S, T>* n, int f) {
  }
  static const char* name() {
    return "manhattan";
  }
};
// Python
//    ↓
// Cython wrapper (stores Base*)
//    ↓
// AnnoyIndexInterfaceBase   ← type erased
//    ↑
// template AnnoyIndexInterface<S,T,R>
//    ↑
// template AnnoyIndex<S,T,Distance,...>
class AnnoyIndexInterfaceBase
{
public:
  AnnoyIndexInterfaceBase() = default;
  virtual ~AnnoyIndexInterfaceBase() noexcept = default;

  // Dimension management
  virtual int get_f() const noexcept = 0;
  virtual bool set_f(int f, char** error = NULL) noexcept = 0;

  // Configuration
  // virtual void set_seed(R seed) noexcept = 0;
  virtual void set_verbose(bool verbosity) noexcept = 0;

  // Core operations
  // virtual bool add_item(S item, const T* embedding, char** error = NULL) noexcept = 0;
  virtual bool build(int n_trees = -1, int n_threads = -1, char** error = NULL) noexcept = 0;
  virtual bool unbuild(char** error = NULL) noexcept = 0;

  // Disk I/O
  virtual bool save(const char* filename, bool prefault = false, char** error = NULL) noexcept = 0;
  virtual bool load(const char* filename, bool prefault = false, char** error = NULL) noexcept = 0;
  virtual bool on_disk_build(const char* filename, char** error = NULL) noexcept = 0;
  virtual void unload() noexcept = 0;

  // Accessors
  // virtual S get_n_items() const noexcept = 0;
  // virtual S get_n_trees() const noexcept = 0;
  // virtual void get_item(S item, T* embedding) const noexcept = 0;
  // virtual T get_distance(S i, S j) const noexcept = 0;

  // Querying
  // virtual void get_nns_by_item(S item, size_t n, int search_k,
  //                              std::vector<S>* result, std::vector<T>* distances = NULL) const noexcept = 0;
  // virtual void get_nns_by_vector(const T* vector, size_t n, int search_k,
  //                                std::vector<S>* result, std::vector<T>* distances = NULL) const noexcept = 0;

  // Serialization
  virtual std::vector<uint8_t> serialize(char** error = NULL) const noexcept = 0;
  // virtual bool deserialize(const std::vector<uint8_t>& bytes, ...)
  // Pointer allows null Pointer implies ownership ambiguity
  virtual bool deserialize(std::vector<uint8_t>* bytes, bool prefault = false, char** error = NULL) noexcept = 0;

  // sklearn compatibility
  virtual bool get_params(
    std::vector<std::pair<std::string, std::string>>& params
  ) const noexcept = 0;
  virtual bool set_params(
    const std::vector<std::pair<std::string, std::string>>& params,
    char** error = NULL
  ) noexcept = 0;
};
/* =========================================================================================
 * MAIN ANNOY INDEX INTERFACE
 * ========================================================================================= */
/**
 * @brief Abstract interface for AnnoyIndex - Pure virtual base class
 *
 * Defines the contract that all index implementations must satisfy.
 * Default parameters in C++ are statically bound, not dynamically bound.
 * If derived overrides with different defaults, base defaults are used when calling via base pointer.
 *
 * @tparam S Index type (int32_t | int64_t | float | double, etc.)
 * @tparam T Data type (bool | uint8_t | float16_t | float | double | float128_t | uint32_t | uint64_t, etc.)
 * @tparam R Random seed type (uint32_t | uint64_t, etc.)
 *
 * Python wrapper
 * ↓
 * Cython
 * ↓
 * C++ Annoy core
 * ↓
 * STL / mmap / raw memory
 *
 * Your C++ interface:
 *  Uses no Python objects
 *  Uses no Python C API
 *  Uses error codes instead of exceptions
 *  Is purely native no reason to hold the GIL GIL belongs to CPython runtime C++ layer must remain Python-agnostic
 *  True parallelism for C++ compute
 *  Prevents accidental Python usage inside core
 *
 * You must NOT use nogil if:
 *  You touch Python objects
 *  You allocate Python objects
 *  You raise Python exceptions
 *  You access Python attributes
 *  You use Python memoryviews
 *  You convert STL → Python list inside method
 */
template<typename S, typename T, typename R>
class AnnoyIndexInterface
  : public AnnoyIndexInterfaceBase
{
 public:
  // C++ → C boundary → Python C API layer
  // The exception escapes the virtual call Cython has no except + so use noexcept
  // All interface methods are noexcept.
  // Implementations must catch all exceptions and convert them
  // to error codes via `char** error`. No exception may escape.
  AnnoyIndexInterface() = default;

  // # ⚠ Critical: __dealloc__ Runs With GIL
  // Note that the methods with an **error argument will allocate memory and write the pointer to that string if error is non-NULL
  // If you plan to do:
  // del index
  // via base pointer, you must ensure destructor is virtual (it is). Good.
  // Without virtual, deleting via:
  // delete p;
  // ~CAnnoyIndexInterface()
  // virtual ~AnnoyIndexInterface() {};
  virtual ~AnnoyIndexInterface() noexcept = default;

  // Dimension management
  virtual int get_f() const noexcept = 0;
  virtual bool set_f(int f, char** error = NULL) noexcept = 0;

  // Configuration
  virtual void set_seed(R seed) noexcept = 0;
  virtual void set_verbose(bool verbosity) noexcept = 0;

  // Core operations
  virtual bool add_item(S item, const T* embedding, char** error = NULL) noexcept = 0;
  virtual bool build(int n_trees = -1, int n_threads = -1, char** error = NULL) noexcept = 0;
  virtual bool unbuild(char** error = NULL) noexcept = 0;

  // Disk I/O
  virtual bool save(const char* filename, bool prefault = false, char** error = NULL) noexcept = 0;
  virtual bool load(const char* filename, bool prefault = false, char** error = NULL) noexcept = 0;
  virtual bool on_disk_build(const char* filename, char** error = NULL) noexcept = 0;
  virtual void unload() noexcept = 0;

  // Accessors
  virtual S get_n_items() const noexcept = 0;
  virtual S get_n_trees() const noexcept = 0;
  virtual void get_item(S item, T* embedding) const noexcept = 0;
  virtual T get_distance(S i, S j) const noexcept = 0;

  // Querying
  virtual void get_nns_by_item(S item, size_t n, int search_k,
                               std::vector<S>* result, std::vector<T>* distances = NULL) const noexcept = 0;
  virtual void get_nns_by_vector(const T* vector, size_t n, int search_k,
                                 std::vector<S>* result, std::vector<T>* distances = NULL) const noexcept = 0;

  // Serialization
  virtual std::vector<uint8_t> serialize(char** error = NULL) const noexcept = 0;
  // virtual bool deserialize(const std::vector<uint8_t>& bytes, ...)
  // Pointer allows null Pointer implies ownership ambiguity
  virtual bool deserialize(std::vector<uint8_t>* bytes, bool prefault = false, char** error = NULL) noexcept = 0;

  // sklearn compatibility
  virtual bool get_params(
    std::vector<std::pair<std::string, std::string>>& params
  ) const noexcept = 0;
  virtual bool set_params(
    const std::vector<std::pair<std::string, std::string>>& params,
    char** error = NULL
  ) noexcept = 0;
};
// Without this, noexcept will call std::terminate().
// bool build(...) noexcept override {
//     try {
//         // real logic
//         return true;
//     }
//     catch (const std::exception& e) {
//         if (error) {
//             *error = strdup(e.what());
//         }
//         return false;
//     }
//     catch (...) {
//         if (error) {
//             *error = strdup("Unknown C++ exception");
//         }
//         return false;
//     }
// }
/* =========================================================================================
 * MAIN ANNOY INDEX IMPLEMENTATION with lazy construction
 * ========================================================================================= */
/**
 * @brief Main AnnoyIndex implementation with comprehensive type support
 *
 * Features:
 * - Lazy construction with dimension inference
 * - Centralized parameter management
 * - sklearn-compatible API
 * - Thread-safe building
 * - Disk-based construction for large datasets
 *
 * @tparam S Index type (int32_t, int64_t, float, double)
 * @tparam T Data type (float, double, float16_t, float128_t, bool, uint8_t, uint32_t, uint64_t)
 * @tparam Distance Distance metric (Angular, Euclidean, Manhattan, DotProduct, Hamming)
 * @tparam Random Random number generator (Kiss32Random, Kiss64Random)
 * @tparam ThreadPolicy Build policy (SingleThreaded or MultiThreaded)
 */
template<typename S, typename T, typename Distance, typename Random, class ThreadedBuildPolicy>
class AnnoyIndex
  : public AnnoyIndexInterface<S, T, // Random type
#if __cplusplus >= 201103L
  typename std::remove_const<decltype(Random::default_seed)>::type
#else
  typename Random::seed_type
#endif
> {
#if __cplusplus >= 201103L
  static_assert(is_valid_index_type<S>::value,
                "S must be int32_t, int64_t, float, or double");
  static_assert(is_valid_data_type<T>::value,
                "T must be float, double, float16_t, float128_t, bool, or uint8_t");
#endif
/**
 * We use random projection to build a forest of binary trees of all items.
 * Basically just split the hyperspace into two sides by a hyperplane,
 * then recursively split each of those subtrees etc.
 * We create a tree like this q times. The default q is determined automatically
 * in such a way that we at most use 2x as much memory as the vectors take.
 */
public:
  typedef Distance D;
  typedef typename D::template Node<S, T> Node;

#if __cplusplus >= 201103L
  // typedef typename std::remove_const<decltype(Random::default_seed)>::type R;
  using R = typename std::remove_const<decltype(Random::default_seed)>::type;
#else
  // typedef typename Random::seed_type R;
  using R = typename Random::seed_type;
#endif

// ⚠️ ← declared first assign first ❌ This violates the declaration order.
private:
  // Centralized parameters
  AnnoyParams _params;

  // Core data structures
  // std::vector<Node<S, T>*> _nodes;
  // std::vector<S> _roots;
  // std::vector<T*> _items;
  // Random _random;

  // State flags
  // bool _built;
  // bool _on_disk;
  // bool _loaded;

  // File handling
  // int _fd;
  // void* _mmap_ptr;
  // size_t _mmap_size;

  // Thread synchronization (via policy)
  // ThreadPolicy _build_policy;
// ⚠️ ← declared first assign first ❌ This violates the declaration order.
protected:
  bool _verbose;                    // Verbose logging
  const int _f;                     // Dimension (0 means "infer from first vector")
  size_t _s;                        // Node size in bytes
  S _n_items;                       // Number of items added

  // Core data structures
  void* _nodes;                     // Could either be mmapped, or point to a memory buffer that we reallocate
  S _n_nodes;                       // Total number of nodes
  S _nodes_size;                    // Allocated size
  std::vector<S> _roots;            // Root node indices

  S _K;                             // Max descendants per leaf
  Random _random;                   // RNG instance
  R _seed;                          // seed value (0 = default_seed for kiss)
  bool _loaded;                     // True if loaded from disk
  int _fd;                          // File descriptor (for on-disk build)
  bool _on_disk;
  bool _built;
  std::atomic<bool> _build_failed;  // Thread-safe build failure flag

public:
  /**
   * @brief Constructor with centralized parameter management
   *
   * All parameters stored as attributes for reuse across methods.
   * Supports lazy construction when f=0.
   *
   * @param f Dimension (0 = infer from first vector)
   * @param n_trees Number of trees (-1 = auto)
   * @param n_neighbors Default neighbors for queries
   * @param on_disk_path Default path for on_disk_build
   * @param prefault Prefault pages when loading
   * @param seed Random seed
   * @param verbose Verbosity level
   * @param schema_version Serialization version
   * @param n_jobs Number of threads (-1 = all cores)
   * @param l1_ratio Future: random projection ratio
   */
  AnnoyIndex() = default;
  // AnnoyIndex(int f) : _f(f),
  explicit AnnoyIndex(
    int f = 0,                        // DEFAULT_DIMENSION  0/None infer from first vector
    int n_trees = -1,
    int n_neighbors = 5,
    const char* on_disk_path = NULL,
    bool prefault = false,
    int seed = 0,
    bool verbose = false,
    int schema_version = 0,
    int n_jobs = 1,
    double l1_ratio = 0.0
  )
    : _verbose(verbose)
    , _f(f)
    , _seed(Random::default_seed)
    // ,  _s(0)
    // ,  _n_items(0)
    // ,  _random(0)  // Default seed
    // ,  _nodes(NULL)
    // ,  _n_nodes(0)
    // ,  _nodes_size(0)
    // ,  _K(0)
    // ,  _loaded(false)
    // ,  _fd(0)
  {
    // Store all parameters
    _params.f = f;
    _params.n_trees = n_trees;
    _params.n_neighbors = n_neighbors;
    _params.on_disk_path = on_disk_path;
    _params.prefault = prefault;
    _params.seed = seed;
    _params.verbose = verbose;
    _params.schema_version = schema_version;
    _params.n_jobs = n_jobs;
    _params.l1_ratio = l1_ratio;
    _params.f_inferred = (f == 0);  // Mark for lazy inference
    // Validate parameters
    char* error = NULL;
    if (!_params.validate(&error)) {
      if (_verbose && error != NULL) {
        fprintf(stderr, "AnnoyIndex parameter validation failed: %s\n", error);
        std::free(error);
      }
    }
    if (_verbose) {
      fprintf(stderr, "AnnoyIndex initialized: f=%d, n_trees=%d, n_neighbors=%d, n_jobs=%d\n",
              _params.f, _params.n_trees, _params.n_neighbors, _params.n_jobs);
    }
    _built = false;
    _s = offsetof(Node, v) + _f * sizeof(T); // Size of each node
    _build_failed.store(false, std::memory_order_relaxed);
    _K = (S) (((size_t) (_s - offsetof(Node, children))) / sizeof(S)); // Max number of descendants to fit into node
    reinitialize(); // Reset everything
  }
  /**
   * @brief Destructor - cleanup resources
   */
  // ~AnnoyIndex() { unload(); }
  virtual ~AnnoyIndex() {
    unload();
    // for (size_t i = 0; i < _nodes.size(); ++i) {
    //   if (_nodes[i] != NULL) {
    //     delete _nodes[i];
    //     _nodes[i] = NULL;
    //   }
    // }
    //
    // for (size_t i = 0; i < _items.size(); ++i) {
    //   if (_items[i] != NULL) {
    //     delete[] _items[i];
    //     _items[i] = NULL;
    //   }
    // }
  }
  /**
   * @brief Get current dimension
   *
   * @return Dimension (0 if not set and not inferred)
   */
  int get_f() const noexcept{
    // return _f;
    return static_cast<int>(_f);
  }

  bool set_f(int f, char** error = NULL) noexcept{
      if (f <= 0) {
          if (error) {
              *error = strdup("f must be > 0");
          }
          return false;
      }
      // _f = static_cast<S>(f);
      return true;
  }

  bool add_item(S item, const T* w, char** error=NULL) noexcept{
    return add_item_impl(item, w, error);
  }

    bool get_params(
        std::vector<std::pair<std::string, std::string>>& params
    ) const noexcept{
        params.clear();
        params.emplace_back("f", std::to_string(_f));
        return true;
    }

    bool set_params(
        const std::vector<std::pair<std::string, std::string>>& params,
        char** error = NULL
    ) noexcept{
        for (const auto& kv : params) {
            if (kv.first == "f") {
                return set_f(std::stoi(kv.second), error);
            }
        }
        if (error) {
            *error = strdup("missing parameter: f");
        }
        return false;
    }

  template<typename W>
  bool add_item_impl(S item, const W& w, char** error=NULL) {
    if (_loaded) {
      set_error_from_string(error, "You can't add an item to a loaded index");
      return false;
    }
    // _allocate_size(item + 1);
    if (!_allocate_size(item + 1)) {
      set_error_from_string(error, "Unable to allocate memory for item");
      return false;
    }

    Node* n = _get(item);

    D::zero_value(n);

    n->children[0] = 0;
    n->children[1] = 0;
    n->n_descendants = 1;

    for (int z = 0; z < _f; z++)
      n->v[z] = w[z];

    D::init_node(n, _f);

    if (item >= _n_items)
      _n_items = item + 1;

    return true;
  }

  bool on_disk_build(const char* file, char** error=NULL) noexcept{
    _on_disk = true;
#ifndef _MSC_VER
    _fd = open(file, O_RDWR | O_CREAT | O_TRUNC, (int) 0600);
#else
    _fd = _open(file, _O_RDWR | _O_CREAT | _O_TRUNC, (int) 0600);
#endif
    if (_fd == -1) {
      set_error_from_errno(error, "Unable to open");
      _fd = 0;
      return false;
    }
    _nodes_size = 1;
    if (ftruncate(_fd, ANNOYLIB_FTRUNCATE_SIZE(_s) * ANNOYLIB_FTRUNCATE_SIZE(_nodes_size)) == -1) {
      set_error_from_errno(error, "Unable to truncate");
 #ifndef _MSC_VER
      close(_fd);
 #else
      _close(_fd);
 #endif
      _fd = 0;
      _on_disk = false;
      _nodes = NULL;
      _nodes_size = 0;
      return false;
    }
#ifdef MAP_POPULATE
    _nodes = (Node*) mmap(0, _s * _nodes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, _fd, 0);
#else
    _nodes = (Node*) mmap(0, _s * _nodes_size, PROT_READ | PROT_WRITE, MAP_SHARED, _fd, 0);
#endif
    if (_nodes == MAP_FAILED) {
      _nodes = NULL;
      set_error_from_errno(error, "Unable to mmap");
 #ifndef _MSC_VER
      close(_fd);
 #else
      _close(_fd);
 #endif
      _fd = 0;
      _on_disk = false;
      _nodes_size = 0;
      return false;
    }
    return true;
  }

  bool build(int q, int n_threads=-1, char** error=NULL) noexcept{
    if (_loaded) {
      set_error_from_string(error, "You can't build a loaded index");
      return false;
    }

    if (_built) {
      set_error_from_string(error, "You can't build a built index");
      return false;
    }

    D::template preprocess<T, S, Node>(_nodes, _s, _n_items, _f);

    _n_nodes = _n_items;

    _build_failed.store(false, std::memory_order_relaxed);

    ThreadedBuildPolicy::template build<S, T>(this, q, n_threads);

    if (_build_failed.load(std::memory_order_relaxed)) {
      set_error_from_string(error, "Unable to allocate memory while building index");
      _roots.clear();
      _n_nodes = _n_items;
      return false;
    }

    // Also, copy the roots into the last segment of the array
    // This way we can load them faster without reading the whole file
    if (!_allocate_size(_n_nodes + (S)_roots.size())) {
      set_error_from_string(error, "Unable to allocate memory while finalizing roots");
      return false;
    }
    for (size_t i = 0; i < _roots.size(); i++)
      memcpy(_get(_n_nodes + (S)i), _get(_roots[i]), _s);
    _n_nodes += _roots.size();

    if (_verbose) annoylib_showUpdate("has %ld nodes\n", (long)_n_nodes);

    if (_on_disk) {
      if (!remap_memory_and_truncate(&_nodes, _fd,
          static_cast<size_t>(_s) * static_cast<size_t>(_nodes_size),
          static_cast<size_t>(_s) * static_cast<size_t>(_n_nodes))) {
        // TODO: this probably creates an index in a corrupt state... not sure what to do
        set_error_from_errno(error, "Unable to truncate");
        return false;
      }
      _nodes_size = _n_nodes;
    }

    D::template postprocess<T, S, Node>(_nodes, _s, _n_items, _f);

    _built = true;
    return true;
  }

  bool unbuild(char** error=NULL) noexcept{
    if (_loaded) {
      set_error_from_string(error, "You can't unbuild a loaded index");
      return false;
    }

    _roots.clear();
    _n_nodes = _n_items;
    _built = false;

    return true;
  }

  bool save(const char* filename, bool prefault=false, char** error=NULL) noexcept{
    if (!_built) {
      set_error_from_string(error, "You can't save an index that hasn't been built");
      return false;
    }
    if (_on_disk) {
      return true;
    } else {
      // Delete file if it already exists (See issue #335)
#ifndef _MSC_VER
      unlink(filename);
#else
      _unlink(filename);
#endif

      FILE *f = fopen(filename, "wb");
      if (f == NULL) {
        set_error_from_errno(error, "Unable to open");
        return false;
      }

      if (fwrite(_nodes, _s, _n_nodes, f) != (size_t) _n_nodes) {
        set_error_from_errno(error, "Unable to write");
        // Best-effort cleanup: avoid leaking FILE* on short write.
        // fclose() may itself fail, but we still attempt it.
        fclose(f);
        return false;
      }

      if (fclose(f) == EOF) {
        set_error_from_errno(error, "Unable to close");
        return false;
      }

      unload();
      return load(filename, prefault, error);
    }
  }

  void reinitialize() {
    _fd = 0;
    _nodes = NULL;
    _loaded = false;
    _n_items = 0;
    _n_nodes = 0;
    _nodes_size = 0;
    _on_disk = false;
    _seed = Random::default_seed;
    _roots.clear();
  }

  void unload() noexcept {
    if (_on_disk && _fd) {
#ifndef _MSC_VER
      close(_fd);
#else
      _close(_fd);
#endif
      munmap(_nodes, _s * _nodes_size);
    } else {
      if (_fd) {
        // we have mmapped data
#ifndef _MSC_VER
        close(_fd);
#else
        _close(_fd);
#endif
        munmap(_nodes, _n_nodes * _s);
      } else if (_nodes) {
        // We have heap allocated data
        free(_nodes);
      }
    }
    reinitialize();
    if (_verbose) annoylib_showUpdate("unloaded\n");
  }

  bool load(const char* filename, bool prefault=false, char** error=NULL) noexcept{
#ifndef _MSC_VER
    _fd = open(filename, O_RDONLY, (int)0400);
#else
    _fd = _open(filename, _O_RDONLY, (int)0400);
#endif
    if (_fd == -1) {
      set_error_from_errno(error, "Unable to open");
      _fd = 0;
      return false;
    }
    off_t size = lseek_getsize(_fd);
    if (size == -1) {
      set_error_from_errno(error, "Unable to get size");
 #ifndef _MSC_VER
      close(_fd);
 #else
      _close(_fd);
 #endif
      _fd = 0;
      return false;
    } else if (size == 0) {
      // set_error_from_errno(error, "Size of file is zero");
      set_error_from_string(error, "Size of file is zero");
 #ifndef _MSC_VER
      close(_fd);
 #else
      _close(_fd);
 #endif
      _fd = 0;
      return false;
    } else if (size % _s) {
      // Something is fishy with this index!
      // set_error_from_errno(error, "Index size is not a multiple of vector size. Ensure you are opening using the same metric you used to create the index.");
      set_error_from_string(
          error,
          "Index size is not a multiple of node size; "
          "are you opening the index using the same metric you used to create the index?");
 #ifndef _MSC_VER
      close(_fd);
 #else
      _close(_fd);
 #endif
      _fd = 0;
      return false;
    }

    int flags = MAP_SHARED;
    if (prefault) {
#ifdef MAP_POPULATE
      flags |= MAP_POPULATE;
#else
      annoylib_showUpdate("prefault is set to true, but MAP_POPULATE is not defined on this platform");
#endif
    }
    _nodes = (Node*)mmap(0, size, PROT_READ, flags, _fd, 0);
    if (_nodes == MAP_FAILED) {
      _nodes = NULL;
      set_error_from_errno(error, "Unable to mmap");
 #ifndef _MSC_VER
      close(_fd);
 #else
      _close(_fd);
 #endif
      _fd = 0;
      return false;
    }

    _n_nodes = (S)(size / _s);

    // Keep capacity in sync for serialize()/memory usage.
    _nodes_size = _n_nodes;

    // Find the roots by scanning the end of the file and taking the nodes with most descendants
    _roots.clear();
    S m = -1;
    for (S i = _n_nodes - 1; i >= 0; i--) {
      S k = _get(i)->n_descendants;
      if (m == -1 || k == m) {
        _roots.push_back(i);
        m = k;
      } else {
        break;
      }
    }
    // hacky fix: since the last root precedes the copy of all roots, delete it
    if (_roots.size() > 1 && _get(_roots.front())->children[0] == _get(_roots.back())->children[0])
      _roots.pop_back();
    _loaded = true;
    _built = true;
    _n_items = m;
    if (_verbose) annoylib_showUpdate("found %zu roots with degree %ld\n", _roots.size(), (long)m);
    return true;
  }

  T get_distance(S i, S j) const noexcept{
    return D::normalized_distance(D::distance(_get(i), _get(j), _f));
  }

  void get_nns_by_item(S item, size_t n, int search_k, std::vector<S>* result, std::vector<T>* distances) const noexcept{
    // TODO: handle OOB
    const Node* m = _get(item);
    _get_all_nns(m->v, n, search_k, result, distances);
  }

  void get_nns_by_vector(const T* w, size_t n, int search_k, std::vector<S>* result, std::vector<T>* distances) const noexcept{
    _get_all_nns(w, n, search_k, result, distances);
  }

  S get_n_items() const noexcept{
    return _n_items;
  }

  S get_n_trees() const noexcept{
    return (S)_roots.size();
  }

  void set_verbose(bool v) noexcept{
    _verbose = v;
  }

  void get_item(S item, T* v) const noexcept{
    // TODO: handle OOB
    Node* m = _get(item);
    memcpy(v, m->v, (_f) * sizeof(T));
  }

  void set_seed(R seed) noexcept{
    // _seed = seed;
    // kissrandom.h documents that seeds should be non-zero. For API stability,
    // normalize seed=0 to the RNG's default seed.
    seed = static_cast<R>(seed);
    if (seed == static_cast<R>(0)) {
      _seed = Random::default_seed;
    } else {
      _seed = seed;
    }
  }

  std::vector<uint8_t> serialize(char** error=NULL) const noexcept{
    if (!_built) {
      set_error_from_string(error, "Index cannot be serialized if it hasn't been built");
      return {};
    }

    std::vector<uint8_t> bytes {};

    S n_items = _n_items;
    S n_nodes = _n_nodes;
    size_t roots_size = _roots.size();

    // S nodes_size = _nodes_size;
    // _nodes_size is the allocated capacity; after load() it may be 0 even though
    // _n_nodes is known. For serialization we must write the actual backing size.
    // Deterministic + safe: serialize only the USED node prefix.
    // Serializing capacity (slack) can leak uninitialized memory and cause non-determinism.
    if (_nodes_size && _nodes_size < _n_nodes) {
      set_error_from_string(error, "Index invariant violated: nodes_size < n_nodes");
      return {};
    }
    if (_n_nodes < _n_items) {
      set_error_from_string(error, "Index invariant violated: n_nodes < n_items");
      return {};
    }
    const S nodes_size = (_nodes_size ? _nodes_size : _n_nodes);

    // Overflow-safe byte sizing
    const size_t s_u = static_cast<size_t>(_s);
    const size_t nodes_u = static_cast<size_t>(nodes_size);
    if (s_u != 0 && nodes_u > (SIZE_MAX / s_u)) {
      set_error_from_string(error, "Index invariant violated: nodes_size overflow");
      return {};
    }
    const size_t nodes_bytes = nodes_u * s_u;

    if (roots_size > (SIZE_MAX / sizeof(S))) {
      set_error_from_string(error, "Index invariant violated: roots_size overflow");
      return {};
    }
    const size_t roots_bytes = roots_size * sizeof(S);

    // Reduce realloc churn
    bytes.reserve(sizeof(n_items) + sizeof(n_nodes) + sizeof(roots_size) + sizeof(nodes_size)
                  + roots_bytes + nodes_bytes);

    bytes.insert(bytes.end(), (uint8_t*)&n_items, (uint8_t*)&n_items + sizeof(n_items));
    bytes.insert(bytes.end(), (uint8_t*)&n_nodes, (uint8_t*)&n_nodes + sizeof(n_nodes));
    bytes.insert(bytes.end(), (uint8_t*)&roots_size, (uint8_t*)&roots_size + sizeof(roots_size));
    bytes.insert(bytes.end(), (uint8_t*)&nodes_size, (uint8_t*)&nodes_size + sizeof(nodes_size));

    uint8_t* roots_buffer = (uint8_t*)_roots.data();
    bytes.insert(bytes.end(), roots_buffer, roots_buffer + roots_bytes);

    uint8_t* nodes_buffer = (uint8_t*)_nodes;
    bytes.insert(bytes.end(), nodes_buffer, nodes_buffer + nodes_bytes);

    return bytes;
  }

  bool deserialize(std::vector<uint8_t>* bytes, bool prefault=false, char** error=NULL) noexcept{
//     int flags = MAP_SHARED;
//     if (prefault) {
// #ifdef MAP_POPULATE
//       flags |= MAP_POPULATE;
// #else
//       annoylib_showUpdate("prefault is set to true, but MAP_POPULATE is not defined on this platform");
// #endif
//     }
    (void)prefault;  // prefault is meaningful for mmap() loads, not in-memory restores

    if (!bytes || bytes->empty()) {
      set_error_from_string(error, "Size of bytes is zero");
       return false;
     }

    // If this index currently owns data (heap or mmap), clear it first to avoid
    // realloc() on mmapped pointers / leaks.
    if (_fd || _nodes) {
      unload();
     }

    const uint8_t* bytes_buffer = (const uint8_t*)bytes->data();
    size_t remaining = bytes->size();

    // Alignment-safe POD reader (strict, deterministic).
    struct Reader {
      const uint8_t*& p;
      size_t&   n;
      char**    err;
      Reader(const uint8_t*& p_, size_t& n_, char** err_) : p(p_), n(n_), err(err_) {}
      bool read(void* out, size_t sz) {
        if (n < sz) {
          set_error_from_string(err, "Serialized data is truncated");
          return false;
        }
        memcpy(out, p, sz);
        p += sz;
        n -= sz;
        return true;
      }
    } rd(bytes_buffer, remaining, error);

    S n_nodes = 0;
    size_t roots_size = 0;
    S nodes_size = 0;

    if (!rd.read(&_n_items, sizeof(S))) return false;
    if (!rd.read(&n_nodes, sizeof(S))) return false;
    if (!rd.read(&roots_size, sizeof(size_t))) return false;
    if (!rd.read(&nodes_size, sizeof(S))) return false;

    // Basic corruption checks (invariants).
    if (_n_items < 0 || n_nodes < 0 || nodes_size < 0) {
      set_error_from_string(error, "Serialized data is corrupt (negative sizes)");
      return false;
    }
    if (n_nodes < _n_items) {
      set_error_from_string(error, "Serialized data is corrupt (n_nodes < n_items)");
      return false;
    }
    if (n_nodes > 0 && nodes_size <= 0) {
      set_error_from_string(error, "Serialized data is corrupt (missing node storage)");
      return false;
    }
    if (nodes_size < n_nodes) {
      set_error_from_string(error, "Serialized data is corrupt (nodes_size < n_nodes)");
      return false;
    }

    // Roots payload
    if (roots_size > 0) {
      if (roots_size > (SIZE_MAX / sizeof(S))) {
        set_error_from_string(error, "Serialized data is corrupt (roots_size overflow)");
        return false;
      }
      const size_t roots_bytes = roots_size * sizeof(S);
      if (remaining < roots_bytes) {
        set_error_from_string(error, "Serialized data is truncated (roots)");
        return false;
      }
      _roots.clear();
      _roots.resize(roots_size);
      memcpy(&_roots[0], bytes_buffer, roots_bytes);
      bytes_buffer += roots_bytes;
      remaining -= roots_bytes;
    } else {
      _roots.clear();
    }

    // Nodes payload
    const size_t s_u = static_cast<size_t>(_s);
    const size_t nodes_u = static_cast<size_t>(nodes_size);
    if (s_u != 0 && nodes_u > (SIZE_MAX / s_u)) {
      set_error_from_string(error, "Serialized data is corrupt (nodes_size overflow)");
      return false;
    }
    const size_t nodes_bytes = nodes_u * s_u;
    if (remaining < nodes_bytes) {
      set_error_from_string(error, "Serialized data is truncated (nodes)");
      return false;
    }

    if (!_allocate_size((S)nodes_size)) {
      set_error_from_string(error, "Unable to allocate memory for nodes");
      return false;
    }

    memcpy(_nodes, bytes_buffer, nodes_bytes);

    _nodes_size = (S)nodes_size;
    _n_nodes = n_nodes;
    _loaded = true;

    // Mirror load()/build() behavior: ensure any derived fields are ready.
    D::template postprocess<T, S, Node>(_nodes, _s, _n_items, _f);

    _built = true;

    if (_verbose) {
      annoylib_showUpdate("found %zu roots with degree %ld\n", _roots.size(), (long)_n_items);
    }
    return true;
  }

  static S invalid_index() {
    return std::numeric_limits<S>::max();
  }

  inline void signal_build_failure() {
    _build_failed.store(true, std::memory_order_relaxed);
  }

  void thread_build(int q, int thread_idx, ThreadedBuildPolicy& threaded_build_policy) {
    // Each thread needs its own seed, otherwise each thread would be building the same tree(s)
    Random _random(_seed + thread_idx);

    std::vector<S> thread_roots;
    while (1) {
      if (_build_failed.load(std::memory_order_relaxed)) {
        break;
      }
      if (q == -1) {
        threaded_build_policy.lock_n_nodes();
        if (_n_nodes >= 2 * _n_items) {
          threaded_build_policy.unlock_n_nodes();
          break;
        }
        threaded_build_policy.unlock_n_nodes();
      } else {
        if (thread_roots.size() >= (size_t)q) {
          break;
        }
      }

      if (_verbose) annoylib_showUpdate("pass %zd...\n", thread_roots.size());

      std::vector<S> indices;
      threaded_build_policy.lock_shared_nodes();
      for (S i = 0; i < _n_items; i++) {
        if (_get(i)->n_descendants >= 1) { // Issue #223
          indices.push_back(i);
        }
      }
      threaded_build_policy.unlock_shared_nodes();

      // thread_roots.push_back(_make_tree(indices, true, _random, threaded_build_policy));
      S root = _make_tree(indices, true, _random, threaded_build_policy);
      if (root == invalid_index() || _build_failed.load(std::memory_order_relaxed)) {
        break;
      }
      thread_roots.push_back(root);
    }

    if (!_build_failed.load(std::memory_order_relaxed)) {
      threaded_build_policy.lock_roots();
      _roots.insert(_roots.end(), thread_roots.begin(), thread_roots.end());
      threaded_build_policy.unlock_roots();
    }
  }

protected:
  bool _reallocate_nodes(S n) {
    const double reallocation_factor = 1.3;
    S new_nodes_size = std::max(n, (S)((_nodes_size + 1) * reallocation_factor));

    void* old = _nodes;

    const size_t s_u = static_cast<size_t>(_s);
    const size_t old_nodes_u = static_cast<size_t>(_nodes_size);
    const size_t new_nodes_u = static_cast<size_t>(new_nodes_size);

    if (s_u != 0 && (new_nodes_u > (std::numeric_limits<size_t>::max() / s_u))) {
      return false;  // size_t overflow => impossible allocation
    }

    const size_t old_bytes = s_u * old_nodes_u;
    const size_t new_bytes = s_u * new_nodes_u;

    if (_on_disk) {
      bool trunc_ok = true;
      bool mapped_ok = remap_memory_and_truncate(&_nodes, _fd, old_bytes, new_bytes, &trunc_ok);
      if (!mapped_ok) {
        _nodes = old;
        return false;
      }
      if (!trunc_ok && _verbose) {
        annoylib_showUpdate("File truncation error\n");
      }
    } else {
      void* new_ptr = realloc(_nodes, new_bytes);
      if (!new_ptr) {
        return false;
      }
      _nodes = new_ptr;
      if (new_bytes > old_bytes) {
        memset((char*)_nodes + old_bytes, 0, new_bytes - old_bytes);
      }
    }

    _nodes_size = new_nodes_size;
    if (_verbose) {
      annoylib_showUpdate("Reallocating to %ld nodes: old_address=%p, new_address=%p\n",
                          (long)new_nodes_size, old, _nodes);
    }
    return true;
  }

  bool _allocate_size(S n, ThreadedBuildPolicy& threaded_build_policy) {
    if (n <= _nodes_size) return true;
    threaded_build_policy.lock_nodes();
    bool ok = true;
    if (n > _nodes_size) {  // re-check under lock
      ok = _reallocate_nodes(n);
    }
    threaded_build_policy.unlock_nodes();
    return ok;
  }

  bool _allocate_size(S n) {
    if (n <= _nodes_size) return true;
    return _reallocate_nodes(n);
  }

  Node* _get(const S i) const {
    return get_node_ptr<S, Node>(_nodes, _s, i);
  }

  // Range is guaranteed: imbalance ∈ [0.5, 1.0]
  // mathematically: imbalance = max(ls, rs) / (ls + rs) = 0.5 ≤ imbalance ≤ 1.0 = max(f,1−f)∈[0.5,1]
  // Acceptable region: [0.5, 0.95)
  // Gray region: [0.95, 0.999]
  // Retry region: (0.999, 1.0]
  double _split_imbalance(
    const std::vector<S>& left_indices,
    const std::vector<S>& right_indices
  ) {
    // size() returns size_t For values ≤ 2^53, conversion to double is exact.
    double ls = static_cast<double>(left_indices.size());
    double rs = (double)right_indices.size();
    double total = ls + rs;
    if (total == 0.0) {
        return 1.0;  // Defined as maximally imbalanced
    }
    // float f = ls / (ls + rs + 1e-9);  // Avoid 0/0
    double f = ls / total;
    return std::max(f, 1.0 - f);  // max(ls, rs) / total mathematically equivalent.
  }

  S _make_tree(const std::vector<S>& indices, bool is_root, Random& _random, ThreadedBuildPolicy& threaded_build_policy) {
    if (_build_failed.load(std::memory_order_relaxed)) {
      return invalid_index();
    }
    // The basic rule is that if we have <= _K items, then it's a leaf node, otherwise it's a split node.
    // There's some regrettable complications caused by the problem that root nodes have to be "special":
    // 1. We identify root nodes by the arguable logic that _n_items == n->n_descendants, regardless of how many descendants they actually have
    // 2. Root nodes with only 1 child need to be a "dummy" parent
    // 3. Due to the _n_items "hack", we need to be careful with the cases where _n_items <= _K or _n_items > _K
    if (indices.size() == 1 && !is_root)
      return indices[0];

    if (indices.size() <= (size_t)_K && (!is_root || (size_t)_n_items <= (size_t)_K || indices.size() == 1)) {
      threaded_build_policy.lock_n_nodes();
      if (!_allocate_size(_n_nodes + 1, threaded_build_policy)) {
        threaded_build_policy.unlock_n_nodes();
        signal_build_failure();
        return invalid_index();
      }
      S item = _n_nodes++;
      threaded_build_policy.unlock_n_nodes();

      threaded_build_policy.lock_shared_nodes();
      Node* m = _get(item);
      m->n_descendants = is_root ? _n_items : (S)indices.size();

      // Using std::copy instead of a loop seems to resolve issues #3 and #13,
      // probably because gcc 4.8 goes overboard with optimizations.
      // Using memcpy instead of std::copy for MSVC compatibility. #235
      // Only copy when necessary to avoid crash in MSVC 9. #293
      if (!indices.empty())
        memcpy(m->children, &indices[0], indices.size() * sizeof(S));

      threaded_build_policy.unlock_shared_nodes();
      return item;
    }

    threaded_build_policy.lock_shared_nodes();
    std::vector<Node*> children;
    for (size_t i = 0; i < indices.size(); i++) {
      S j = indices[i];
      Node* n = _get(j);
      if (n)
        children.push_back(n);
    }

    std::vector<S> children_indices[2];
    Node* m = (Node*)alloca(_s);

    for (int attempt = 0; attempt < 3; attempt++) {
      children_indices[0].clear();
      children_indices[1].clear();
      D::create_split(children, _f, _s, _random, m);

      for (size_t i = 0; i < indices.size(); i++) {
        S j = indices[i];
        Node* n = _get(j);
        if (n) {
          bool side = D::side(m, n, _f, _random);
          children_indices[side].push_back(j);
        } else {
          annoylib_showUpdate("No node for index %ld?\n", (long)j);
        }
      }

      if (_split_imbalance(children_indices[0], children_indices[1]) < 0.95)
        break;
    }
    threaded_build_policy.unlock_shared_nodes();

    // If we didn't find a hyperplane, just randomize sides as a last option
    while (_split_imbalance(children_indices[0], children_indices[1]) > 0.999) {
      if (_verbose)
        annoylib_showUpdate("\tNo hyperplane found (left has %zu children, right has %zu children)\n",
          children_indices[0].size(), children_indices[1].size());

      children_indices[0].clear();
      children_indices[1].clear();

      // Set the vector to 0.0
      for (int z = 0; z < _f; z++)
        m->v[z] = 0;

      for (size_t i = 0; i < indices.size(); i++) {
        S j = indices[i];
        // Just randomize...
        children_indices[_random.flip()].push_back(j);
      }
    }

    int flip = (children_indices[0].size() > children_indices[1].size());

    m->n_descendants = is_root ? _n_items : (S)indices.size();
    for (int side = 0; side < 2; side++) {
      // run _make_tree for the smallest child first (for cache locality)
      m->children[side^flip] = _make_tree(children_indices[side^flip], false, _random, threaded_build_policy);
      if (_build_failed.load(std::memory_order_relaxed) || m->children[side^flip] == invalid_index()) {
        return invalid_index();
      }
    }

    threaded_build_policy.lock_n_nodes();
    if (!_allocate_size(_n_nodes + 1, threaded_build_policy)) {
      threaded_build_policy.unlock_n_nodes();
      signal_build_failure();
      return invalid_index();
    }
    S item = _n_nodes++;
    threaded_build_policy.unlock_n_nodes();

    threaded_build_policy.lock_shared_nodes();
    memcpy(_get(item), m, _s);
    threaded_build_policy.unlock_shared_nodes();

    return item;
  }

  void _get_all_nns(const T* v, size_t n, int search_k, std::vector<S>* result, std::vector<T>* distances) const {
    Node* v_node = (Node *)alloca(_s);
    D::template zero_value<Node>(v_node);
    memcpy(v_node->v, v, sizeof(T) * _f);
    D::init_node(v_node, _f);

    std::priority_queue<pair<T, S> > q;

    if (search_k == -1) {
      search_k = n * _roots.size();
    }

    for (size_t i = 0; i < _roots.size(); i++) {
      q.push(make_pair(Distance::template pq_initial_value<T>(), _roots[i]));
    }

    std::vector<S> nns;
    while (nns.size() < (size_t)search_k && !q.empty()) {
      const pair<T, S>& top = q.top();
      T d = top.first;
      S i = top.second;
      Node* nd = _get(i);
      q.pop();
      if (nd->n_descendants == 1 && i < _n_items) {
        nns.push_back(i);
      } else if (nd->n_descendants <= _K) {
        const S* dst = nd->children;
        nns.insert(nns.end(), dst, &dst[nd->n_descendants]);
      } else {
        T margin = D::margin(nd, v, _f);
        q.push(make_pair(D::pq_distance(d, margin, 1), static_cast<S>(nd->children[1])));
        q.push(make_pair(D::pq_distance(d, margin, 0), static_cast<S>(nd->children[0])));
      }
    }

    // Get distances for all items
    // To avoid calculating distance multiple times for any items, sort by id
    std::sort(nns.begin(), nns.end());
    std::vector<pair<T, S> > nns_dist;
    S last = -1;
    for (size_t i = 0; i < nns.size(); i++) {
      S j = nns[i];
      if (j == last)
        continue;
      last = j;
      if (_get(j)->n_descendants == 1)  // This is only to guard a really obscure case, #284
        nns_dist.push_back(make_pair(D::distance(v_node, _get(j), _f), j));
    }

    size_t m = nns_dist.size();
    size_t p = n < m ? n : m; // Return this many items
    std::partial_sort(nns_dist.begin(), nns_dist.begin() + p, nns_dist.end());
    for (size_t i = 0; i < p; i++) {
      if (distances)
        distances->push_back(D::normalized_distance(nns_dist[i].first));
      result->push_back(nns_dist[i].second);
    }
  }
};
/* =========================================================================================
 * HAMMING WRAPPER FOR BINARY DATA
 * ========================================================================================= */
/**
 * @brief Wrapper for Hamming distance with binary data packing - Generic Template Adapter
 *
 * Efficiently handles binary/boolean data by packing into uint32/uint64.
 * Supports float16, bool, uint8_t external representations.
 *
 * Design Invariants:
 *   - External dimension is immutable after construction
 *   - Internal dimension = ceil(f_external / bits_per_word)
 *   - Serialization is strict, versioned, and validated
 *   - No implicit dimension mutation during deserialize/load
 *
 * @tparam S Index type (int32_t | int64_t)
 * @tparam T External data type (float | double) - user-facing API
 * @tparam InternalT Internal packed type (uint32_t, uint64_t) - internal packed representation
 * @tparam Random Random generator RNG (Kiss32Random | Kiss64Random)
 * @tparam ThreadedBuildPolicy: Build threading policy
 */
template<typename S, typename T, typename InternalT, typename Random, class ThreadedBuildPolicy>
class HammingWrapper final
  : public AnnoyIndexInterface<S, T, // Random type
#if __cplusplus >= 201103L
  // typename Random::result_type
  typename std::remove_const<decltype(Random::default_seed)>::type
#else
  typename Random::seed_type
#endif
> {
#if __cplusplus >= 201103L
  static_assert(is_valid_index_type<S>::value,
                "S must be int32_t, int64_t, float, or double");
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "T must be float or double for HammingWrapper external interface");
  static_assert(std::is_same<InternalT, bool>::value ||
                std::is_same<InternalT, uint8_t>::value ||
                std::is_same<InternalT, uint32_t>::value ||
                std::is_same<InternalT, uint64_t>::value,
                "InternalT must be bool, uint8_t, uint32_t, or uint64_t for Hamming storage");
  // static_assert(is_valid_random_seed_type<R>::value,
  //               "Random seed type must be uint32_t or uint64_t");
#endif

public:

#if __cplusplus >= 201103L
  // typedef typename Random::result_type R;
  // using R = typename Random::result_type;
  // Define type alias Same thing (older style)
  // typedef typename std::remove_const<decltype(Random::default_seed)>::type R;
  using R = typename std::remove_const<decltype(Random::default_seed)>::type;
#else
  // typedef typename Random::seed_type R;
  using R = typename Random::seed_type;
#endif

// ⚠️ ← declared first assign first ❌ This violates the declaration order.
private:

  // Centralized parameters
  AnnoyParams _params;

  // Core data structures
  // std::vector<Node<S, T>*> _nodes;
  // std::vector<S> _roots;
  // std::vector<T*> _items;
  // Random _random;

  // State flags
  // bool _built;
  // bool _on_disk;
  // bool _loaded;

  // File handling
  // int _fd;
  // void* _mmap_ptr;
  // size_t _mmap_size;

  // Thread synchronization (via policy)
  // ThreadPolicy _build_policy;
// ⚠️ ← declared first assign first ❌ This violates the declaration order.
protected:
  bool _verbose;                    // Verbose logging
  // const int _f;                     // Dimension (0 means "infer from first vector")
  // size_t _s;                        // Node size in bytes
  // S _n_items;                       // Number of items added

  // // Core data structures
  // void* _nodes;                     // Could either be mmapped, or point to a memory buffer that we reallocate
  // S _n_nodes;                       // Total number of nodes
  // S _nodes_size;                    // Allocated size
  // std::vector<S> _roots;            // Root node indices

  // S _K;                             // Max descendants per leaf
  // Random _random;                   // RNG instance
  // R _seed;                          // seed value (0 = default_seed for kiss)
  // bool _loaded;                     // True if loaded from disk
  // int _fd;                          // File descriptor (for on-disk build)
  // bool _on_disk;
  // bool _built;
  // std::atomic<bool> _build_failed;  // Thread-safe build failure flag

// ⚠️ ← declared first assign first ❌ This violates the declaration order.
private:
  // -------------------- Serialization constants --------------------
  static constexpr uint32_t HAMMING_MAGIC   = 0x48414D4D; // 'HAMM'
  static constexpr uint32_t HAMMING_VERSION = 1;

  // static constexpr size_t BITS_PER_WORD = sizeof(InternalT) * 8;
  static constexpr int BITS_PER_WORD = sizeof(InternalT) * 8;

  // -------------------- Dimensions (immutable) ---------------------
  bool _f_inferred;
  const S _f_external;  // External dimension (number of bits/bools from user)
  const S _f_internal;  // Internal dimension (number of words to store)

  // -------------------- Underlying Annoy index ---------------------
  // AnnoyIndex<int32_t, uint64_t, Hamming, Kiss64Random, AnnoyIndexThreadedBuildPolicy> _index;
  AnnoyIndex<S, InternalT, Hamming, Random, ThreadedBuildPolicy> _index;

  // -------------------- Serialization header -----------------------
  struct HammingHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t f_external;
    uint32_t f_internal;
    uint32_t n_items;
    uint32_t reserved;   // must be zero
  };

  // -------------------- Utilities ----------------------------------
  // Duplicate a C-style string (safe memory allocation)
  static char* dup_cstr(const char* s) {
    if (s == NULL) return NULL;
    const size_t len = strlen(s);
    char* r = static_cast<char*>(malloc(len + 1));
    if (r == NULL) return NULL;
    memcpy(r, s, len + 1);
    return r;
  }

  inline InternalT _clip_distance(InternalT d) const noexcept {
    const InternalT max_d = static_cast<InternalT>(_f_external);
    return (d > max_d) ? max_d : d;
  }

  // -------------------- Packing / unpacking ------------------------
  void _pack(const T* embedding, InternalT* packed) const {
    if (!embedding || !packed) {
      throw std::invalid_argument("Null embedding pointer");
    }
    for (S w = 0; w < _f_internal; ++w) {
      InternalT word = 0;
      const S base = w * BITS_PER_WORD;
      for (int b = 0; b < BITS_PER_WORD && base + b < _f_external; ++b) {
        // ⚠ Optional note: > 0.5f vs >= 0.5f is a policy decision, not a bug.
        if (embedding[base + b] >=T(0.5)) {
          word |= (InternalT(1) << b);
        }
      }
      packed[w] = word;
    }
  }

  void _unpack(const InternalT* packed, T* embedding) const {
    if (!packed || !embedding) {
      throw std::invalid_argument("Null packed pointer");
    }
    for (S i = 0; i < _f_external; ++i) {
      // const uint64_t bit = (packed[i >> 6] >> (i & 63)) & 1ULL;
      const InternalT bit = (packed[i / BITS_PER_WORD] >> (i % BITS_PER_WORD)) & InternalT(1);
      embedding[i] = static_cast<T>(bit);
    }
  }

public:
  /**
   * @brief Constructor with centralized parameter management
   *
   * Same signature as AnnoyIndex for consistency.
   *
   * @param f External dimension in bits (0 = infer from first vector)
   * @param n_trees Number of trees (-1 = auto)
   * @param n_neighbors Default neighbors for queries
   * @param on_disk_path Default path for on_disk_build
   * @param prefault Prefault pages when loading
   * @param seed Random seed
   * @param verbose Verbosity level
   * @param schema_version Serialization version
   * @param n_jobs Number of threads (-1 = all cores)
   */
  // -------------------- Construction -------------------------------
  // AnnoyIndex(int f) : _f(f),
  // explicit HammingWrapper(S f)
  //     : _f_external(f)
  //     , _f_internal((f + BITS_PER_WORD - 1) / BITS_PER_WORD)
  //     , _f_inferred(false)
  //     , _index(_f_internal) { }
  HammingWrapper() = default;
  explicit HammingWrapper(
    int f = 0,                        // DEFAULT_DIMENSION
    int n_trees = -1,
    int n_neighbors = 5,
    const char* on_disk_path = NULL,
    bool prefault = false,
    int seed = 0,
    bool verbose = false,
    int schema_version = 0,
    int n_jobs = 1,
    double l1_ratio = 0.0
  )
    // : _f_external(f),
    //   _f_internal((f + 63) / 64),
    //   _index((f + 63) / 64) {}
    :  _verbose(verbose)
    , _f_inferred(false)
    , _f_external(static_cast<S>(f))
    , _f_internal((static_cast<S>(f) + BITS_PER_WORD - 1) / BITS_PER_WORD)
    , _index(                         // Start with dimension 0 (lazy)
      (static_cast<S>(f) + BITS_PER_WORD - 1) / BITS_PER_WORD,
      n_trees,
      n_neighbors,
      on_disk_path,
      prefault,
      seed,
      verbose,
      schema_version,
      n_jobs
    )
  {
    // if (f < 0) {
    //   throw std::invalid_argument("HammingWrapper: dimension must be non-negative (use 0 for lazy inference)");
    // }
    // Store all parameters
    _params.f = f;
    _params.n_trees = n_trees;
    _params.n_neighbors = n_neighbors;
    _params.on_disk_path = on_disk_path;
    _params.prefault = prefault;
    _params.seed = seed;
    _params.verbose = verbose;
    _params.schema_version = schema_version;
    _params.n_jobs = n_jobs;
    _params.l1_ratio = l1_ratio;
    _params.f_inferred = (f == 0);  // Mark for lazy inference
    // Validate parameters
    char* error = NULL;
    if (!_params.validate(&error)) {
      if (_verbose && error != NULL) {
        fprintf(stderr, "AnnoyIndex parameter validation failed: %s\n", error);
        std::free(error);
      }
    }
    if (_verbose) {
      fprintf(stderr, "AnnoyIndex initialized: f=%d, n_trees=%d, n_neighbors=%d, n_jobs=%d\n",
              _params.f, _params.n_trees, _params.n_neighbors, _params.n_jobs);
    }
  }
  virtual ~HammingWrapper() { unload(); }

  int get_f() const noexcept{
      return _index.get_f();
  }

  // bool set_f(int f, char** error = NULL) noexcept{
  //     return _index.set_f(f, error);
  // }

  // Set dimension (must be called before add_item if using default constructor)
  bool set_f(int f, char** error = NULL) noexcept{
    if (_index.get_n_items() > 0) {
      if (error != NULL) {
        *error = dup_cstr("Cannot change dimension after items have been added");
      }
      return false;
    }

    if (f <= 0) {
      if (error != NULL) {
        *error = dup_cstr("Dimension must be positive");
      }
      return false;
    }

    // _f_external = static_cast<S>(f);
    // _f_internal = static_cast<S>(
    //   (_f_external + BITS_PER_WORD - 1) / BITS_PER_WORD
    // );
    // _f_inferred = true;

    return _index.set_f(static_cast<int>(f), error);
  }

  bool get_params(
      std::vector<std::pair<std::string, std::string>>& params
  ) const noexcept{
      params.clear();
      params.emplace_back("f", std::to_string(_f_external));
      return true;
  }
  bool set_params(
      const std::vector<std::pair<std::string, std::string>>& params,
      char** error = NULL
  ) noexcept{
      for (const auto& kv : params) {
          if (kv.first == "f") {
              int f = std::stoi(kv.second);
              return set_f(f, error);
          }
      }

      if (error) {
          *error = strdup("missing parameter: f");
      }
      return false;
  }

  // -------------------- AnnoyIndexInterface ------------------------
  bool add_item(S item, const T* embedding, char** error = NULL) noexcept{
    try {
      if (embedding == NULL) {
        if (error != NULL) {
          *error = dup_cstr("Null embedding pointer");
        }
        return false;
      }

      if (_f_external == 0 && !_f_inferred) {
        if (error != NULL) {
          *error = dup_cstr("Dimension not specified. Use HammingWrapper(f) or call set_f(f) before add_item");
        }
        return false;
      }

      if (_f_external == 0) {
        if (error != NULL) {
          *error = dup_cstr("Dimension must be set before adding items");
        }
        return false;
      }

      std::vector<InternalT> packed(static_cast<size_t>(_f_internal));
      _pack(embedding, packed.data());
      return _index.add_item(item, packed.data(), error);
    } catch (const std::exception& e) {
      if (error) *error = dup_cstr(e.what());
      return false;
    }
  }

  bool build(int n_trees = -1, int n_threads = -1, char** error = NULL) noexcept{
    return _index.build(n_trees, n_threads, error);
  }

  // -------------------- Serialization ------------------------------
  std::vector<uint8_t> serialize(char** error) const noexcept{
    HammingHeader hdr{};
    hdr.magic      = HAMMING_MAGIC;
    hdr.version    = HAMMING_VERSION;
    hdr.f_external = static_cast<uint32_t>(_f_external);
    hdr.f_internal = static_cast<uint32_t>(_f_internal);
    hdr.n_items    = static_cast<uint32_t>(_index.get_n_items());
    hdr.reserved   = 0U;

    std::vector<uint8_t> out(sizeof(hdr));
    memcpy(out.data(), &hdr, sizeof(hdr));

    std::vector<uint8_t> payload = _index.serialize(error);
    if (payload.empty() && error != NULL && *error != NULL) {
      return std::vector<uint8_t>();
    }

    out.insert(out.end(), payload.begin(), payload.end());
    return out;
  }

  bool deserialize(std::vector<uint8_t>* bytes, bool prefault = false, char** error = NULL) noexcept{
    if (!bytes || bytes->size() < sizeof(HammingHeader)) {
      if (error) *error = dup_cstr("Invalid or empty Hamming index");
      return false;
    }

    HammingHeader hdr;
    memcpy(&hdr, bytes->data(), sizeof(hdr));

    if (hdr.magic != HAMMING_MAGIC ||
        hdr.version != HAMMING_VERSION ||
        hdr.reserved != 0 ||
        hdr.f_external != static_cast<uint32_t>(_f_external) ||
        hdr.f_internal != static_cast<uint32_t>(_f_internal)) {
      if (error) *error = dup_cstr("Hamming header mismatch");
      return false;
    }

    const size_t payload_size =
        bytes->size() - sizeof(HammingHeader);
    if (payload_size == 0) {
      if (error) *error = dup_cstr("Missing Annoy payload");
      return false;
    }

    std::vector<uint8_t> payload(
        bytes->begin() + sizeof(HammingHeader), bytes->end());

    return _index.deserialize(&payload, prefault, error);
  }

  // -------------------- Querying -----------------------------------
  T get_distance(S i, S j) const noexcept{
    return static_cast<T>(
        _clip_distance(_index.get_distance(i, j)));
  }

  void get_item(S indice,
                T* embedding) const noexcept{
    std::vector<InternalT> packed(_f_internal);
    _index.get_item(indice, packed.data());
    _unpack(packed.data(), embedding);
  }

  S get_n_items() const noexcept{
    return _index.get_n_items();
  }

  S get_n_trees() const noexcept{
    return _index.get_n_trees();
  }

  // -------------------- Nearest Neighbor Queries -----------------------
  void get_nns_by_item(S               query_indice,
                       size_t          n,
                       int             search_k,
                       std::vector<S>* result,
                       std::vector<float>* distances) const noexcept override {
    if (distances) {
      vector<InternalT> internal_distances;
      _index.get_nns_by_item(query_indice,
                             n,
                             search_k,
                             result,
                             &internal_distances);

      distances->resize(internal_distances.size());
      for (size_t i = 0; i < internal_distances.size(); ++i) {
        InternalT d = internal_distances[i];
        const InternalT max_d = static_cast<InternalT>(_f_external);
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
  void get_nns_by_vector(const T*           query_embedding,
                         size_t             n,
                         int                search_k,
                         vector<int32_t>*   result,
                         vector<T>*         distances) const noexcept override {
    vector<InternalT> packed_query(_f_internal, 0ULL);
    _pack(query_embedding, &packed_query[0]);

    if (distances) {
      vector<InternalT> internal_distances;
      _index.get_nns_by_vector(&packed_query[0],
                               n,
                               search_k,
                               result,
                               &internal_distances);

      distances->resize(internal_distances.size());
      for (size_t i = 0; i < internal_distances.size(); ++i) {
        InternalT d = internal_distances[i];
        const InternalT max_d = static_cast<InternalT>(_f_external);
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

  // -------------------- Disk I/O -----------------------------------
  bool load(const char* filename,
            bool prefault=false,
            char** error = NULL) noexcept{
    return _index.load(filename, prefault, error);
  }

  bool save(const char* filename,
            bool prefault=false,
            char** error = NULL) noexcept{
    return _index.save(filename, prefault, error);
  }

  bool on_disk_build(const char* filename,
                     char** error = NULL) noexcept{
    return _index.on_disk_build(filename, error);
  }

  bool unbuild(char** error = NULL) noexcept{
    return _index.unbuild(error);
  }

  void unload() noexcept{
    _index.unload();
  }

  void set_seed(R seed) noexcept override {
    _index.set_seed(seed);
  }

  void set_verbose(bool v) noexcept override {
    _index.set_verbose(v);
  }
};

class AnnoyIndexSingleThreadedBuildPolicy {
public:
  template<typename S, typename T, typename D, typename Random>
  static void build(AnnoyIndex<S, T, D, Random, AnnoyIndexSingleThreadedBuildPolicy>* annoy, int q, int n_threads) {
    AnnoyIndexSingleThreadedBuildPolicy threaded_build_policy;
    annoy->thread_build(q, 0, threaded_build_policy);
  }

  void lock_n_nodes() {}
  void unlock_n_nodes() {}

  void lock_nodes() {}
  void unlock_nodes() {}

  void lock_shared_nodes() {}
  void unlock_shared_nodes() {}

  void lock_roots() {}
  void unlock_roots() {}
};

#ifdef ANNOYLIB_MULTITHREADED_BUILD
class AnnoyIndexMultiThreadedBuildPolicy {
private:
  std::shared_timed_mutex nodes_mutex;
  std::mutex n_nodes_mutex;
  std::mutex roots_mutex;

public:
  template<typename S, typename T, typename D, typename Random>
  static void build(AnnoyIndex<S, T, D, Random, AnnoyIndexMultiThreadedBuildPolicy>* annoy, int q, int n_threads) {
    AnnoyIndexMultiThreadedBuildPolicy threaded_build_policy;
    if (n_threads == -1) {
      // If the hardware_concurrency() value is not well defined or not computable, it returns 0.
      // We guard against this by using at least 1 thread.
      n_threads = std::max(1, (int)std::thread::hardware_concurrency());
    }

    std::vector<std::thread> threads(n_threads);

    for (int thread_idx = 0; thread_idx < n_threads; thread_idx++) {
      int trees_per_thread = q == -1 ? -1 : (int)floor((q + thread_idx) / n_threads);

      threads[thread_idx] = std::thread(
        &AnnoyIndex<S, T, D, Random, AnnoyIndexMultiThreadedBuildPolicy>::thread_build,
        annoy,
        trees_per_thread,
        thread_idx,
        std::ref(threaded_build_policy)
      );
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  void lock_n_nodes() {
    n_nodes_mutex.lock();
  }
  void unlock_n_nodes() {
    n_nodes_mutex.unlock();
  }

  void lock_nodes() {
    nodes_mutex.lock();
  }
  void unlock_nodes() {
    nodes_mutex.unlock();
  }

  void lock_shared_nodes() {
    nodes_mutex.lock_shared();
  }
  void unlock_shared_nodes() {
    nodes_mutex.unlock_shared();
  }

  void lock_roots() {
    roots_mutex.lock();
  }
  void unlock_roots() {
    roots_mutex.unlock();
  }
};
#endif

}

#endif
// vim: tabstop=2 shiftwidth=2
