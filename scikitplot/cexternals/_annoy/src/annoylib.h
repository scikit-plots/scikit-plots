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
//
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | Type                 | Introduced | Exact Width Guaranteed | Signedness            | ISO C Minimum Width | Typical Width (Modern) | Minimum (Formula) | Maximum (Formula) | Overflow Semantics | Portability Risk | Notes                                          |
// +======================+============+========================+=======================+=====================+========================+===================+===================+====================+==================+================================================+
// | char                 | C89        | No                     | Implementation-defined| >=8 bits            | 8 bits                 | 0 OR -2^(n-1)     | 2^n-1 OR 2^(n-1)-1| Signed: UB         | HIGH             | Cannot assume signedness                       |
// |                      |            |                        |                       |                     |                        |                   |                   | Unsigned: modulo   |                  |                                                |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | signed char          | C89        | No                     | Signed                | >=8 bits            | 8 bits                 | -2^(n-1)          | 2^(n-1)-1         | UB on overflow     | LOW              | Distinct type from char                        |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | unsigned char        | C89        | No                     | Unsigned              | >=8 bits            | 8 bits                 | 0                 | 2^n-1             | Modulo 2^n         | LOW              | Only type guaranteed to hold raw byte data     |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | short                | C89        | No                     | Signed                | >=16 bits           | 16 bits                | -2^(n-1)          | 2^(n-1)-1         | UB on overflow     | LOW              | At least 16 bits                               |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | unsigned short       | C89        | No                     | Unsigned              | >=16 bits           | 16 bits                | 0                 | 2^n-1             | Modulo 2^n         | LOW              |                                                |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | int                  | C89        | No                     | Signed                | >=16 bits           | 32 bits                | -2^(n-1)          | 2^(n-1)-1         | UB on overflow     | MED              | Most efficient native integer                  |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | unsigned int         | C89        | No                     | Unsigned              | >=16 bits           | 32 bits                | 0                 | 2^n-1             | Modulo 2^n         | LOW              |                                                |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | long                 | C89        | No                     | Signed                | >=32 bits           | 32 or 64 bits          | -2^(n-1)          | 2^(n-1)-1         | UB on overflow     | HIGH             | 32-bit (Windows), 64-bit (Linux/macOS)         |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | unsigned long        | C89        | No                     | Unsigned              | >=32 bits           | 32 or 64 bits          | 0                 | 2^n-1             | Modulo 2^n         | HIGH             | Platform dependent                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | long long            | C99        | No                     | Signed                | >=64 bits           | 64 bits                | -2^63             | 2^63-1            | UB on overflow     | LOW              | At least 64 bits                               |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | unsigned long long   | C99        | No                     | Unsigned              | >=64 bits           | 64 bits                | 0                 | 2^64-1            | Modulo 2^64        | LOW              |                                                |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | int8_t               | C99        | Yes (if exists)        | Signed                | Exactly 8 bits      | 8 bits                 | -2^7              | 2^7-1             | UB on overflow     | NONE             | max 127 items, Requires exact-width support    |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | uint8_t              | C99        | Yes (if exists)        | Unsigned              | Exactly 8 bits      | 8 bits                 | 0                 | 2^8-1             | Modulo 2^8         | NONE             | max 255 items                                  |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | int16_t              | C99        | Yes (if exists)        | Signed                | Exactly 16 bits     | 16 bits                | -2^15             | 2^15-1            | UB on overflow     | NONE             | max 32 767 items                               |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | uint16_t             | C99        | Yes (if exists)        | Unsigned              | Exactly 16 bits     | 16 bits                | 0                 | 2^16-1            | Modulo 2^16        | NONE             | max 65 535 items                               |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | int32_t              | C99        | Yes (if exists)        | Signed                | Exactly 32 bits     | 32 bits                | -2^31             | 2^31-1            | UB on overflow     | NONE             | max ~2.1 E8 B items, 4 bytes/node-id           |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | uint32_t             | C99        | Yes (if exists)        | Unsigned              | Exactly 32 bits     | 32 bits                | 0                 | 2^32-1            | Modulo 2^32        | NONE             | max ~4.3 E9 (10⁹) B items, 4 bytes/node-id     |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | int64_t              | C99        | Yes (if exists)        | Signed                | Exactly 64 bits     | 64 bits                | -2^63             | 2^63-1            | UB on overflow     | NONE             | max ~9.2 E18 items                             |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
// | uint64_t             | C99        | Yes (if exists)        | Unsigned              | Exactly 64 bits     | 64 bits                | 0                 | 2^64-1            | Modulo 2^64        | NONE             | max ~1.8 × E19 (10¹⁹) items, 8 bytes/node-id   |
// +----------------------+------------+------------------------+-----------------------+---------------------+------------------------+-------------------+-------------------+--------------------+------------------+------------------------------------------------+
//
/* =========================================================================================
 * ENHANCED ANNOYLIB - COMPREHENSIVE TYPE SUPPORT & PARAMETER MANAGEMENT
 * =========================================================================================
 *
 * fully support:
 *
 *   * 32-bit ID
 *   * 64-bit ID
 *   * 32-bit OS
 *   * 64-bit OS
 *   * Linux / Windows / macOS
 *   * GCC / Clang / MSVC
 *
 * Canonical semantics:
 *
 *   * "indice"        : integer ID returned by Annoy
 *   * "index"         : 0..n-1 row position in a result set (Python side)
 *   * "distance"      : Hamming distance (clipped to [0, f_external])
 *   * "embedding"     : binary embedding represented as float[0,1] or float[-inf,inf] on the API
 *
 * Features:
 *
 * - Flexible index types: int32_t, int64_t, uint32_t, uint64_t
 * - Extended floating-point: float16, float32, float64, float128
 * - Boolean/binary data: bool, uint8_t for 1-bit/8-bit data
 * - Future-proof: Generic template design with compile-time validation
 * - Cross-platform: Windows/Mac/Linux/Arch compatibility
 * - Lazy construction: Dimension inference from first add_item
 * - Default initialization: All variables initialized to safe defaults
 *
 * Key Enhancements:
 *
 * 1. **Extended Type Support**:
 *    - float16, float32, float64, float128 for data
 *    - bool, uint8_t for binary/boolean data
 *    - int32_t, int64_t, uint32_t, uint64_t for indices
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
 *
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

/* =========================
 * Standard C/C++ library
 * ========================= */
// #include <limits>     // (Pure C++) C++ interface headers first
// #include <climits>    // (C++ + C compatibility macros if needed) C compatibility wrapper second
// #include <limits.h>   // (❌ Do not include Pure C) legacy requirement headers last to avoid macro side effects leaking upward (last)
// extern "C" {
// #include <some_c_library.h>
// }

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

// (C++ Header) C++ headers place them in std::
#include <stdexcept>
#include <utility>
#include <algorithm>  // std::max
#include <queue>
#include <limits>  // C++ Type Traits Interface  // std::numeric_limits
#include <string>  // REQUIRED for std::to_string, std::stoi
#include <vector>

// (C++ Header C Macro Compat) C Compatibility Header
#include <climits>
#include <cstring>
#include <cerrno>
#include <cmath>  // std::abs std::copysign
#include <cstddef>
#include <cstdio>   // C++ correct fprintf, stderr
#include <cstdlib>
#include <thread>   // std::thread::hardware_concurrency() — required by resolve_n_jobs()

// For 201103L "This library requires at least C++11 (-std=c++11)."
// For 201402L "This library requires at least C++14 (-std=c++14)."
// For 201703L "This library requires at least C++17 (-std=c++17)."
#if __cplusplus >= 201103L
  // provides compile-time type information and type transformations
  #include <type_traits>  // std::is_floating_point
  #include <unordered_set>
  // REQUIRED for std::atomic requires C++11 or newer.
  #include <atomic>
#endif

#ifdef ANNOYLIB_MULTITHREADED_BUILD
  #include <thread>
  #include <functional>
  #include <mutex>
  #if __cplusplus >= 201402L
    /* std::shared_timed_mutex is standardised in C++14.
     * std::shared_mutex (without "timed") requires C++17. */
    #include <shared_mutex>
  #else
    /* C++11: <shared_mutex> is absent from the standard.
     * AnnoyIndexMultiThreadedBuildPolicy falls back to std::mutex
     * (exclusive lock only). lock_shared / unlock_shared map to
     * lock / unlock in the C++11 build path. */
    #include <mutex>
  #endif
#endif

// (C Header) C headers place symbols in global namespace
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>   // C legacy guarantees snprintf mapping
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

// only on non-Windows systems.
#if !defined(_WIN32)
  #include <unistd.h>
#endif

/* =========================
 * Compile-time ABI assertions (C++11)
 *
 * Verify that the fundamental types have the exact widths the rest of the
 * code assumes.  These fire at compile time — zero runtime cost.
 * The C++11 guard is redundant (line 278 enforces C++11 via #error) but is
 * kept as documentation of the minimum requirement for each assertion.
 * ========================= */
#if __cplusplus >= 201103L
  static_assert(sizeof(uint8_t)  == 1, "uint8_t must be exactly 8 bits");
  static_assert(sizeof(uint16_t) == 2, "uint16_t must be exactly 16 bits");
  static_assert(sizeof(uint32_t) == 4, "uint32_t must be exactly 32 bits");
  static_assert(sizeof(uint64_t) == 8, "uint64_t must be exactly 64 bits");
  static_assert(sizeof(int32_t)  == 4, "int32_t must be exactly 32 bits");
  static_assert(sizeof(int64_t)  == 8, "int64_t must be exactly 64 bits");
  static_assert(sizeof(float)    == 4, "float must be 32-bit IEEE 754");
  static_assert(sizeof(double)   == 8, "double must be 64-bit IEEE 754");
#endif

#if __cplusplus < 201103L
  #error "Annoy requires at least C++11 or newer"
#endif

#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
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
      // Implicit constructor: allows float→float16_t in arithmetic expressions,
      // ternary results, and function return values.  This mirrors how native
      // floating-point promotions work and is the standard approach used by
      // PyTorch (c10::Half) and Eigen (Eigen::half).
      float16_t(float f) {  // NOLINT(google-explicit-constructor)
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
      // Compound assignment operators — compute in float, store back as half.
      float16_t& operator+=(float16_t rhs) { *this = float16_t(float(*this) + float(rhs)); return *this; }
      float16_t& operator-=(float16_t rhs) { *this = float16_t(float(*this) - float(rhs)); return *this; }
      float16_t& operator*=(float16_t rhs) { *this = float16_t(float(*this) * float(rhs)); return *this; }
      float16_t& operator/=(float16_t rhs) { *this = float16_t(float(*this) / float(rhs)); return *this; }
      // Unary negation.
      float16_t operator-() const { return float16_t(-float(*this)); }
    };
    #define ANNOY_HAS_F16C_FLOAT16 1
  #endif
#else
  // Portable software implementation
  struct float16_t {
    uint16_t data;
    float16_t() : data(0) {}
    // Implicit constructor: allows float→float16_t in arithmetic expressions,
    // ternary results, and function return values.  This mirrors how native
    // floating-point promotions work and is the standard approach used by
    // PyTorch (c10::Half) and Eigen (Eigen::half).
    float16_t(float f) {  // NOLINT(google-explicit-constructor)
      // IEEE 754 half-precision conversion
      uint32_t x;
      std::memcpy(&x, &f, sizeof(float));
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
      // <cstring>   // must be before usage
      std::memcpy(&fresult, &result, sizeof(float));
      return fresult;
    }
    float16_t& operator=(float f) {
      *this = float16_t(f);
      return *this;
    }
    // Compound assignment operators — compute in float, store back as half.
    float16_t& operator+=(float16_t rhs) { *this = float16_t(float(*this) + float(rhs)); return *this; }
    float16_t& operator-=(float16_t rhs) { *this = float16_t(float(*this) - float(rhs)); return *this; }
    float16_t& operator*=(float16_t rhs) { *this = float16_t(float(*this) * float(rhs)); return *this; }
    float16_t& operator/=(float16_t rhs) { *this = float16_t(float(*this) / float(rhs)); return *this; }
    // Unary negation.
    float16_t operator-() const { return float16_t(-float(*this)); }
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

#ifdef _MSC_VER
  // Needed for Visual Studio to disable runtime checks for std::memcpy / low-level memory ops
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
  /* ---- AVX-512: GCC only (Clang excluded — known correctness issues, see #402) ---- */
  #if defined(ANNOYLIB_COMPILER_GCC) && defined(__AVX512F__)
    #define ANNOYLIB_USE_AVX512 1
  /* ---- AVX: GCC / Clang / MSVC with full SSE feature set ---- */
  #elif defined(__AVX__) && defined(__SSE__) && defined(__SSE2__) && defined(__SSE3__)
    #define ANNOYLIB_USE_AVX 1
  #else
    /* Scalar fallback: hardware lacks AVX, or compiler does not expose it */
    #define ANNOYLIB_SIMD_SCALAR 1
  #endif
#else
  /* NO_MANUAL_VECTORIZATION is defined: force scalar path unconditionally.
   * Without this define the invariant check below fires (sum == 0 != 1). */
  #define ANNOYLIB_SIMD_SCALAR 1
#endif /* NO_MANUAL_VECTORIZATION */

/* Invariant: exactly one backend active. Catches mis-configured builds early. */
#if defined(ANNOYLIB_USE_AVX512) && defined(ANNOYLIB_USE_AVX)
  #error "Invalid SIMD state: both ANNOYLIB_USE_AVX512 and ANNOYLIB_USE_AVX are defined"
#endif
#if (defined(ANNOYLIB_USE_AVX512) + defined(ANNOYLIB_USE_AVX) + defined(ANNOYLIB_SIMD_SCALAR)) != 1
  #error "Exactly one SIMD backend must be active (AVX512, AVX, or SCALAR)"
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

/* NOTE: dup_cstr and set_error_from_string live in namespace Annoy (below).
 * No global-scope duplicates — callers inside the namespace resolve them
 * without qualification; callers outside use Annoy::dup_cstr. */
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
//   std::memcpy(&x, &f, sizeof(float));
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
//   std::memcpy(&result, &bits, sizeof(float));
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
static const int DEFAULT_TREES = -1;         // Default n_trees -1 auto or 10
static const int DEFAULT_NEIGHBORS = 5;      // Default n_neighbors
static const int DEFAULT_SEARCH_K = -1;      // -1 means auto "use automatic value"
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
  // QA-3: constexpr is only valid here when float128_t is a literal type.
  //   - On GCC/Clang with native __float128: __float128 is NOT a literal type
  //     in the C++ standard sense, so constexpr would be ill-formed in strict
  //     mode.  Guard with #ifdef so MSVC / strict builds use the base template.
  //   - On MSVC or generic fallback (float128_t == long double): long double IS
  //     a literal type; constexpr is valid.
#ifdef ANNOYLIB_HAVE_FLOAT128
  // __float128 path: drop constexpr (not a literal type), keep noexcept.
  static float128_t get() noexcept { return static_cast<float128_t>(0); }
#else
  // long double path (MSVC / generic): constexpr is valid.
  static constexpr float128_t get() noexcept { return 0.0L; }
#endif
};
/**
 * safe_divide — numerically robust division for floating-point types.
 *
 * Returns `default_val` when the denominator is effectively zero
 * (|denom| < epsilon * max(|num|, |denom|)).  Otherwise returns num/denom
 * directly — no denominator clamping, no sign alteration.
 *
 * Rationale
 * ---------
 * The previous code had two overloads with conflicting semantics:
 *   - 3-arg version:  trivial (b != 0) check only — misses near-zero denoms.
 *   - 2-arg version:  clamps denom to ±threshold — alters the result sign
 *                     and magnitude when denom is tiny but non-zero.
 *
 * This single form avoids both problems:
 *   - 0 / 0               → default_val (scale == 0, threshold == 0, condition true)
 *   - near-zero / near-zero → default_val
 *   - normal division     → num / denom unchanged
 *   - huge / tiny         → default_val (prevents overflow)
 *
 * Parameters
 * ----------
 * num         Numerator.
 * denom       Denominator.
 * default_val Returned when |denom| is effectively zero. Defaults to T{}.
 *
 * Returns
 * -------
 * T  Stable division result, or default_val.
 *
 * Requirements
 * ------------
 * T must be a floating-point type (enforced at compile time).
 */
template<typename T>
inline T safe_divide(T num, T denom, T default_val = T{}) noexcept {
  static_assert(std::is_floating_point<T>::value,
                "safe_divide requires a floating-point type");
  const T eps       = std::numeric_limits<T>::epsilon();
  const T abs_num   = std::abs(num);
  const T abs_denom = std::abs(denom);
  const T scale     = std::max(abs_num, abs_denom);
  // scale == 0 means both inputs are zero: 0/0 → default_val.
  // abs_denom < eps*scale means denom is negligibly small relative to the
  // magnitude of the problem: avoid division by a near-zero value.
  if (scale == T{} || abs_denom < eps * scale) {
    return default_val;
  }
  return num / denom;
}

/* Forward declarations — full definitions follow in the UTILITY FUNCTIONS
 * section below.  AnnoyParams::validate() needs these before the definitions
 * appear in the translation unit. */
inline char* dup_cstr(const char* s) noexcept;
inline void  set_error_msg(char** error, const char* msg) noexcept;

// ---------------------------------------------------------------------------
// FUT-1: Typed verbosity constants
// ---------------------------------------------------------------------------
// AnnoyVerbose provides named, type-safe verbosity levels.  The underlying
// member type in AnnoyParams (and _verbose) remains int so existing callers
// that pass integer literals are not broken.  New callers should prefer the
// enum.
//
// Level semantics:
//   Silent (-2) : suppress everything, including errors.
//   Error  (-1) : only fatal errors.
//   Quiet  ( 0) : no output (default).
//   Info   ( 1) : progress milestones (build passes, load events).
//   Debug  ( 2) : per-node / per-thread detail.
//
// All levels are clamped to [-2, 2] at the constructor / set_verbose boundary.
// ---------------------------------------------------------------------------
enum class AnnoyVerbose : int {
  Silent = -2,
  Error  = -1,
  Quiet  =  0,
  Info   =  1,
  Debug  =  2,
};

/// Convert AnnoyVerbose → int (implicit int used throughout the implementation).
inline int annoy_verbose_level(AnnoyVerbose v) noexcept {
  return static_cast<int>(v);
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
  int  verbose;                   // Verbosity level: 0=quiet, 1=info, 2=debug (clamped to [-2,2])
  bool f_inferred;                // Whether f was inferred from first add_item

  // Paths and strings
  // NOTE: std::string (not const char*) so AnnoyParams owns its string data.
  // Eliminates the dangling-pointer risk that existed when these were non-owning
  // const char* pointers whose lifetime was the caller's responsibility.
  std::string on_disk_path;       // Default path for on_disk_build ("" = none)
  std::string metric;             // Metric name
  std::string dtype;              // Data type string
  std::string index_dtype;        // Index type string
  std::string wrapper_dtype;      // Hamming type string
  std::string random_dtype;       // Random type string
  std::string y_dtype;            // Label type string
  std::string y_return_type;      // Return type (list/dict)

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
    , verbose(0)              // 0 = quiet; use int not bool (leveled: 1=info, 2=debug)
    , f_inferred(false)
    , on_disk_path("")     // "" means no path configured; non-empty enables on_disk_build
    , metric("angular")
    , dtype("float")
    , index_dtype("int32")
    , wrapper_dtype("uint32")
    , random_dtype("uint64")  // matches Kiss64Random / uint64_t default seed type
    , y_dtype("float")
    , y_return_type("list")
    , l1_ratio(0.0)
  {}

  /**
   * @brief Destructor - cleanup allocated strings
   */
  ~AnnoyParams() {
    // std::string members are destroyed automatically by their own destructors.
    // No manual memory management required.
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
 * One definition each. AnnoyParams::validate() forward-declared these above;
 * the actual definitions are here. HammingWrapper must NOT add its own copies.
 * ========================================================================================= */

// dup_cstr — safe C-string duplication (malloc-owned, caller must free).
// Returns NULL for NULL input; returns NULL on allocation failure.
inline char* dup_cstr(const char* s) noexcept {
  if (s == NULL) return NULL;
  const size_t len = std::strlen(s);
  char* r = static_cast<char*>(std::malloc(len + 1));
  if (r != NULL) std::memcpy(r, s, len + 1);
  return r;
}

// set_error_msg — write a malloc-owned copy of msg into *error.
// Safe to call with error == NULL (no-op).
inline void set_error_msg(char** error, const char* msg) noexcept {
  if (error != NULL && msg != NULL) {
    *error = dup_cstr(msg);
  }
}

// set_error_from_string — log to stderr via annoylib_showUpdate AND write
// a malloc-owned copy of msg into *error if error is non-NULL.
inline void set_error_from_string(char** error, const char* msg) noexcept {
  annoylib_showUpdate("%s\n", msg);
  if (error != NULL) {
    *error = dup_cstr(msg);
  }
}

// set_error_from_errno — log prefix + strerror(errno) + errno number, and
// write the formatted string into *error.  Uses dynamic allocation so the
// buffer is always large enough for platform-specific strerror() strings.
inline void set_error_from_errno(char** error, const char* prefix) noexcept {
  const char* err_str = std::strerror(errno);
  const int   err_num = errno;
  annoylib_showUpdate("%s: %s (%d)\n", prefix, err_str, err_num);
  if (error == NULL) return;
  // Dynamic size: prefix + ": " + err_str + " (NNN)\0"
  const size_t n = std::strlen(prefix) + 2 + std::strlen(err_str) + 16 + 1;
  *error = static_cast<char*>(std::malloc(n));
  if (*error != NULL) {
    std::snprintf(*error, n, "%s: %s (%d)", prefix, err_str, err_num);
  }
}

// dup_error — alias kept for internal callers that used the old name.
inline char* dup_error(const char* msg) noexcept {
  return dup_cstr(msg);
}

// ---------------------------------------------------------------------------
// FUT-2: Thread count resolution — resolve_n_jobs()
// ---------------------------------------------------------------------------
// Translates the joblib-style n_jobs convention into a concrete positive
// thread count.  This is the single authoritative implementation; build()
// and any future parallel query path should call this rather than inlining
// their own -1 / hardware_concurrency logic.
//
// Convention:
//   -1 → std::thread::hardware_concurrency() (clamped to >= 1)
//    N → N threads verbatim          (N must be >= 1)
//
// Always returns >= 1 so callers can pass the result directly to a
// std::vector<std::thread> constructor without an additional guard.
// ---------------------------------------------------------------------------
inline int resolve_n_jobs(int n_jobs) noexcept {
  if (n_jobs == -1) {
    const unsigned int hw = std::thread::hardware_concurrency();
    return (hw > 0u) ? static_cast<int>(hw) : 1;
  }
  return (n_jobs < 1) ? 1 : n_jobs;
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
 *
 * Supported index types and their item-count ceilings:
 *   int8_t   — max 127 items  (tiny test indices only)
 *   uint8_t  — max 255 items
 *   int16_t  — max 32 767 items
 *   uint16_t — max 65 535 items
 *   int32_t  — max ~2.1 B items, 2,147,483,647
 *   uint32_t — max ~4.3 B items, 4,294,967,295
 *   int64_t  — max ~9.2 E18 items, 9,223,372,036,854,775,807
 *   uint64_t — max ~1.8 E19 items, 18,446,744,073,709,551,615
 *
 * NOTE on uint64_t: the _w bridge uses uint64_t wire type, so item IDs up to
 * UINT64_MAX are accepted at the C++ level.  Callers (Python/Cython) must
 * validate that the ID fits in uint64_t before calling the bridge.
 */
template<typename S>
struct is_valid_index_type {
  static constexpr bool value =
    std::is_same<S, int8_t>::value   ||
    std::is_same<S, int16_t>::value  ||
    std::is_same<S, int32_t>::value  ||
    std::is_same<S, int64_t>::value  ||
    std::is_same<S, uint8_t>::value  ||
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

// ---------------------------------------------------------------------------
// FUT-3: Compile-time type-name strings — TypeName<T>
// ---------------------------------------------------------------------------
// TypeName<T>::value() returns a stable const char* for each supported type.
// Usage:
//   annoylib_showUpdate("index dtype: %s\n", TypeName<S>::value());
//
// Design rationale:
//   - No RTTI (typeid) — works with -fno-rtti.
//   - No std::string allocation — returns a string literal.
//   - Primary template gives "unknown" so novel types compile without error.
//   - All specializations are explicit so there is no ambiguity between,
//     e.g., long (32-bit Windows) and int32_t (which may be a distinct type).
// ---------------------------------------------------------------------------
template<typename T>
struct TypeName {
  static const char* value() noexcept { return "unknown"; }
};

// --- Data types (T parameter) ---
template<> struct TypeName<float16_t>   { static const char* value() noexcept { return "float16";  } };
template<> struct TypeName<float>       { static const char* value() noexcept { return "float32";  } };
template<> struct TypeName<double>      { static const char* value() noexcept { return "float64";  } };
template<> struct TypeName<float128_t>  { static const char* value() noexcept { return "float128"; } };
template<> struct TypeName<bool>        { static const char* value() noexcept { return "bool";     } };
template<> struct TypeName<uint8_t>     { static const char* value() noexcept { return "uint8";    } };
template<> struct TypeName<uint16_t>    { static const char* value() noexcept { return "uint16";   } };
template<> struct TypeName<uint32_t>    { static const char* value() noexcept { return "uint32";   } };
template<> struct TypeName<uint64_t>    { static const char* value() noexcept { return "uint64";   } };
template<> struct TypeName<int8_t>      { static const char* value() noexcept { return "int8";     } };
template<> struct TypeName<int16_t>     { static const char* value() noexcept { return "int16";    } };
template<> struct TypeName<int32_t>     { static const char* value() noexcept { return "int32";    } };
template<> struct TypeName<int64_t>     { static const char* value() noexcept { return "int64";    } };


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

/* =========================================================================================
 * TYPE-SAFE MATH HELPERS
 *
 * Problem
 * -------
 * <cmath> does not provide overloads of std::sqrt / std::fabs for non-standard
 * types such as GCC's __float128.  Calling sqrt(__float128) or fabs(__float128)
 * produces "call of overloaded '…' is ambiguous" because the compiler cannot
 * choose between float, double, and long double overloads.
 *
 * Solution
 * --------
 * annoy_sqrt<T> / annoy_fabs<T> — thin wrappers that:
 *   • Delegate to std::sqrt / std::fabs for every standard arithmetic type.
 *   • Specialise for float128_t when it is a native __float128 (GCC/Clang x86)
 *     by routing through the long-double overload.  Precision loss is at most
 *     ~15 significant decimal digits vs the 33 that true quad precision offers,
 *     but this is the only portable option without linking -lquadmath.
 *   • On platforms where float128_t is already long double (MSVC / generic
 *     fallback), no specialisation is needed and std::sqrt / std::fabs apply
 *     directly.
 *
 * float16_t
 * ---------
 * The struct-based float16_t variants (F16C and software) have an implicit
 * operator float(), so std::sqrt(float16_t) and std::fabs(float16_t) resolve
 * to their float overloads.  However, the return type (float) must be
 * converted back to float16_t, so explicit template specialisations are
 * provided below to make this conversion explicit and avoid relying solely
 * on the implicit float→float16_t constructor.
 * ========================================================================================= */
#ifndef ANNOY_MATH_HELPERS_DEFINED
#define ANNOY_MATH_HELPERS_DEFINED

/// Type-safe sqrt: delegates to std::sqrt for all standard types.
template<typename T>
inline T annoy_sqrt(T x) {
  return std::sqrt(x);
}

/// Type-safe fabs: delegates to std::fabs for all standard types.
template<typename T>
inline T annoy_fabs(T x) {
  return std::fabs(x);
}

#if defined(ANNOY_HAS_FLOAT128)
// float128_t == __float128 (native GCC/Clang quad precision).
// std::sqrt / std::fabs have no __float128 overload in <cmath>.
// Cast through long double (sqrtl / fabsl) as the closest portable option.
// Developer note: if -lquadmath is available on your toolchain you can replace
// these with sqrtq() / fabsq() from <quadmath.h> for full 33-digit precision.
template<>
inline float128_t annoy_sqrt<float128_t>(float128_t x) {
  return static_cast<float128_t>(sqrtl(static_cast<long double>(x)));
}
template<>
inline float128_t annoy_fabs<float128_t>(float128_t x) {
  return static_cast<float128_t>(fabsl(static_cast<long double>(x)));
}
#endif // ANNOY_HAS_FLOAT128

// float16_t struct variants (F16C and software) need explicit specializations
// because std::sqrt/std::fabs return float, requiring conversion back to
// float16_t.  When float16_t is a native typedef (__fp16 on ARM), no
// specialisation is needed — the compiler resolves through built-in overloads.
#if defined(ANNOY_HAS_F16C_FLOAT16) || defined(ANNOY_HAS_SOFTWARE_FLOAT16)
template<>
inline float16_t annoy_sqrt<float16_t>(float16_t x) {
  return float16_t(std::sqrt(static_cast<float>(x)));
}
template<>
inline float16_t annoy_fabs<float16_t>(float16_t x) {
  return float16_t(std::fabs(static_cast<float>(x)));
}
#endif // ANNOY_HAS_F16C_FLOAT16 || ANNOY_HAS_SOFTWARE_FLOAT16

#endif // ANNOY_MATH_HELPERS_DEFINED

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
  // Developer note: use T s{} (value-initialization) rather than T s = 0.
  // Although float16_t now has an implicit float constructor, value-init
  // via T s{} is preferred for generic numeric code — it works for all
  // arithmetic types and clearly conveys "zero" intent.
  T s{};
  for (int z = 0; z < f; z++) {
    s += (*x) * (*y);
    x++;
    y++;
  }
  return s;
}

template<typename T>
inline T manhattan_distance(const T* x, const T* y, int f) {
  // T d{} — value-init to zero; see dot() note.
  // annoy_fabs<T> — resolves fabs(__float128) ambiguity; see ANNOY_MATH_HELPERS.
  T d{};
  for (int i = 0; i < f; i++)
    d += annoy_fabs<T>(x[i] - y[i]);
  return d;
}

template<typename T>
inline T euclidean_distance(const T* x, const T* y, int f) {
  // Don't use dot-product: avoid catastrophic cancellation in #314.
  // T d{} — value-init to zero; see dot() note.
  T d{};
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
    T norm = cosine ? Distance::template get_norm<T, Node>(nodes[k], f) : T(1);
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
// NOTE: The concrete node types used by AnnoyIndex are defined as
// D::template Node<S,T> inside each metric struct (Angular, Euclidean,
// Manhattan, DotProduct, Hamming below).  There is no single top-level
// Node<S,T> — the type alias inside AnnoyIndex is:
//   typedef typename D::template Node<S, T> Node;
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
    std::memcpy(dest->v, source->v, f * sizeof(T));
  }
  template<typename T, typename Node>
  static inline T get_norm(Node* node, int f) {
    // annoy_sqrt<T>: type-safe wrapper; handles __float128 ambiguity.
    // See ANNOY_MATH_HELPERS_DEFINED block above.
    return annoy_sqrt<T>(dot(node->v, node->v, f));
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
      return static_cast<T>(2.0) - static_cast<T>(2.0) * pq / annoy_sqrt<T>(ppqq);
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
    // annoy_sqrt<T>: type-safe wrapper; handles __float128 ambiguity.
    return annoy_sqrt<T>(std::max(distance, T(0)));
  }
  template<typename T>
  static inline T pq_distance(T distance, T margin, int child_nr) {
    if (child_nr == 0)
      margin = -margin;
    return std::min(distance, margin);
  }
  template<typename T>
  static inline T pq_initial_value() {
    return std::numeric_limits<T>::infinity();
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
     * Extension of Angular node for dot-product metric.
     *
     * Layout (all fields T-aligned or explicitly padded):
     *   n_descendants  S         — child count
     *   children[2]    S[2]      — tree child indices
     *   dot_factor     T         — augmentation scalar for preprocess
     *   norm           T         — precomputed L2 norm
     *   built          uint8_t   — flag: index is built (exact dot available)
     *   _pad[]         uint8_t[] — explicit pad to align v[] on sizeof(T) boundary
     *   v[]            T[]       — embedding vector (ANNOYLIB_V_ARRAY_SIZE elements)
     *
     * Developer note: using `bool built` here was ARCH-6: the compiler inserts
     * implicit padding after bool to realign T, making the layout fragile across
     * compilers and optimization flags. uint8_t + explicit pad pins the layout
     * and the static_assert below guarantees it at compile time.
     */
    S n_descendants;
    S children[2]; // Will possibly store more than 2
    T dot_factor;
    T norm;
    uint8_t  built;                      // 0 = not built, 1 = built
    uint8_t  _pad[sizeof(T) - 1];        // explicit pad: align v[] to sizeof(T)
    T v[ANNOYLIB_V_ARRAY_SIZE];
  };

  // Verify the explicit padding produces the same offset for v[] as the
  // compiler would choose on its own.  If this fires, the pad formula needs
  // adjustment for the target T (e.g. long double / __float128 alignment > 8).
  //
  // Developer note: the C preprocessor macro offsetof() cannot accept a
  // comma-bearing template instantiation (e.g. Node<int32_t, float>) as an
  // argument, because it treats the comma as a macro argument separator.
  // The typedef alias sidesteps the macro limitation without changing semantics.
  typedef Node<int32_t, float> _DotProductNodeForStaticAssert;
  static_assert(
    offsetof(_DotProductNodeForStaticAssert, v) ==
      offsetof(_DotProductNodeForStaticAssert, built) + sizeof(float),
    "DotProduct::Node: explicit _pad does not align v[] correctly for float");

  static const char* name() {
    return "dot";
  }

  template<typename T, typename Node>
  static inline T get_norm(Node* node, int f) {
    // annoy_sqrt<T>: type-safe wrapper; handles __float128 ambiguity.
    return annoy_sqrt<T>(dot(node->v, node->v, f) + node->dot_factor * node->dot_factor);
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
    // Note: static_cast<T> on the false branch resolves ternary ambiguity when
    // T is float16_t (both T→float and float→T are one user-defined conversion).
    T pp = x->norm ? x->norm : static_cast<T>(dot(x->v, x->v, f) + x->dot_factor * x->dot_factor);
    T qq = y->norm ? y->norm : static_cast<T>(dot(y->v, y->v, f) + y->dot_factor * y->dot_factor);
    T pq = dot(x->v, y->v, f) + x->dot_factor * y->dot_factor;
    T ppqq = pp * qq;

    if (ppqq > 0) return static_cast<T>(2.0) - static_cast<T>(2.0) * pq / annoy_sqrt<T>(ppqq);
    else return static_cast<T>(2.0);
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
    std::memcpy(dest->v, source->v, f * sizeof(T));
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
    // annoy_sqrt<T> / std::pow: type-safe wrapper; handles __float128 ambiguity.
    T df = node->dot_factor;
    T norm = annoy_sqrt<T>(dot(node->v, node->v, f) + df * df);
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
      T norm = d < 0 ? T{} : annoy_sqrt<T>(d);
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
      T squared_norm_diff = max_norm * max_norm - node_norm * node_norm;
      T dot_factor = squared_norm_diff < T{} ? T{} : annoy_sqrt<T>(squared_norm_diff);

      node->norm = max_norm * max_norm;
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
    return std::numeric_limits<T>::max();
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
    return std::numeric_limits<T>::infinity();
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
    // annoy_sqrt<T>: type-safe wrapper; handles __float128 ambiguity.
    return annoy_sqrt<T>(std::max(distance, T(0)));
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
// These are fundamentally different mechanisms.
// Templates are compile-time polymorphism.
// Base-class pointers are runtime polymorphism.
//
// Non-templated base + virtual functions (most practical)
// Introduce a Non-Templated Polymorphic Base Then your concrete class derives from it.
// std::unique_ptr<AnnoyIndexInterfaceBase> ptr; std::make_unique<...>;  // Now the pointer is type-erased and works for all combinations.
// AnnoyIndexInterface<int32_t, float, uint64_t>* ptr = self->ptr;       // switch types manually. That is unsafe, non-extensible, and breaks type safety.
//
// pattern:
//   enum class → finite runtime configuration
//   std::unique_ptr → RAII ownership
//   std::make_unique → exception-safe allocation
//   Non-templated base with virtual destructor → safe polymorphic deletion
//
// Non-templated base
// +
// Factory
// +
// unique_ptr
// +
// Explicit supported combinations
//
// Step 1 — Define strict runtime type set
// If you support only specific combinations:
// enum class IndexKind {
//     Int32Float,
//     Int64Double
// };
// struct IndexConfig {
//     std::string type;  // "int32_float"
//     int f;
// };
// // Step 2 — Factory Function
// std::unique_ptr<AnnoyIndexInterfaceBase>
// create_index(IndexKind kind, int f) {
//     switch (kind) {
//         case IndexKind::Int32Float:
//             return std::make_unique<
//                 AnnoyIndex<int32_t, float, Angular,
//                            Kiss64Random,
//                            AnnoyIndexThreadedBuildPolicy>
//             >(f);
//         case IndexKind::Int64Double:
//             return std::make_unique<
//                 AnnoyIndex<int64_t, double, Angular,
//                            Kiss64Random,
//                            AnnoyIndexThreadedBuildPolicy>
//             >(f);
//         default:
//             throw std::invalid_argument("Unsupported index kind");
//     }
// }
// // Now:
// self->ptr = create_index(IndexKind::Int32Float, self->f);
//
// 🧩 Alternative Few known combinations: std::variant (If No Virtual overhead Functions) And use std::visit.
// std::unique_ptr<Base> in C++ ✅ Better for pure-C++ usage  Adds RAII in C++; unnecessary complexity for Cython which owns the pointer itself
// std::variant + std::visit    ⚠️ Over-engineering           20 types × 5 template params → variant is unwieldy; virtual dispatch is cleaner and already in place
//
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
  /// Typed overload — preferred over the bool version for new callers.
  virtual void set_verbose(AnnoyVerbose level) noexcept {
    set_verbose(annoy_verbose_level(level) > 0);
  }

  // Core operations
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
  virtual bool deserialize(
    std::vector<uint8_t>* bytes,
    bool prefault = false,
    char** error = NULL
  ) noexcept = 0;

  // sklearn compatibility
  virtual bool get_params(
    std::vector<std::pair<std::string,
    std::string>>& params
  ) const noexcept = 0;
  virtual bool set_params(
    const std::vector<std::pair<std::string,
    std::string>>& params,
    char** error = NULL
  ) noexcept = 0;

  // ─────────────────────────────────────────────────────────────────────────
  // Widened-type virtual API  (suffix _w = "widened wire types")
  //
  // Problem
  // -------
  // AnnoyIndexInterfaceBase is non-templated so it cannot declare virtuals
  // whose signatures contain the template parameters S, T, or R of the
  // derived AnnoyIndexInterface<S,T,R>.  Cython stores the pointer as
  // AnnoyIndexInterfaceBase* (the only type-safe owner for all 20 concrete
  // index types); it therefore cannot reach any S/T/R-typed method without
  // a dangerous reinterpret-cast.
  //
  // Solution: "widened wire types"
  // --------------------------------
  // Declare every type-dependent method here using the *widest* fixed types
  // that safely cover all supported S / T / R combinations:
  //
  //   S (item ID) → uint64_t  covers every signed/unsigned integer type:
  //                            int8_t … int64_t all fit in uint64_t when
  //                            non-negative (item IDs are always ≥ 0).
  //                            The concrete bridge casts uint64_t → S.
  //   T (vector)  → double    covers float    (Python floats are 64-bit)
  //   R (seed)    → uint64_t  covers uint32_t (truncated on call)
  //
  // uint64_t vs int64_t for item IDs
  // ---------------------------------
  // The previous wire type was int64_t, which silently rejected item IDs
  // above 2^63-1 that uint64_t (and even uint32_t) can legitimately hold.
  // uint64_t is the correct wire type because:
  //   1. Item IDs are semantically non-negative.
  //   2. uint64_t ⊇ uint32_t ⊇ uint16_t ⊇ uint8_t (no truncation).
  //   3. uint64_t ⊇ int32_t / int64_t for non-negative values.
  //
  // Callers (Cython/Python) must validate that the Python integer fits in
  // uint64_t before calling; the bridge performs no further range check.
  //
  // Naming: _w suffix avoids C++ overload ambiguity with the identically
  // named typed virtuals on AnnoyIndexInterface<S,T,R>.
  // ─────────────────────────────────────────────────────────────────────────

  // Configuration
  virtual void set_seed_w(uint64_t seed) noexcept = 0;

  // Core operations
  virtual bool add_item_w(uint64_t item, const double* embedding,
                          char** error = NULL) noexcept = 0;

  // Accessors
  virtual uint64_t get_n_items_w()   const noexcept = 0;
  virtual uint64_t get_n_trees_w()   const noexcept = 0;
  virtual void    get_item_w(uint64_t item, double* embedding) const noexcept = 0;
  virtual double  get_distance_w(uint64_t i, uint64_t j) const noexcept = 0;

  // Querying
  virtual void get_nns_by_item_w(
    uint64_t item, size_t n, int search_k,
    std::vector<uint64_t>* result,
    std::vector<double>*  distances) const noexcept = 0;

  virtual void get_nns_by_vector_w(
    const double* vec, size_t n, int search_k,
    std::vector<uint64_t>* result,
    std::vector<double>*  distances) const noexcept = 0;
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
 * @tparam S Index type (int32_t | int64_t | uint32_t | uint64_t, etc.)
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

  // ─────────────────────────────────────────────────────────────────────────
  // Typed abstract methods — implemented by AnnoyIndex / HammingWrapper
  // ─────────────────────────────────────────────────────────────────────────

  // Dimension management
  virtual int get_f() const noexcept = 0;
  virtual bool set_f(int f, char** error = NULL) noexcept = 0;

  // Configuration
  virtual void set_seed(R seed) noexcept = 0;
  virtual void set_verbose(bool verbosity) noexcept = 0;
  /// Typed overload — preferred over the bool version for new callers.
  virtual void set_verbose(AnnoyVerbose level) noexcept {
    set_verbose(annoy_verbose_level(level) > 0);
  }

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
  virtual bool deserialize(
    std::vector<uint8_t>* bytes,
    bool prefault = false,
    char** error = NULL
  ) noexcept = 0;

  // sklearn compatibility
  virtual bool get_params(
    std::vector<std::pair<std::string,
    std::string>>& params
  ) const noexcept = 0;
  virtual bool set_params(
    const std::vector<std::pair<std::string,
    std::string>>& params,
    char** error = NULL
  ) noexcept = 0;

  // ─────────────────────────────────────────────────────────────────────────
  // Bridge implementations of AnnoyIndexInterfaceBase widened API (_w)
  //
  // These are non-abstract `final` overrides that convert from the fixed wire
  // types (int64_t / double / uint64_t) to S / T / R and delegate to the
  // typed pure-virtual methods declared below.
  //
  // `final` prevents derived classes from overriding the bridges; they
  // override only the typed S/T/R methods.
  //
  // All bridges are noexcept and catch std::bad_alloc (from temporary
  // std::vector allocations) routing it through the char** error convention
  // that the rest of the API uses.
  // ─────────────────────────────────────────────────────────────────────────

  void set_seed_w(uint64_t seed) noexcept override final {
    set_seed(static_cast<R>(seed));
  }

  bool add_item_w(uint64_t item, const double* embedding,
                  char** error = NULL) noexcept override final {
    try {
      const int f = get_f();
      std::vector<T> tmp(static_cast<size_t>(f));
      for (int i = 0; i < f; ++i) tmp[i] = static_cast<T>(embedding[i]);
      return add_item(static_cast<S>(item), tmp.data(), error);
    } catch (const std::bad_alloc&) {
      if (error) *error = dup_cstr("out of memory in add_item_w");
      return false;
    }
  }

  uint64_t get_n_items_w() const noexcept override final {
    return static_cast<uint64_t>(get_n_items());
  }

  uint64_t get_n_trees_w() const noexcept override final {
    return static_cast<uint64_t>(get_n_trees());
  }

  void get_item_w(uint64_t item, double* embedding) const noexcept override final {
    try {
      const int f = get_f();
      std::vector<T> tmp(static_cast<size_t>(f));
      get_item(static_cast<S>(item), tmp.data());
      for (int i = 0; i < f; ++i) embedding[i] = static_cast<double>(tmp[i]);
    } catch (...) {}
  }

  double get_distance_w(uint64_t i, uint64_t j) const noexcept override final {
    return static_cast<double>(get_distance(static_cast<S>(i), static_cast<S>(j)));
  }

  void get_nns_by_item_w(
      uint64_t item, size_t n, int search_k,
      std::vector<uint64_t>* result,
      std::vector<double>*  distances) const noexcept override final {
    try {
      std::vector<S> sr;
      if (distances) {
        std::vector<T> sd;
        get_nns_by_item(static_cast<S>(item), n, search_k, &sr, &sd);
        result->resize(sr.size());
        distances->resize(sd.size());
        for (size_t k = 0; k < sr.size(); ++k) (*result)[k]    = static_cast<uint64_t>(sr[k]);
        for (size_t k = 0; k < sd.size(); ++k) (*distances)[k] = static_cast<double>(sd[k]);
      } else {
        get_nns_by_item(static_cast<S>(item), n, search_k, &sr, NULL);
        result->resize(sr.size());
        for (size_t k = 0; k < sr.size(); ++k) (*result)[k] = static_cast<uint64_t>(sr[k]);
      }
    } catch (...) {}
  }

  void get_nns_by_vector_w(
      const double* vec, size_t n, int search_k,
      std::vector<uint64_t>* result,
      std::vector<double>*  distances) const noexcept override final {
    try {
      const int f = get_f();
      std::vector<T> query(static_cast<size_t>(f));
      for (int i = 0; i < f; ++i) query[i] = static_cast<T>(vec[i]);
      std::vector<S> sr;
      if (distances) {
        std::vector<T> sd;
        get_nns_by_vector(query.data(), n, search_k, &sr, &sd);
        result->resize(sr.size());
        distances->resize(sd.size());
        for (size_t k = 0; k < sr.size(); ++k) (*result)[k]    = static_cast<uint64_t>(sr[k]);
        for (size_t k = 0; k < sd.size(); ++k) (*distances)[k] = static_cast<double>(sd[k]);
      } else {
        get_nns_by_vector(query.data(), n, search_k, &sr, NULL);
        result->resize(sr.size());
        for (size_t k = 0; k < sr.size(); ++k) (*result)[k] = static_cast<uint64_t>(sr[k]);
      }
    } catch (...) {}
  }
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
 */
// ---------------------------------------------------------------------------
// FUT-4: On-disk index file prologue
// ---------------------------------------------------------------------------
// AnnoyFileHeader is written at byte 0 of every index file created by
// AnnoyIndex::save().  load() reads it first to detect:
//   - Wrong file type (magic mismatch → clear error, not silent garbage).
//   - Format version mismatch (header_version != ANNOY_FILE_VERSION).
//   - Metric or node-size mismatch (metric_hash, node_size).
//   - Dimension mismatch (_f stored in the header).
//   - Type-size mismatch (sizeof_T, sizeof_S).
//
// Old files that were saved without this header do NOT start with the magic
// 0x414E4958 ('ANIX') so load() can detect them and emit a migration error
// rather than misinterpreting the raw node bytes.
//
// On-disk layout (64 bytes, 8-byte aligned, little-endian by convention):
//   Offset  Size  Field
//      0     4    magic          = 0x414E4958 ('ANIX')
//      4     2    header_version = ANNOY_FILE_VERSION (1)
//      6     1    sizeof_T       = sizeof(T) in bytes
//      7     1    sizeof_S       = sizeof(S) in bytes
//      8     4    node_size      = _s (bytes per node, includes children[])
//     12     4    n_dimensions   = _f at build time
//     16     8    n_items        = item count at build time
//     24     4    metric_hash    = FNV-1a hash of metric name string
//     28     4    reserved       = 0
//     32    32    reserved2      = 0 (future: dtype strings, checksum, etc.)
// ---------------------------------------------------------------------------
static const uint32_t ANNOY_FILE_MAGIC   = 0x414E4958u; // 'ANIX'
static const uint16_t ANNOY_FILE_VERSION = 1u;

// FNV-1a 32-bit — tiny, constexpr-friendly, used only for the metric name.
// Not a security hash; used only to detect obvious metric mismatches at load.
inline uint32_t annoy_fnv1a32(const char* s) noexcept {
  uint32_t h = 2166136261u;
  while (*s) {
    h ^= static_cast<uint8_t>(*s++);
    h *= 16777619u;
  }
  return h;
}

#pragma pack(push, 1)
struct AnnoyFileHeader {
  uint32_t magic;           // must equal ANNOY_FILE_MAGIC
  uint16_t header_version;  // must equal ANNOY_FILE_VERSION
  uint8_t  sizeof_T;        // sizeof(T) — detects float32 vs float64 mismatch
  uint8_t  sizeof_S;        // sizeof(S) — detects int32 vs int64 mismatch
  uint32_t node_size;       // _s: bytes per node
  uint32_t n_dimensions;    // _f at build time
  uint64_t n_items;         // item count
  uint32_t metric_hash;     // FNV-1a of metric name (e.g. "angular")
  uint32_t reserved;        // must be zero
  uint8_t  reserved2[32];   // must be zero; future: dtype strings, checksum
};
#pragma pack(pop)
static_assert(sizeof(AnnoyFileHeader) == 64,
  "AnnoyFileHeader size changed — update ANNOY_FILE_VERSION and save/load paths");

/**
 * @tparam S Index type (int8_t | int16_t | int32_t | int64_t | uint8_t | uint16_t | uint32_t | uint64_t)
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
                "S must be one of: int8_t, int16_t, int32_t, int64_t, "
                "uint8_t, uint16_t, uint32_t, uint64_t");
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

// Member declarations are ordered to exactly match the constructor initializer
// list.  C++ initializes members in *declaration* order regardless of the
// order items appear in the initializer list; mismatches cause subtle bugs
// (e.g. _s computed before _f is set).  Keep this order in sync with the
// constructor below.
private:
  // Centralized parameters — single source of truth for all constructor args.
  AnnoyParams _params;

protected:
  // ---- fields initialized by the constructor initializer list ----
  int    _verbose;    // Verbosity level: 0=quiet 1=info 2=debug (clamped to [-2,2]).
                      // int, not bool — leveled verbosity must not be truncated.
  int    _f;          // Embedding dimension. NOT const: set_f() must be able to
                      // update it during lazy initialization (first add_item when
                      // constructed with f=0). Protected by _n_items==0 guard.
  size_t _s;          // Node size in bytes = offsetof(Node,v) + _f*sizeof(T).
                      // Recomputed whenever _f changes via set_f().
  S      _n_items;    // Number of items added so far.

  // Core data structures
  void*  _nodes;      // Either mmap-backed or heap-realloc'd node array.
  S      _n_nodes;    // Total nodes currently stored (items + internal nodes).
  S      _nodes_size; // Allocated capacity in nodes.
  std::vector<S> _roots; // Root node indices (one per tree).

  S      _K;          // Max children per leaf node = (_s - offsetof(Node,children)) / sizeof(S).
                      // Recomputed whenever _f changes via set_f().
  Random _random;     // RNG instance (stateful; seeded in constructor).
  R      _seed;       // Seed value passed to _random.
  bool   _loaded;     // True after load(); prevents add_item / build.
  int    _fd;         // File descriptor for on-disk build (0 = not open).
  bool   _on_disk;    // True when on_disk_build() has been called.
  bool   _built;      // True after build() completes successfully.

  std::atomic<bool> _build_failed; // Thread-safe build failure flag.
                                    // Always std::atomic<bool>: C++11 is
                                    // required (enforced by #error above).

  // True mmap region when load() maps a headered file (FUT-4).
  // mmap(2) requires its offset to be page-aligned; sizeof(AnnoyFileHeader)=64
  // is not.  We therefore map the entire file from offset 0, store the base
  // and total size here, and point _nodes at (base + header_size).
  // unload() uses these fields for munmap instead of _nodes.
  // Both are NULL/0 when not in use (on_disk path or heap-allocated nodes).
  void*  _mmap_base;  // true mmap base returned by mmap(); NULL if unused
  size_t _mmap_size;  // total bytes passed to mmap();     0   if unused

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
  // NOTE: No default constructor. AnnoyIndex() = default would be either
  // deleted (int _f has no in-class initializer) or leave _f
  // indeterminate — both are undefined behaviour.  All construction goes
  // through the parameterized constructor below; every argument has a default.
  explicit AnnoyIndex(
    int f = 0,                        // 0 → lazy: dimension inferred from first add_item
    int n_trees = -1,                 // -1 → auto (2x memory budget)
    int n_neighbors = 5,              // default result count for get_nns queries
    const std::string& on_disk_path = "",  // non-empty → on_disk_build path
    bool prefault = false,            // MAP_POPULATE on load
    int seed = 0,                     // RNG seed (0 → library default)
    int verbose = 0,                  // 0=quiet 1=info 2=debug (int, not bool)
    int schema_version = 0,           // serialization schema marker
    int n_jobs = 1,                   // threads; -1=all cores (joblib semantics)
    double l1_ratio = 0.0             // future: random projection ratio
  )
    // Initializer list order MUST match member declaration order above.
    : _params()               // zero-initialized via AnnoyParams()
    , _verbose(std::max(-2, std::min(2, verbose)))  // clamp to [-2, 2]
    , _f(f)
    , _s(0)                   // computed below; 0 when f==0 (lazy init)
    , _n_items(S(0))
    , _nodes(NULL)
    , _n_nodes(S(0))
    , _nodes_size(S(0))
    , _roots()
    , _K(S(0))                // computed below
    , _random()
    , _seed(Random::default_seed)
    , _loaded(false)
    , _fd(0)
    , _on_disk(false)
    , _built(false)
    , _build_failed(false)
    , _mmap_base(NULL)    // no mmap-with-header in use yet
    , _mmap_size(0)
  {
    // Store all parameters in the central params struct.
    _params.f             = f;
    _params.n_trees       = n_trees;
    _params.n_neighbors   = n_neighbors;
    _params.on_disk_path  = on_disk_path;
    _params.prefault      = prefault;
    _params.seed          = seed;
    _params.verbose       = _verbose;  // store clamped value
    _params.schema_version = schema_version;
    _params.n_jobs        = n_jobs;
    _params.l1_ratio      = l1_ratio;
    _params.f_inferred    = (f == 0);

    // Validate parameters. Report but do not abort on invalid params so that
    // the object is always in a defined (if unusable) state.
    {
      char* err = NULL;
      if (!_params.validate(&err)) {
        if (_verbose > 0 && err != NULL) {
          annoylib_showUpdate("AnnoyIndex: invalid parameter: %s\n", err);
        }
        std::free(err);
      }
    }

    if (_verbose > 0) {
      annoylib_showUpdate(
        "AnnoyIndex: f=%d n_trees=%d n_neighbors=%d n_jobs=%d verbose=%d"
        " dtype=%s index_dtype=%s\n",
        _params.f, _params.n_trees, _params.n_neighbors, _params.n_jobs, _verbose,
        TypeName<T>::value(), TypeName<S>::value());
    }

    // BUG-6 fix: only compute _s and _K when f > 0.
    // When f == 0 (lazy init), _s and _K remain 0 and are recomputed by
    // set_f() on the first add_item call.  Computing them here with f==0
    // yields _s == offsetof(Node,v) (no vector storage) and _K == 0, which
    // corrupts every subsequent node-pointer and capacity calculation.
    if (_f > 0) {
      _s = offsetof(Node, v) + static_cast<size_t>(_f) * sizeof(T);
      _K = static_cast<S>((_s - offsetof(Node, children)) / sizeof(S));
    }
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
  int get_f() const noexcept override {
    // return _f;
    return static_cast<int>(_f);
  }

  bool set_f(int new_f, char** error = NULL) noexcept override {
    if (new_f <= 0) {
      set_error_from_string(error, "f must be > 0");
      return false;
    }
    if (_n_items > S(0) || _built) {
      set_error_from_string(error,
        "Cannot change f after items have been added or the index has been built");
      return false;
    }
    _f = new_f;
    // Recompute derived constants that depend on _f.
    // These were 0 when constructed with f=0 (lazy init); they must be
    // correct before any node pointer arithmetic or allocation.
    _s = offsetof(Node, v) + static_cast<size_t>(_f) * sizeof(T);
    _K = static_cast<S>((_s - offsetof(Node, children)) / sizeof(S));
    _params.f          = new_f;
    _params.f_inferred = false;
    return true;
  }

  bool add_item(S item, const T* w, char** error=NULL) noexcept override {
    // Pass _f as the known vector size so add_item_impl can validate or infer.
    // When _f==0 (lazy mode) and this is the first call, w_size==0 triggers
    // an error: the caller must supply a concrete dimension via set_f() first
    // or use the w-bridge (add_item_w) which carries the vector length.
    return add_item_impl(item, w, _f, error);
  }

    bool get_params(
        std::vector<std::pair<std::string, std::string>>& params
    ) const noexcept override {
        // Export every field in AnnoyParams so callers can round-trip the full
        // configuration.  The order here matches AnnoyParams declaration order
        // so diffs are readable.  All values are serialized as strings for the
        // sklearn-style API; numeric types use std::to_string.
        params.clear();
        params.emplace_back("f",              std::to_string(_f));
        params.emplace_back("n_trees",        std::to_string(_params.n_trees));
        params.emplace_back("n_neighbors",    std::to_string(_params.n_neighbors));
        params.emplace_back("seed",           std::to_string(_params.seed));
        params.emplace_back("n_jobs",         std::to_string(_params.n_jobs));
        params.emplace_back("schema_version", std::to_string(_params.schema_version));
        params.emplace_back("prefault",       _params.prefault ? "true" : "false");
        params.emplace_back("verbose",        std::to_string(_params.verbose));
        params.emplace_back("f_inferred",     _params.f_inferred ? "true" : "false");
        params.emplace_back("on_disk_path",   _params.on_disk_path);
        params.emplace_back("metric",         _params.metric);
        params.emplace_back("dtype",          _params.dtype);
        params.emplace_back("index_dtype",    _params.index_dtype);
        params.emplace_back("wrapper_dtype",  _params.wrapper_dtype);
        params.emplace_back("random_dtype",   _params.random_dtype);
        params.emplace_back("l1_ratio",       std::to_string(_params.l1_ratio));
        return true;
    }

    bool set_params(
        const std::vector<std::pair<std::string, std::string>>& params,
        char** error = NULL
    ) noexcept override {
        // Iterate all supplied key-value pairs; update the corresponding field.
        // Unknown keys are silently ignored so forward-compatible clients work.
        // Returns false only when a value is syntactically or semantically wrong.
        for (const auto& kv : params) {
            const std::string& key = kv.first;
            const std::string& val = kv.second;

            // --- numeric fields ---
            if (key == "f") {
                int v = 0;
                try { v = std::stoi(val); } catch (...) {
                    set_error_from_string(error, "set_params: 'f' must be an integer");
                    return false;
                }
                if (!set_f(v, error)) return false;

            } else if (key == "n_trees") {
                try { _params.n_trees = std::stoi(val); } catch (...) {
                    set_error_from_string(error, "set_params: 'n_trees' must be an integer");
                    return false;
                }

            } else if (key == "n_neighbors") {
                try {
                    _params.n_neighbors = std::stoi(val);
                } catch (...) {
                    set_error_from_string(error, "set_params: 'n_neighbors' must be an integer");
                    return false;
                }
                if (_params.n_neighbors < 1) {
                    set_error_from_string(error, "set_params: 'n_neighbors' must be >= 1");
                    return false;
                }

            } else if (key == "seed") {
                try { _params.seed = std::stoi(val); } catch (...) {
                    set_error_from_string(error, "set_params: 'seed' must be an integer");
                    return false;
                }

            } else if (key == "n_jobs") {
                try { _params.n_jobs = std::stoi(val); } catch (...) {
                    set_error_from_string(error, "set_params: 'n_jobs' must be an integer");
                    return false;
                }
                if (_params.n_jobs < -1 || _params.n_jobs == 0) {
                    set_error_from_string(error, "set_params: 'n_jobs' must be -1 or >= 1");
                    return false;
                }

            } else if (key == "schema_version") {
                try { _params.schema_version = std::stoi(val); } catch (...) {
                    set_error_from_string(error, "set_params: 'schema_version' must be an integer");
                    return false;
                }

            } else if (key == "verbose") {
                int v = 0;
                try { v = std::stoi(val); } catch (...) {
                    set_error_from_string(error, "set_params: 'verbose' must be an integer");
                    return false;
                }
                // Clamp to [-2, 2] — same range as set_verbose.
                _verbose = (v < -2) ? -2 : (v > 2) ? 2 : v;
                _params.verbose = _verbose;

            } else if (key == "l1_ratio") {
                try { _params.l1_ratio = std::stod(val); } catch (...) {
                    set_error_from_string(error, "set_params: 'l1_ratio' must be a number");
                    return false;
                }

            // --- boolean fields ---
            } else if (key == "prefault") {
                _params.prefault = (val == "true" || val == "1");

            // --- string fields (std::string; value-copied safely) ---
            } else if (key == "on_disk_path") {
                // std::string owns the storage, so we can safely copy val here.
                // Empty string means "no path configured" (equivalent to the old NULL).
                _params.on_disk_path = val;

            }
            // f_inferred, metric, dtype, index_dtype, wrapper_dtype, random_dtype
            // are read-only from the external API; silently ignored if supplied.
        }
        return true;
    }

  template<typename W>
  bool add_item_impl(S item, const W& w, int w_size, char** error=NULL) {
    // Developer note: w_size is the dimensionality of the vector pointed to by
    // w.  It is used for two purposes:
    //   (a) Lazy dimension inference: if _f==0 and this is the first item,
    //       we call set_f(w_size) to lock in the index dimension.
    //   (b) Dimension mismatch detection: if _f>0 and w_size>0 and they
    //       disagree, we fail fast rather than silently truncate/overrun.
    // w_size==0 is only legal when _f>0 (the caller knows the dimension).

    if (_loaded) {
      set_error_from_string(error, "You can't add an item to a loaded index");
      return false;
    }

    // --- Lazy dimension inference (BUG-6) ---
    if (_f == 0) {
      if (w_size <= 0) {
        set_error_from_string(error,
          "Dimension is not set (f=0). "
          "Call set_f(n) before add_item, or pass the vector size explicitly.");
        return false;
      }
      // First item determines the index dimension.
      if (!set_f(w_size, error)) {
        return false;  // set_f already wrote the error string
      }
    }

    // --- Dimension mismatch guard (ROB-2) ---
    if (w_size > 0 && w_size != _f) {
      // Build a compact, actionable message without dynamic allocation.
      // snprintf is safe here; the buffer is stack-allocated.
      char buf[128];
      std::snprintf(buf, sizeof(buf),
        "Vector size mismatch: expected %d dimensions, got %d",
        _f, w_size);
      set_error_from_string(error, buf);
      return false;
    }

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
      n->v[z] = static_cast<T>(w[z]);

    D::init_node(n, _f);

    if (item >= _n_items)
      _n_items = item + 1;

    return true;
  }

  bool on_disk_build(const char* file, char** error=NULL) noexcept override {
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

  bool build(int q, int n_threads=-1, char** error=NULL) noexcept override {
    if (_loaded) {
      set_error_from_string(error, "You can't build a loaded index");
      return false;
    }

    if (_built) {
      set_error_from_string(error, "You can't build a built index");
      return false;
    }

    // ARCH-4 + FUT-2: Resolve thread count via resolve_n_jobs() so all
    // parallel paths use one canonical rule.
    //   n_threads == -1 (call-site default) → honour _params.n_jobs.
    //   _params.n_jobs or n_threads == -1   → all hardware threads.
    //   Explicit n_threads > 0              → honour the override.
    if (n_threads == -1) {
      n_threads = resolve_n_jobs(_params.n_jobs);
    } else {
      n_threads = resolve_n_jobs(n_threads);
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
      std::memcpy(_get(_n_nodes + (S)i), _get(_roots[i]), _s);
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

  bool unbuild(char** error=NULL) noexcept override {
    if (_loaded) {
      set_error_from_string(error, "You can't unbuild a loaded index");
      return false;
    }

    _roots.clear();
    _n_nodes = _n_items;
    _built = false;

    return true;
  }

  bool save(const char* filename, bool prefault=false, char** error=NULL) noexcept override {
    if (!_built) {
      set_error_from_string(error, "You can't save an index that hasn't been built");
      return false;
    }
    if (_on_disk) {
      // on_disk_build files are already flushed to disk via mmap; no further
      // write needed.  They do not carry an AnnoyFileHeader (the mmap path
      // owns the fd and cannot prepend data without a full rewrite).
      return true;
    }

    // Delete existing file first (avoids partial-overwrite corruption on short
    // writes and matches the original behaviour — see issue #335).
#ifndef _MSC_VER
    unlink(filename);
#else
    _unlink(filename);
#endif

    FILE* f = fopen(filename, "wb");
    if (f == NULL) {
      set_error_from_errno(error, "Unable to open");
      return false;
    }

    // FUT-4: Write the file prologue so load() can validate type/metric/size
    // before touching any node data.
    AnnoyFileHeader hdr{};
    hdr.magic          = ANNOY_FILE_MAGIC;
    hdr.header_version = ANNOY_FILE_VERSION;
    hdr.sizeof_T       = static_cast<uint8_t>(sizeof(T));
    hdr.sizeof_S       = static_cast<uint8_t>(sizeof(S));
    hdr.node_size      = static_cast<uint32_t>(_s);
    hdr.n_dimensions   = static_cast<uint32_t>(_f);
    hdr.n_items        = static_cast<uint64_t>(_n_items);
    hdr.metric_hash    = annoy_fnv1a32(D::name());
    hdr.reserved       = 0u;
    std::memset(hdr.reserved2, 0, sizeof(hdr.reserved2));

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) {
      set_error_from_errno(error, "Unable to write file header");
      fclose(f);
      return false;
    }

    if (fwrite(_nodes, _s, _n_nodes, f) != static_cast<size_t>(_n_nodes)) {
      set_error_from_errno(error, "Unable to write");
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
    _mmap_base = NULL;  // cleared on every unload; set only by load() for headered files
    _mmap_size = 0;
  }

  void unload() noexcept override {
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
        // When load() mapped a headered file (FUT-4), _nodes points into
        // the mapping at offset sizeof(AnnoyFileHeader), not at the base.
        // munmap must receive the true base and total size stored in
        // _mmap_base/_mmap_size.  For legacy headerless files _mmap_base
        // is NULL and _nodes is the true base (_n_nodes * _s bytes).
        if (_mmap_base) {
          munmap(_mmap_base, _mmap_size);
        } else {
          munmap(_nodes, _n_nodes * _s);
        }
      } else if (_nodes) {
        // We have heap allocated data
        free(_nodes);
      }
    }
    reinitialize();
    if (_verbose) annoylib_showUpdate("unloaded\n");
  }

  bool load(const char* filename, bool prefault=false, char** error=NULL) noexcept override {
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
      set_error_from_string(error, "Size of file is zero");
#ifndef _MSC_VER
      close(_fd);
#else
      _close(_fd);
#endif
      _fd = 0;
      return false;
    }

    // FUT-4: Detect AnnoyFileHeader prologue.
    // Read the first 4 bytes as a candidate magic number.  If it matches
    // ANNOY_FILE_MAGIC we validate the full 64-byte header and skip past it
    // before mapping node data, catching type/metric/version mismatches early.
    // If it does not match we treat the file as a legacy headerless index so
    // existing saved files continue to load without rebuilding.
    //
    // CRITICAL: lseek_getsize() uses lseek(fd, 0, SEEK_END) on POSIX and
    // _lseeki64(fd, 0, SEEK_END) on Windows.  Both leave the fd position AT
    // EOF.  We must seek back to byte 0 BEFORE reading candidate_magic;
    // otherwise ::read returns 0 bytes (EOF), the magic check silently fails,
    // and the file is treated as a legacy headerless index.  For any index
    // file whose total size is not divisible by _s (e.g. 64-byte header +
    // N*36-byte nodes gives file_size%36 == 28 for Angular f=3, S=uint64_t)
    // this produces the misleading "not a multiple of node size" error.
    off_t node_data_offset = 0;
    {
      uint32_t candidate_magic = 0u;
#ifndef _MSC_VER
      lseek(_fd, 0, SEEK_SET);  // rewind after lseek_getsize left fd at EOF
      ssize_t r = ::read(_fd, &candidate_magic, sizeof(candidate_magic));
      lseek(_fd, 0, SEEK_SET);  // rewind again so full 64-byte header read starts at byte 0
#else
      _lseek(_fd, 0, SEEK_SET);
      int r = ::_read(_fd, &candidate_magic, static_cast<unsigned>(sizeof(candidate_magic)));
      _lseek(_fd, 0, SEEK_SET);
#endif
      if (r == static_cast<decltype(r)>(sizeof(candidate_magic))
          && candidate_magic == ANNOY_FILE_MAGIC)
      {
        AnnoyFileHeader hdr{};
#ifndef _MSC_VER
        ssize_t hr = ::read(_fd, &hdr, sizeof(hdr));
#else
        int hr = ::_read(_fd, &hdr, static_cast<unsigned>(sizeof(hdr)));
#endif
        if (hr != static_cast<decltype(hr)>(sizeof(hdr))) {
          set_error_from_string(error, "File header truncated");
#ifndef _MSC_VER
          close(_fd);
#else
          _close(_fd);
#endif
          _fd = 0;
          return false;
        }

        if (hdr.header_version != ANNOY_FILE_VERSION) {
          char buf[128];
          std::snprintf(buf, sizeof(buf),
            "Unsupported index file version %u (expected %u). Rebuild the index.",
            static_cast<unsigned>(hdr.header_version),
            static_cast<unsigned>(ANNOY_FILE_VERSION));
          set_error_from_string(error, buf);
#ifndef _MSC_VER
          close(_fd);
#else
          _close(_fd);
#endif
          _fd = 0;
          return false;
        }

        if (hdr.sizeof_T != static_cast<uint8_t>(sizeof(T)) ||
            hdr.sizeof_S != static_cast<uint8_t>(sizeof(S)))
        {
          char buf[192];
          std::snprintf(buf, sizeof(buf),
            "Type size mismatch: file sizeof(T)=%u sizeof(S)=%u, "
            "instance sizeof(T)=%u sizeof(S)=%u.",
            static_cast<unsigned>(hdr.sizeof_T), static_cast<unsigned>(hdr.sizeof_S),
            static_cast<unsigned>(sizeof(T)), static_cast<unsigned>(sizeof(S)));
          set_error_from_string(error, buf);
#ifndef _MSC_VER
          close(_fd);
#else
          _close(_fd);
#endif
          _fd = 0;
          return false;
        }

        uint32_t expected_metric = annoy_fnv1a32(Distance::name());
        if (hdr.metric_hash != 0u && hdr.metric_hash != expected_metric) {
          char buf[192];
          std::snprintf(buf, sizeof(buf),
            "Metric mismatch: file metric hash 0x%08X, expected 0x%08X (%s).",
            static_cast<unsigned>(hdr.metric_hash),
            static_cast<unsigned>(expected_metric), Distance::name());
          set_error_from_string(error, buf);
#ifndef _MSC_VER
          close(_fd);
#else
          _close(_fd);
#endif
          _fd = 0;
          return false;
        }

        if (_s > 0 && hdr.node_size != static_cast<uint32_t>(_s)) {
          char buf[192];
          std::snprintf(buf, sizeof(buf),
            "Node size mismatch: file node_size=%u, instance _s=%u (f=%d).",
            static_cast<unsigned>(hdr.node_size),
            static_cast<unsigned>(_s), _f);
          set_error_from_string(error, buf);
#ifndef _MSC_VER
          close(_fd);
#else
          _close(_fd);
#endif
          _fd = 0;
          return false;
        }

        // Lazy-init: infer _f from header when _f==0.
        if (_f == 0 && hdr.n_dimensions > 0) {
          if (!set_f(static_cast<int>(hdr.n_dimensions), error)) {
#ifndef _MSC_VER
            close(_fd);
#else
            _close(_fd);
#endif
            _fd = 0;
            return false;
          }
        }

        node_data_offset = static_cast<off_t>(sizeof(AnnoyFileHeader));

        if (_verbose > 0) {
          annoylib_showUpdate(
            "load: header v%u — metric=0x%08X f=%u n_items=%llu\n",
            static_cast<unsigned>(hdr.header_version),
            static_cast<unsigned>(hdr.metric_hash),
            static_cast<unsigned>(hdr.n_dimensions),
            static_cast<unsigned long long>(hdr.n_items));
        }
      } else {
        if (_verbose > 0) {
          annoylib_showUpdate("load: no header found — loading as legacy headerless index\n");
        }
      }
    }

    off_t node_data_size = size - node_data_offset;

    if (node_data_size <= 0) {
      set_error_from_string(error, "No node data after header");
#ifndef _MSC_VER
      close(_fd);
#else
      _close(_fd);
#endif
      _fd = 0;
      return false;
    }

    if (_s > 0 && static_cast<size_t>(node_data_size) % static_cast<size_t>(_s)) {
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

    // FUT-4 mmap alignment fix:
    // mmap(2) requires its offset argument to be a multiple of the page size
    // (typically 4096).  sizeof(AnnoyFileHeader) = 64 is not page-aligned, so
    // we cannot pass node_data_offset directly as the mmap offset.
    //
    // Solution: always map the ENTIRE file from offset 0.  When a header is
    // present (node_data_offset > 0) we advance the _nodes pointer past the
    // header and record the true base/size in _mmap_base/_mmap_size so that
    // unload() can munmap the correct region.  For legacy headerless files
    // node_data_offset == 0 so _nodes == map base and _mmap_base stays NULL.
    const size_t map_size = static_cast<size_t>(size);  // entire file
    void* map_base = mmap(0, map_size, PROT_READ, flags, _fd, 0);
    if (map_base == MAP_FAILED) {
      set_error_from_errno(error, "Unable to mmap");
#ifndef _MSC_VER
      close(_fd);
#else
      _close(_fd);
#endif
      _fd = 0;
      return false;
    }
    if (node_data_offset > 0) {
      // Headered file: _nodes points past the header; munmap needs true base.
      _mmap_base = map_base;
      _mmap_size = map_size;
      _nodes = (Node*)((uint8_t*)map_base + static_cast<size_t>(node_data_offset));
    } else {
      // Legacy headerless file: _nodes IS the map base; _mmap_base stays NULL.
      _nodes = (Node*)map_base;
    }

    _n_nodes = (_s > 0) ? static_cast<S>(node_data_size / static_cast<off_t>(_s)) : S(0);

    // Keep capacity in sync for serialize() / memory-usage queries.
    _nodes_size = _n_nodes;

    // Scan backwards to find root nodes.
    //
    // Roots are stored at the tail of the node array: the last `n_trees`
    // nodes all have the same n_descendants value (the total item count).
    //
    // BUG-1 / ROB-5 fix: the original code used
    //
    //   S m = -1;
    //   for (S i = _n_nodes - 1; i >= 0; i--) { if (m == -1 || ...) }
    //
    // For unsigned S (uint32_t, uint64_t):
    //   - "-1" wraps to S::max, making the initial sentinel match any node
    //     whose n_descendants == S::max — silent wrong result.
    //   - "i >= 0" is always true for unsigned types — infinite loop.
    //
    // Fix: use a bool flag instead of a sentinel value, and use the
    // post-decrement idiom "i-- > 0" which is safe for both signed and
    // unsigned S.
    _roots.clear();
    if (_n_nodes > S(0)) {
      S m_val = S(0);
      bool m_set = false;
      for (S i = _n_nodes; i-- > S(0); ) {
        S k = _get(i)->n_descendants;
        if (!m_set || k == m_val) {
          _roots.push_back(i);
          m_val = k;
          m_set = true;
        } else {
          break;
        }
      }
      // hacky fix: since the last root precedes the copy of all roots, delete it
      if (_roots.size() > 1 &&
          _get(_roots.front())->children[0] == _get(_roots.back())->children[0]) {
        _roots.pop_back();
      }
      _n_items = m_set ? m_val : S(0);
      if (_verbose > 0) {
        annoylib_showUpdate("found %zu roots with degree %ld\n",
                            _roots.size(), (long)_n_items);
      }
    }
    _loaded = true;
    _built  = true;
    return true;
  }

  T get_distance(S i, S j) const noexcept override {
    return D::normalized_distance(D::distance(_get(i), _get(j), _f));
  }

  void get_nns_by_item(S item, size_t n, int search_k, std::vector<S>* result, std::vector<T>* distances) const noexcept override {
    // TODO: handle OOB
    const Node* m = _get(item);
    _get_all_nns(m->v, n, search_k, result, distances);
  }

  void get_nns_by_vector(const T* w, size_t n, int search_k, std::vector<S>* result, std::vector<T>* distances) const noexcept override {
    _get_all_nns(w, n, search_k, result, distances);
  }

  S get_n_items() const noexcept override {
    return _n_items;
  }

  S get_n_trees() const noexcept override {
    return (S)_roots.size();
  }

  void set_verbose(bool v) noexcept override {
    // Bool accepted for interface compat. true → Info(1), false → Quiet(0).
    set_verbose_level(v ? annoy_verbose_level(AnnoyVerbose::Info)
                        : annoy_verbose_level(AnnoyVerbose::Quiet));
  }

  /// Typed enum overload — preferred; maps directly to leveled storage.
  void set_verbose(AnnoyVerbose level) noexcept {
    set_verbose_level(annoy_verbose_level(level));
  }

  // Single implementation point for verbosity changes.
  void set_verbose_level(int level) noexcept {
    _verbose = std::max(-2, std::min(2, level));
    _params.verbose = _verbose;
  }

  void get_item(S item, T* v) const noexcept override {
    // TODO: handle OOB
    Node* m = _get(item);
    std::memcpy(v, m->v, (_f) * sizeof(T));
  }

  void set_seed(R seed) noexcept override {
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

  std::vector<uint8_t> serialize(char** error=NULL) const noexcept override {
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

  bool deserialize(std::vector<uint8_t>* bytes, bool prefault=false, char** error=NULL) noexcept override {
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
        std::memcpy(out, p, sz);
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
      std::memcpy(&_roots[0], bytes_buffer, roots_bytes);
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

    std::memcpy(_nodes, bytes_buffer, nodes_bytes);

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

    // ARCH-5: Compute new_nodes_size in size_t to avoid narrowing overflow
    // when S is a small type (int8_t, uint16_t, etc.).  We then cap against
    // the maximum representable value of S before casting back.
    const size_t s_max = static_cast<size_t>(std::numeric_limits<S>::max());
    const size_t current = static_cast<size_t>(_nodes_size);
    const size_t requested = static_cast<size_t>(n);

    // Candidate size: max(requested, floor((current + 1) * growth_factor)).
    // All arithmetic is in size_t — no narrowing until the final cast.
    size_t grown = static_cast<size_t>(
        static_cast<double>(current + 1u) * reallocation_factor);
    size_t candidate = (grown > requested) ? grown : requested;

    // Clamp to S::max so the cast below is defined.
    if (candidate > s_max) {
      candidate = s_max;
    }

    // If even clamped max < requested the index has exceeded S capacity.
    if (candidate < requested) {
      return false;  // S is too narrow for this many items; caller must choose a wider index_dtype
    }

    S new_nodes_size = static_cast<S>(candidate);

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
  // Gray region: [0.95, 0.9999999]
  // Retry region: (0.9999999, 1.0]
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
      // Using std::memcpy instead of std::copy for MSVC compatibility. #235
      // Only copy when necessary to avoid crash in MSVC 9. #293
      if (!indices.empty())
        std::memcpy(m->children, &indices[0], indices.size() * sizeof(S));

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
    while (_split_imbalance(children_indices[0], children_indices[1]) > 0.9999999) {
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
    std::memcpy(_get(item), m, _s);
    threaded_build_policy.unlock_shared_nodes();

    return item;
  }

  void _get_all_nns(const T* v, size_t n, int search_k, std::vector<S>* result, std::vector<T>* distances) const {
    Node* v_node = (Node *)alloca(_s);
    D::template zero_value<Node>(v_node);
    std::memcpy(v_node->v, v, sizeof(T) * _f);
    D::init_node(v_node, _f);

    std::priority_queue<std::pair<T, S> > q;

    if (search_k == -1) {
      search_k = n * _roots.size();
    }

    for (size_t i = 0; i < _roots.size(); i++) {
      q.push(std::make_pair(Distance::template pq_initial_value<T>(), _roots[i]));
    }

    std::vector<S> nns;
    while (nns.size() < (size_t)search_k && !q.empty()) {
      const std::pair<T, S>& top = q.top();
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
        q.push(std::make_pair(D::pq_distance(d, margin, 1), static_cast<S>(nd->children[1])));
        q.push(std::make_pair(D::pq_distance(d, margin, 0), static_cast<S>(nd->children[0])));
      }
    }

    // Deduplicate and compute distances.
    //
    // Sort by ID so each ID appears in a contiguous run; then skip duplicates.
    //
    // BUG-1 fix: the original code used
    //
    //   S last = -1;
    //   if (j == last) continue;
    //
    // For unsigned S (uint32_t, uint64_t), -1 wraps to S::max.  Any item
    // whose ID == S::max would be treated as already seen and skipped on its
    // first encounter — a silent result corruption.
    //
    // Fix: use a bool flag for "have we seen the previous ID yet".
    std::sort(nns.begin(), nns.end());
    std::vector<std::pair<T, S> > nns_dist;
    bool have_last = false;
    S last = S(0);
    for (size_t i = 0; i < nns.size(); i++) {
      S j = nns[i];
      if (have_last && j == last)
        continue;
      have_last = true;
      last = j;
      if (_get(j)->n_descendants == 1)  // guard for obscure case, #284
        nns_dist.push_back(std::make_pair(D::distance(v_node, _get(j), _f), j));
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
 * @tparam S Index type (int32_t | int64_t | uint32_t | uint64_t)
 * @tparam T External data type (float16_t | float | double | float128_t) - user-facing API
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
                "S must be one of: int8_t, int16_t, int32_t, int64_t, "
                "uint8_t, uint16_t, uint32_t, uint64_t");
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value ||
                std::is_same<T, float16_t>::value || std::is_same<T, float128_t>::value,
                "T must be float, double, float16_t, or float128_t for HammingWrapper external interface");
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
  int _verbose;                     // Verbose logging level: 0=quiet 1=info 2=debug (clamped [-2,2]; int, not bool)
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

#if __cplusplus >= 201103L
  std::atomic<bool> _build_failed;  // Thread-safe build failure flag
#else
  bool _build_failed;
#endif

// ⚠️ ← declared first assign first ❌ This violates the declaration order.
private:
  // -------------------- Serialization constants --------------------
  // HAMMING_MAGIC and HAMMING_VERSION are declared further below, after
  // HammingHeader (they were moved there when the header was bumped to v2
  // to keep the version comment co-located with the struct definition).

  // static constexpr size_t BITS_PER_WORD = sizeof(InternalT) * 8;
  static constexpr int BITS_PER_WORD = sizeof(InternalT) * 8;

  // -------------------- Dimensions ---------------------
  bool _f_inferred;
  S _f_external;  // External dimension (number of bits/bools from user)
  S _f_internal;  // Internal dimension (number of words to store)

  // -------------------- Underlying Annoy index ---------------------
  AnnoyIndex<S, InternalT, Hamming, Random, ThreadedBuildPolicy> _index;

  // -------------------- Serialization header -----------------------
  // Version history:
  //   1 — original; f_external/f_internal/n_items were uint32_t (truncated for S=int64_t/uint64_t)
  //   2 — promoted f_external, f_internal, n_items to uint64_t; header is 48 bytes
  static const uint32_t HAMMING_MAGIC   = 0x414E4E59u; // 'ANNY'
  static const uint32_t HAMMING_VERSION = 2u;

  struct HammingHeader {
    uint32_t magic;       // must equal HAMMING_MAGIC
    uint32_t version;     // must equal HAMMING_VERSION (2)
    uint64_t f_external;  // user-visible bit dimension (was uint32_t in v1)
    uint64_t f_internal;  // packed-word dimension      (was uint32_t in v1)
    uint64_t n_items;     // item count                 (was uint32_t in v1)
    uint64_t reserved;    // must be zero
    uint64_t reserved2;   // must be zero; pad to 48 bytes total
  };
  static_assert(sizeof(HammingHeader) == 48,
    "HammingHeader size changed — update HAMMING_VERSION and save/load paths");

  // -------------------- Utilities ----------------------------------
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
  // NOTE: HammingWrapper() = default is NOT used because _f_external and
  // _f_internal (now non-const, but previously const) have no default member
  // initializers. The default constructor explicitly zeroes them so any
  // accidental default-constructed instance is in a deterministic UNSET state
  // rather than UB. In practice _ensure_index() always calls HammingWrapper(f).
  HammingWrapper()
    : _verbose(0)
    , _f_inferred(true)
    , _f_external(S(0))
    , _f_internal(S(0))
    , _index(0)
    , _build_failed(false)
  {}
  explicit HammingWrapper(
    int f = 0,                        // DEFAULT_DIMENSION (0 = lazy inference)
    int n_trees = -1,
    int n_neighbors = 5,
    const std::string& on_disk_path = "",
    bool prefault = false,
    int seed = 0,
    int verbose = 0,                  // ARCH-3: int not bool; clamped to [-2,2]
    int schema_version = 0,
    int n_jobs = 1,
    double l1_ratio = 0.0
  )
    // Clamp verbose here so _verbose and the forwarded value to _index are
    // both consistently bounded.  Use a named local via the ternary rather
    // than computing it twice in the initializer-list.
    :  _verbose((verbose < -2) ? -2 : (verbose > 2) ? 2 : verbose)
    , _f_inferred(f == 0)
    , _f_external(static_cast<S>(f))
    , _f_internal((static_cast<S>(f) + BITS_PER_WORD - 1) / BITS_PER_WORD)
    , _index(
      // Pass internal (packed-word) dimension to the underlying AnnoyIndex.
      // When f==0 (lazy), both _f_external and _f_internal are 0; set_f()
      // on the first add_item call will recompute them.
      (static_cast<S>(f) + BITS_PER_WORD - 1) / BITS_PER_WORD,
      n_trees,
      n_neighbors,
      on_disk_path,
      prefault,
      seed,
      _verbose,                       // pass the clamped int, not the raw arg
      schema_version,
      n_jobs
    )
  {
    // Populate AnnoyParams mirror so get_params() / set_params() work.
    _params.f             = f;
    _params.n_trees       = n_trees;
    _params.n_neighbors   = n_neighbors;
    _params.on_disk_path  = on_disk_path;
    _params.prefault      = prefault;
    _params.seed          = seed;
    _params.verbose       = _verbose;  // store clamped int
    _params.schema_version= schema_version;
    _params.n_jobs        = n_jobs;
    _params.l1_ratio      = l1_ratio;
    _params.f_inferred    = (f == 0);

    // Validate parameters; log if verbosity allows.
    char* err = NULL;
    if (!_params.validate(&err)) {
      if (_verbose > 0 && err != NULL) {
        annoylib_showUpdate("HammingWrapper parameter validation failed: %s\n", err);
      }
      std::free(err);
    }
    if (_verbose > 0) {
      annoylib_showUpdate(
        "HammingWrapper initialized: f=%d n_trees=%d n_neighbors=%d n_jobs=%d verbose=%d\n",
        _params.f, _params.n_trees, _params.n_neighbors, _params.n_jobs, _verbose);
    }
  }
  virtual ~HammingWrapper() { unload(); }

  int get_f() const noexcept override {
    // Return the user-visible external dimension (number of bits/bools the user
    // works with), NOT the internal packed-word dimension (_f_internal).
    //
    // Rationale: The _w bridge methods in AnnoyIndexInterface (add_item_w,
    // get_item_w, get_nns_by_vector_w) call get_f() to size temporary float/
    // double vectors before calling the typed add_item / get_item /
    // get_nns_by_vector.  HammingWrapper's typed methods expect vectors of
    // length _f_external (the bit count), so get_f() MUST return _f_external.
    //
    // Previously this returned _index.get_f() == _f_internal (the packed-word
    // count, e.g. 1 for 8 bits packed into uint64_t).  That caused the bridge
    // to allocate a 1-element vector, write only embedding[0] into it, and
    // pass the 1-element array to add_item / get_nns_by_vector_w, which then
    // read _f_external elements -- 7 of them garbage -- corrupting all stored
    // Hamming data and producing wrong distances (e.g. dist=0 for different
    // vectors, wrong self-nearest-neighbor results).
    return static_cast<int>(_f_external);
  }

  // Set dimension (must be called before add_item if using default constructor)
  bool set_f(int f, char** error = NULL) noexcept override {
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

    // Update external/internal dimensions.
    // Previously these were const and therefore could not be updated here —
    // that made get_f() return the stale construction-time value (0 for lazy
    // mode) even after set_f() succeeded.  Now that const has been removed
    // these assignments are both valid and required for correctness.
    _f_external = static_cast<S>(f);
    _f_internal = static_cast<S>(
      (_f_external + BITS_PER_WORD - 1) / BITS_PER_WORD
    );
    _f_inferred = false;

    return _index.set_f(static_cast<int>(_f_internal), error);
  }

  bool get_params(
      std::vector<std::pair<std::string, std::string>>& params
  ) const noexcept override {
      params.clear();
      params.emplace_back("f", std::to_string(_f_external));
      return true;
  }
  bool set_params(
      const std::vector<std::pair<std::string, std::string>>& params,
      char** error = NULL
  ) noexcept override {
      for (const auto& kv : params) {
          if (kv.first == "f") {
              int f = std::stoi(kv.second);
              return set_f(f, error);
          }
      }

      if (error) {
        // FIX — use the project-canonical allocator
        // *error = strdup("missing parameter: f");
        *error = dup_cstr("missing parameter: f");
      }
      return false;
  }

  // -------------------- AnnoyIndexInterface ------------------------
  bool add_item(S item, const T* embedding, char** error = NULL) noexcept override {
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

  bool build(int n_trees = -1, int n_threads = -1, char** error = NULL) noexcept override {
    return _index.build(n_trees, n_threads, error);
  }

  // -------------------- Serialization ------------------------------
  std::vector<uint8_t> serialize(char** error) const noexcept override {
    HammingHeader hdr{};
    hdr.magic      = HAMMING_MAGIC;
    hdr.version    = HAMMING_VERSION;
    hdr.f_external = static_cast<uint64_t>(_f_external);
    hdr.f_internal = static_cast<uint64_t>(_f_internal);
    hdr.n_items    = static_cast<uint64_t>(_index.get_n_items());
    hdr.reserved   = 0U;
    hdr.reserved2  = 0U;

    std::vector<uint8_t> out(sizeof(hdr));
    std::memcpy(out.data(), &hdr, sizeof(hdr));

    std::vector<uint8_t> payload = _index.serialize(error);
    if (payload.empty() && error != NULL && *error != NULL) {
      return std::vector<uint8_t>();
    }

    out.insert(out.end(), payload.begin(), payload.end());
    return out;
  }

  bool deserialize(std::vector<uint8_t>* bytes, bool prefault = false, char** error = NULL) noexcept override {
    if (!bytes || bytes->size() < sizeof(HammingHeader)) {
      if (error) *error = dup_cstr("Invalid or empty Hamming index");
      return false;
    }

    HammingHeader hdr;
    std::memcpy(&hdr, bytes->data(), sizeof(hdr));

    if (hdr.magic != HAMMING_MAGIC ||
        hdr.version != HAMMING_VERSION ||
        hdr.reserved != 0 ||
        hdr.reserved2 != 0 ||
        hdr.f_external != static_cast<uint64_t>(_f_external) ||
        hdr.f_internal != static_cast<uint64_t>(_f_internal)) {
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
  T get_distance(S i, S j) const noexcept override {
    return static_cast<T>(
        _clip_distance(_index.get_distance(i, j)));
  }

  void get_item(S indice,
                T* embedding) const noexcept override {
    std::vector<InternalT> packed(_f_internal);
    _index.get_item(indice, packed.data());
    _unpack(packed.data(), embedding);
  }

  S get_n_items() const noexcept override {
    return _index.get_n_items();
  }

  S get_n_trees() const noexcept override {
    return _index.get_n_trees();
  }

  // -------------------- Nearest Neighbor Queries -----------------------
  void get_nns_by_item(S               query_indice,
                       size_t          n,
                       int             search_k,
                       std::vector<S>* result,
                       std::vector<T>* distances) const noexcept override {
    if (distances) {
      std::vector<InternalT> internal_distances;
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
        (*distances)[i] = static_cast<T>(d);
    }
    } else {
      _index.get_nns_by_item(query_indice,
                             n,
                             search_k,
                             result,
                             NULL);
    }
  }
  void get_nns_by_vector(const T*        query_embedding,
                         size_t          n,
                         int             search_k,
                         std::vector<S>* result,
                         std::vector<T>* distances) const noexcept override {
    std::vector<InternalT> packed_query(_f_internal, InternalT(0));
    _pack(query_embedding, &packed_query[0]);

    if (distances) {
      std::vector<InternalT> internal_distances;
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
        (*distances)[i] = static_cast<T>(d);
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
            char** error = NULL) noexcept override {
    return _index.load(filename, prefault, error);
  }

  bool save(const char* filename,
            bool prefault=false,
            char** error = NULL) noexcept override {
    return _index.save(filename, prefault, error);
  }

  bool on_disk_build(const char* filename,
                     char** error = NULL) noexcept override {
    return _index.on_disk_build(filename, error);
  }

  bool unbuild(char** error = NULL) noexcept override {
    return _index.unbuild(error);
  }

  void unload() noexcept override {
    _index.unload();
  }

  void set_seed(R seed) noexcept override {
    _index.set_seed(seed);
  }

  void set_verbose(bool v) noexcept override {
    _verbose = v ? annoy_verbose_level(AnnoyVerbose::Info)
                 : annoy_verbose_level(AnnoyVerbose::Quiet);
    _params.verbose = _verbose;
    _index.set_verbose(v);
  }

  /// Typed enum overload — delegates through to the underlying AnnoyIndex.
  void set_verbose(AnnoyVerbose level) noexcept {
    _verbose = annoy_verbose_level(level);
    _params.verbose = _verbose;
    _index.set_verbose(level);
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
    // Invariant: n_threads >= 1.
    // build() in AnnoyIndex resolves -1 → hardware_concurrency() via
    // resolve_n_jobs() before reaching this policy, so -1 must never arrive
    // here.  The assert documents and enforces that contract; if a future
    // call path bypasses resolve_n_jobs() it will fire immediately in debug
    // builds rather than silently allocating a zero-element thread vector.
    assert(n_threads >= 1 && "n_threads must be resolved via resolve_n_jobs() before calling policy::build");

    AnnoyIndexMultiThreadedBuildPolicy threaded_build_policy;

    std::vector<std::thread> threads(static_cast<size_t>(n_threads));

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
