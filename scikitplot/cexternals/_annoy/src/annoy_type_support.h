// scikitplot/cexternals/_annoy/src/annoy_type_support.h
// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Float-type support registry (CY-016). Turns the compile-time facts about each
// element type (which are otherwise implicit in annoylib.h) into a single,
// runtime-queryable capability report covering the whole ladder:
//
//   bool -> int8/uint8 -> float16 -> float32 -> float64 -> float80/96 ->
//   float128 -> float256 -> float512
//
// For each type it reports: storage size, effective mantissa bits, the support
// TIER (native / runtime-dispatched / emulated / unavailable), whether the C++
// type exists in this build, whether Index(dtype=...) accepts it today, and
// whether its public I/O precision is capped by the double `_w` bridge (CY-012).
//
// The 256/512-bit tiers have NO native C++ type; they are provided only when a
// multiprecision backend is compiled in (define ANNOY_ENABLE_MULTIPRECISION and
// have boost::multiprecision headers). Otherwise they honestly report
// "unavailable" — never a silent long-double alias.

#pragma once

#include "annoylib.h"  // element types + ANNOY_HAS_* capability macros

#include <cfloat>
#include <cstddef>
#include <string>

// ---------------------------------------------------------------------------
// Extended-precision type ladder (future-oriented)
// ---------------------------------------------------------------------------

// float80 / float96: x87 80-bit extended precision. Native on x86 GCC/Clang;
// stored in 12 or 16 bytes depending on ABI. This is a real, available type.
#ifndef ANNOY_FLOAT80_DEFINED
#define ANNOY_FLOAT80_DEFINED
typedef long double float80_t;
#define ANNOY_HAS_FLOAT80 1
#endif

// float256 / float512: no native or hardware type exists. Compile a software
// backend ONLY when explicitly enabled AND boost::multiprecision is present.
#if defined(ANNOY_ENABLE_MULTIPRECISION)
#  if defined(__has_include)
#    if __has_include(<boost/multiprecision/cpp_bin_float.hpp>)
#      include <boost/multiprecision/cpp_bin_float.hpp>
// cpp_bin_float<N> ~ N bits of precision; 237/493 give ~256/512-bit mantissas.
typedef boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<237> > float256_t;
typedef boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<493> > float512_t;
#      define ANNOY_HAS_MULTIPRECISION 1
#    endif
#  endif
#endif

namespace annoy_support {

// Support tier for an element type.
enum class Tier { Native, RuntimeDispatched, Emulated, Unavailable };

inline const char* tier_name(Tier t) {
  switch (t) {
    case Tier::Native:            return "native";
    case Tier::RuntimeDispatched: return "runtime-dispatched";
    case Tier::Emulated:          return "emulated";
    default:                      return "unavailable";
  }
}

struct TypeInfo {
  const char* name;
  std::size_t size_bytes;      // 0 when unavailable
  int         mantissa_bits;   // effective precision; 0 when unavailable
  Tier        tier;
  bool        available;       // the C++ type exists in this build
  bool        usable_as_dtype; // accepted by Index(dtype=...) today
  bool        io_precision_capped;  // capped by the double `_w` bridge (CY-012)
  const char* note;
};

// -- individual tiers -------------------------------------------------------

inline TypeInfo info_float16() {
#if defined(ANNOY_HAS_RUNTIME_DISPATCH_FLOAT16)
  Tier t = Tier::RuntimeDispatched;
  const char* note = "scalar + F16C chosen at runtime; portable";
#elif defined(ANNOY_HAS_NATIVE_FLOAT16)
  Tier t = Tier::Native;
  const char* note = "ARM __fp16";
#elif defined(ANNOY_HAS_F16C_FLOAT16)
  Tier t = Tier::Native;
  const char* note = "x86 F16C (compile-time; not portable)";
#else
  Tier t = Tier::Emulated;
  const char* note = "portable scalar";
#endif
  return TypeInfo{"float16", sizeof(float16_t), 11, t, true, true, true, note};
}

inline TypeInfo info_float32() {
  return TypeInfo{"float32", sizeof(float32_t), FLT_MANT_DIG, Tier::Native,
                  true, true, false, "IEEE binary32"};
}

inline TypeInfo info_float64() {
  return TypeInfo{"float64", sizeof(float64_t), DBL_MANT_DIG, Tier::Native,
                  true, true, false, "IEEE binary64 (bridge width)"};
}

inline TypeInfo info_float80() {
  // The dtype dispatch always accepts float80 (data_types always includes it),
  // so it is always usable. Whether it is *distinct* from float128 depends on
  // the platform: on GCC/Clang float128_t is native __float128 (distinct); where
  // float128_t is already long double, float80 is redundant with float128.
#if defined(ANNOY_HAS_FLOAT128)
  const char* note = "x87 80-bit extended (long double); distinct usable dtype";
#else
  const char* note = "long double == float128 here; usable but redundant with float128";
#endif
  return TypeInfo{"float80", sizeof(float80_t), LDBL_MANT_DIG, Tier::Native,
                  true, true, true, note};
}

inline TypeInfo info_float128() {
#if defined(ANNOY_HAS_FLOAT128)
  return TypeInfo{"float128", sizeof(float128_t), 113, Tier::Native,
                  true, true, true, "__float128 / libquadmath (true quad)"};
#else
  return TypeInfo{"float128", sizeof(float128_t), LDBL_MANT_DIG, Tier::Emulated,
                  true, true, true, "long double (NOT true 128-bit)"};
#endif
}

inline TypeInfo info_float256() {
#if defined(ANNOY_HAS_MULTIPRECISION)
  return TypeInfo{"float256", sizeof(float256_t), 237, Tier::Emulated,
                  true, false, true, "boost cpp_bin_float; not yet a dtype"};
#else
  return TypeInfo{"float256", 0, 0, Tier::Unavailable, false, false, true,
                  "needs a multiprecision backend (boost/MPFR)"};
#endif
}

inline TypeInfo info_float512() {
#if defined(ANNOY_HAS_MULTIPRECISION)
  return TypeInfo{"float512", sizeof(float512_t), 493, Tier::Emulated,
                  true, false, true, "boost cpp_bin_float; not yet a dtype"};
#else
  return TypeInfo{"float512", 0, 0, Tier::Unavailable, false, false, true,
                  "needs a multiprecision backend (boost/MPFR)"};
#endif
}

// -- registry ---------------------------------------------------------------

inline int type_count() { return 7; }

inline TypeInfo type_at(int i) {
  switch (i) {
    case 0: return info_float16();
    case 1: return info_float32();
    case 2: return info_float64();
    case 3: return info_float80();
    case 4: return info_float128();
    case 5: return info_float256();
    default: return info_float512();
  }
}

// -- JSON report (the Cython-facing surface) --------------------------------

inline void append_escaped(std::string& out, const char* s) {
  out += '"';
  for (const char* p = s; p && *p; ++p) {
    if (*p == '"' || *p == '\\') out += '\\';
    out += *p;
  }
  out += '"';
}

inline std::string report_json() {
  std::string out = "[";
  for (int i = 0; i < type_count(); ++i) {
    TypeInfo t = type_at(i);
    if (i) out += ',';
    out += "{\"name\":";
    append_escaped(out, t.name);
    out += ",\"size_bytes\":" + std::to_string(t.size_bytes);
    out += ",\"mantissa_bits\":" + std::to_string(t.mantissa_bits);
    out += ",\"tier\":";
    append_escaped(out, tier_name(t.tier));
    out += ",\"available\":";       out += t.available ? "true" : "false";
    out += ",\"usable_as_dtype\":"; out += t.usable_as_dtype ? "true" : "false";
    out += ",\"io_precision_capped\":"; out += t.io_precision_capped ? "true" : "false";
    out += ",\"note\":";
    append_escaped(out, t.note);
    out += '}';
  }
  out += ']';
  return out;
}

}  // namespace annoy_support
