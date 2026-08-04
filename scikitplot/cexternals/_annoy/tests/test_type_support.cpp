// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Regression test for the float-type support registry (annoy_type_support.h, CY-016).
// Checks the ladder is complete, tiers/widths are sane, unavailable tiers are
// honest (never a silent long-double alias), and the multiprecision gate degrades
// gracefully when the backend is absent.
//
// Build: g++ -std=c++17 -I../src test_type_support.cpp -o t && ./t
//   with backend gate: add -DANNOY_ENABLE_MULTIPRECISION (still compiles + reports
//   unavailable when boost is not installed).

#include "annoy_type_support.h"
#include <cstdio>
#include <cstring>
#include <string>

using namespace annoy_support;

int main() {
  int failures = 0;

  // 1. ladder is exactly the 7 expected entries, in order
  const char* expect[] = {"float16", "float32", "float64", "float80",
                          "float128", "float256", "float512"};
  if (type_count() != 7) { std::printf("FAIL: type_count != 7\n"); failures++; }
  for (int i = 0; i < type_count(); ++i) {
    TypeInfo t = type_at(i);
    if (std::strcmp(t.name, expect[i]) != 0) {
      std::printf("FAIL: [%d] name %s != %s\n", i, t.name, expect[i]); failures++;
    }
    // 2. an available type must have non-zero width + mantissa; unavailable => zero
    if (t.available && (t.size_bytes == 0 || t.mantissa_bits == 0)) {
      std::printf("FAIL: %s available but zero width/precision\n", t.name); failures++;
    }
    if (!t.available && t.tier != Tier::Unavailable) {
      std::printf("FAIL: %s unavailable but tier != Unavailable\n", t.name); failures++;
    }
    if (!t.available && t.size_bytes != 0) {
      std::printf("FAIL: %s unavailable but non-zero size (silent alias?)\n", t.name);
      failures++;
    }
  }

  // 3. float32/64 are native and NOT bridge-capped; wider types ARE capped
  if (info_float64().io_precision_capped) { std::printf("FAIL: float64 capped\n"); failures++; }
  if (!info_float128().io_precision_capped) { std::printf("FAIL: float128 not capped\n"); failures++; }

  // 4. float80 (long double) is available but not yet a usable dtype
  {
    TypeInfo f80 = info_float80();
    if (!f80.available) { std::printf("FAIL: float80 not available\n"); failures++; }
    // float80 is always usable (dispatch always accepts it); distinctness only
    // changes the note, not usability.
    if (!f80.usable_as_dtype) { std::printf("FAIL: float80 should be usable\n"); failures++; }
    if (f80.mantissa_bits != LDBL_MANT_DIG) {
      std::printf("FAIL: float80 mantissa %d != %d\n", f80.mantissa_bits, LDBL_MANT_DIG);
      failures++;
    }
  }

  // 5. multiprecision tiers reflect the compile gate honestly
  {
    TypeInfo f256 = info_float256();
#if defined(ANNOY_HAS_MULTIPRECISION)
    if (!f256.available) { std::printf("FAIL: MP enabled but float256 unavailable\n"); failures++; }
#else
    if (f256.available)  { std::printf("FAIL: no MP backend but float256 available\n"); failures++; }
#endif
  }

  // 6. JSON report is non-empty and well-formed-ish (balanced brackets)
  std::string j = report_json();
  if (j.empty() || j.front() != '[' || j.back() != ']') {
    std::printf("FAIL: report_json malformed\n"); failures++;
  }

  std::printf("%d failures\n", failures);
  return failures ? 1 : 0;
}
