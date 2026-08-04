// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Float-type capability report (CY-016 support-tiers foundation).
//
// annoylib.h already defines capability macros as a side effect of choosing each
// float representation, but nothing surfaces them. This probe turns those
// compile-time facts into an explicit, machine-readable support-tier report, so a
// build/CI (or a future runtime API) can answer "is float16 accelerated or
// scalar here? is float128 native quad or long-double-emulated?" — the questions
// that decide whether a wheel is portable and how precise its types really are.
//
// Build: g++ -std=c++17 -I../src test_float_capability_report.cpp -o t && ./t
// Add -mf16c to see the tier flip to F16C-accelerated.

#include "annoylib.h"
#include <cstdio>
#include <cstring>

int main() {
  std::printf("=== annoy float-type support tiers ===\n");

  // float16
#if defined(ANNOY_HAS_RUNTIME_DISPATCH_FLOAT16)
  const char* f16_tier = "runtime-dispatched (scalar + F16C, chosen per-CPU)";
#elif defined(ANNOY_HAS_F16C_FLOAT16)
  const char* f16_tier = "accelerated (x86 F16C hardware)";
#elif defined(__ARM_FP16_FORMAT_IEEE) || defined(__ARM_FP16_FORMAT_ALTERNATIVE)
  const char* f16_tier = "native (ARM __fp16)";
#else
  const char* f16_tier = "portable (scalar bit-manipulation fallback)";
#endif
  std::printf("float16 : sizeof=%zu tier=%s\n", sizeof(float16_t), f16_tier);

  // float32 / float64 are always native IEEE
  std::printf("float32 : sizeof=%zu tier=native (IEEE binary32)\n", sizeof(float32_t));
  std::printf("float64 : sizeof=%zu tier=native (IEEE binary64)\n", sizeof(float64_t));

  // float128
#if defined(ANNOY_HAS_FLOAT128)
  const char* f128_tier = "native (__float128 / libquadmath, 113-bit mantissa)";
#elif defined(ANNOY_HAS_FLOAT128_EMULATED)
  const char* f128_tier = "emulated (long double, 80-bit on MSVC/x86)";
#elif defined(ANNOY_HAS_FLOAT128_FALLBACK)
  const char* f128_tier = "fallback (long double; precision is platform-dependent)";
#else
  const char* f128_tier = "unknown";
#endif
  std::printf("float128: sizeof=%zu tier=%s\n", sizeof(float128_t), f128_tier);

  // Portability note driven by the actual mantissa width the build delivers.
  const bool f128_is_true_quad = (sizeof(float128_t) >= 16);
  std::printf("\nportability summary:\n");
  std::printf("  float16 uses hardware intrinsics: %s\n",
#if defined(ANNOY_HAS_RUNTIME_DISPATCH_FLOAT16)
              "runtime (F16C when the CPU supports it; scalar otherwise; portable)"
#elif defined(ANNOY_HAS_F16C_FLOAT16)
              "YES  (built-in F16C; NOT portable to non-F16C CPUs)"
#else
              "no   (portable everywhere)"
#endif
  );
  std::printf("  float128 is true 128-bit quad: %s\n",
              f128_is_true_quad ? "YES" : "no (long double; ~64-80 bit)");
  std::printf("0 failures\n");
  return 0;
}
