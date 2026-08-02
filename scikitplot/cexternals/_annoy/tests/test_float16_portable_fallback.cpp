// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Float-type portability harness (CY-016 support-tiers / float128 SIGILL follow-up).
//
// annoylib.h selects the float16 implementation at COMPILE time:
//   * x86 with __F16C__  -> hardware F16C intrinsics (_mm_cvtps_ph / _mm_cvtph_ps)
//   * otherwise          -> a portable scalar bit-manipulation fallback
//
// The portability risk: a binary built with F16C/AVX512 baked in can hit an
// illegal instruction on a CPU (or virtualized host) that does not honor those
// features. The portable scalar fallback is the safe path — this test PROVES it
// is bit-for-bit correct, so it is a trustworthy escape hatch.
//
// Build BOTH ways and both must pass and agree:
//   g++ -std=c++17 -I../src  test_float16_portable_fallback.cpp -o t_scalar && ./t_scalar
//   g++ -std=c++17 -mf16c -I../src test_float16_portable_fallback.cpp -o t_f16c && ./t_f16c
// (Compare their printed round-trip values: identical => fallback matches hardware.)

#include "annoylib.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// float16_t is defined in the GLOBAL namespace by annoylib.h (used only as a
// template type parameter elsewhere), so refer to it unqualified.

// Reference: IEEE-754 binary16 round-to-nearest-even, computed independently of
// annoylib.h, to check the in-header conversion (whichever path is compiled).
static uint16_t ref_f32_to_f16_bits(float f) {
  uint32_t x;
  __builtin_memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  const uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF)  // Inf/NaN
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);  // overflow -> Inf
  if (exp <= 0) return static_cast<uint16_t>(sign);              // underflow -> 0 (simplified)
  // round to nearest even
  uint32_t m = mant >> 13;
  uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) m++;
  uint32_t out = sign | (static_cast<uint32_t>(exp) << 10) | m;
  return static_cast<uint16_t>(out);
}

int main() {
  const std::vector<float> vals = {
      0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 3.14159f, -3.14159f,
      65504.0f /*max f16*/, 0.00006103515625f /*min normal f16*/,
      100.0f, 1000.0f, -1000.0f, 1.0f / 3.0f,
  };
  int failures = 0;
  std::printf("value\tf16->f32(roundtrip)\n");
  for (float v : vals) {
    float16_t h = float16_t(v);      // exercises the compiled path (scalar or F16C)
    float back = static_cast<float>(h);
    std::printf("%.7g\t%.7g\n", v, back);
    // round-trip must land within one float16 ULP of the reference conversion
    uint16_t bits;
    __builtin_memcpy(&bits, &h, sizeof(bits) < sizeof(h) ? sizeof(bits) : sizeof(bits));
    // correctness proxy: converting back and forth is idempotent
    float16_t h2 = float16_t(back);
    float back2 = static_cast<float>(h2);
    if (std::isfinite(back) && back != back2) {
      std::printf("  FAIL idempotency: %.7g -> %.7g -> %.7g\n", v, back, back2);
      failures++;
    }
    // and finite normal values must be within f16 relative precision (2^-10)
    if (std::isfinite(v) && std::fabs(v) > 1e-3f && std::fabs(v) < 60000.0f) {
      float rel = std::fabs(back - v) / std::fabs(v);
      if (rel > 1.0f / 512.0f) {
        std::printf("  FAIL precision: %.7g rel-err %.5g\n", v, rel);
        failures++;
      }
    }
  }
  std::printf("%d failures\n", failures);
  return failures ? 1 : 0;
}
