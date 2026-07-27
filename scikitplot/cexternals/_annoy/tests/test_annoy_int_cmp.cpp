// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Regression test for ANNOY-CMP-001 (guide 6.13): range-correct integer clamp
// and sign-safe comparisons in annoy_int_cmp.h. Checks the REAL header across
// the full signed/unsigned x width property matrix against an __int128 oracle
// (which represents every int64/uint64 value exactly), including the cases the
// old bound-cast-into-S logic got wrong.
//
// Build & run (host):
//   g++ -std=c++17 -I<src> test_annoy_int_cmp.cpp -o t && ./t
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

#include "annoy_int_cmp.h"

static int failures = 0;

// Oracle: clamp computed in __int128 (lossless for all int64/uint64 values).
template <class T, class S>
static T ref_clamp(S v) {
  __int128 x  = static_cast<__int128>(v);
  __int128 lo = static_cast<__int128>(std::numeric_limits<T>::lowest());
  __int128 hi = static_cast<__int128>(std::numeric_limits<T>::max());
  if (x > hi) return std::numeric_limits<T>::max();
  if (x < lo) return std::numeric_limits<T>::lowest();
  return static_cast<T>(v);
}

template <class S>
static bool fits_in_S(__int128 c) {
  return c >= static_cast<__int128>(std::numeric_limits<S>::lowest()) &&
         c <= static_cast<__int128>(std::numeric_limits<S>::max());
}

static const __int128 CANDIDATES[] = {
    0, 1, 2, 100, 127, 128, 255, 256, 32767, 32768, 65535, 65536,
    2147483647LL, 2147483648LL, 4294967295LL,
    (__int128)9223372036854775807LL,                 // INT64_MAX
    (__int128)9223372036854775807LL + 1,             // INT64_MAX+1
    ((__int128)1 << 64) - 1,                          // UINT64_MAX
    -1, -100, -128, -129, -32768,
    (__int128)(-2147483647LL) - 1,                    // INT32_MIN
    (__int128)(-9223372036854775807LL) - 1,          // INT64_MIN
};

template <class T, class S>
static void test_pair(const char* tn, const char* sn) {
  for (__int128 c : CANDIDATES) {
    if (!fits_in_S<S>(c)) continue;
    S v = static_cast<S>(c);
    T got = annoy_clamp_cast_int<T, S>(v);
    T want = ref_clamp<T, S>(v);
    if (got != want) {
      std::printf("[FAIL] clamp<%s,%s>(%lld) got=%lld want=%lld\n",
                  tn, sn, (long long)v, (long long)got, (long long)want);
      ++failures;
    }
  }
}

// Run S over all integer types for a fixed T.
template <class T>
static void test_T(const char* tn) {
  test_pair<T, int8_t>(tn, "i8");    test_pair<T, uint8_t>(tn, "u8");
  test_pair<T, int16_t>(tn, "i16");  test_pair<T, uint16_t>(tn, "u16");
  test_pair<T, int32_t>(tn, "i32");  test_pair<T, uint32_t>(tn, "u32");
  test_pair<T, int64_t>(tn, "i64");  test_pair<T, uint64_t>(tn, "u64");
}

static void check(bool ok, const char* name) {
  std::printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", name);
  if (!ok) ++failures;
}

int main() {
  // full T x S clamp matrix
  test_T<int8_t>("i8");   test_T<uint8_t>("u8");
  test_T<int16_t>("i16"); test_T<uint16_t>("u16");
  test_T<int32_t>("i32"); test_T<uint32_t>("u32");
  test_T<int64_t>("i64"); test_T<uint64_t>("u64");
  check(failures == 0, "full T x S integer clamp matrix matches __int128 oracle");

  // spot-checks that the OLD bound-cast-into-S logic got wrong:
  //  widening int32->int64 must NOT clamp a valid positive value
  check(annoy_clamp_cast_int<int64_t, int32_t>(2147483647) == 2147483647LL,
        "widening i32->i64 keeps INT32_MAX (old code clamped)");
  //  int32 <- uint32 large value clamps to INT32_MAX (old lowest-bound cast wrapped)
  check(annoy_clamp_cast_int<int32_t, uint32_t>(4294967295u) == 2147483647,
        "u32->i32 clamps to INT32_MAX");
  //  uint8 <- signed negative clamps to 0
  check(annoy_clamp_cast_int<uint8_t, int>(-5) == 0, "i32->u8 negative -> 0");

  // sign-safe comparison spot-checks
  check(annoy_cmp_less(-1, 0u) == true,  "cmp_less(-1, 0u)");
  check(annoy_cmp_greater(std::numeric_limits<uint64_t>::max(), 5) == true,
        "cmp_greater(UINT64_MAX, 5)");
  check(annoy_cmp_less(5u, -1) == false, "cmp_less(5u, -1)");

  // bool handling
  check(annoy_clamp_cast_int<bool, int>(7) == true, "int->bool nonzero");
  check(annoy_clamp_cast_int<int, bool>(true) == 1, "bool->int true");

  std::printf("\n%d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
