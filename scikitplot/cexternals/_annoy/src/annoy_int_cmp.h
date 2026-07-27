// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Sign-safe integer comparisons and a range-correct integer clamp
// (ANNOY-CMP-001, guide 6.13).
//
// The previous safe_numeric_cast bounds check compared `value` against
// `static_cast<S>(std::numeric_limits<T>::max())` — i.e. it cast the TARGET
// bound into the SOURCE type S first. When T's bound is not representable in S
// (signed<->unsigned, or T wider than S) that cast wraps, so the comparison is
// against a garbage bound and valid values are clamped or overflowing values
// pass. These helpers compare in a common domain WITHOUT casting either bound
// into the other operand's type (a C++17 backport of C++20 std::cmp_* / in_range).
#ifndef ANNOY_INT_CMP_H
#define ANNOY_INT_CMP_H

#include <limits>
#include <type_traits>

// --- sign-safe comparisons (integral, non-bool) ---
template <class A, class B>
constexpr bool annoy_cmp_less(A a, B b) noexcept {
  static_assert(std::is_integral_v<A> && std::is_integral_v<B>, "integral only");
  static_assert(!std::is_same_v<A, bool> && !std::is_same_v<B, bool>, "non-bool");
  using UA = std::make_unsigned_t<A>;
  using UB = std::make_unsigned_t<B>;
  if constexpr (std::is_signed_v<A> == std::is_signed_v<B>) {
    return a < b;
  } else if constexpr (std::is_signed_v<A>) {  // A signed, B unsigned
    return a < 0 ? true : UA(a) < b;
  } else {                                     // A unsigned, B signed
    return b < 0 ? false : a < UB(b);
  }
}
template <class A, class B>
constexpr bool annoy_cmp_greater(A a, B b) noexcept { return annoy_cmp_less(b, a); }

// --- range-correct integer clamp (no bound cast into S) ---
// Returns `value` clamped into T's representable range, then narrowed.
template <class T, class S>
constexpr T annoy_clamp_cast_int(S value) noexcept {
  static_assert(std::is_integral_v<T> && std::is_integral_v<S>, "integral only");
  if constexpr (std::is_same_v<T, bool>) {
    return value != S(0);
  } else if constexpr (std::is_same_v<S, bool>) {
    return static_cast<T>(value ? 1 : 0);
  } else {
    if (annoy_cmp_greater(value, std::numeric_limits<T>::max())) {
      return std::numeric_limits<T>::max();
    }
    if (annoy_cmp_less(value, std::numeric_limits<T>::lowest())) {
      return std::numeric_limits<T>::lowest();
    }
    return static_cast<T>(value);
  }
}

#endif  // ANNOY_INT_CMP_H
