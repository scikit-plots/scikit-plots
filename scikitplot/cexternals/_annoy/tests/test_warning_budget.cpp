// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// CY-020 warning-budget guard.
//
// The generated `annoylib.pyx.cpp` is compiled with ~35 `-Wno-*` suppressions to
// silence unavoidable Cython-generated noise. Because that generated code and our
// hand-written `annoylib.h` share one translation unit, those suppressions also
// hide *actionable* warnings in OUR C++ (e.g. ignored syscall results). This guard
// compiles the hand-written headers ALONE under strict warnings so the actionable
// categories stay visible and enforced, independent of the generated build.
//
// Intended CI invocation (the enforced, must-be-clean categories):
//   g++ -std=c++17 -O2 -Wall -Wextra -Werror=unused-result -Werror=return-type
//       -Werror=uninitialized -I<src> test_warning_budget.cpp -o /dev/null
//
// If that fails, an actionable warning was (re)introduced into the hand-written
// headers and must be fixed at the source (not suppressed).
//
// Documented BENIGN budget (reported by -Wextra/-Wconversion, intentionally NOT
// -Werror here):
//   * -Wunused-parameter  (~36): interface/stub params in template method
//     signatures kept for API symmetry.
//   * -Wconversion        (~10): bounded int->float (small counts) and
//     size_type->int (index sizes bounded by the index type); value-safe.
//   * -Wshadow            (~2):  INTENTIONAL — each build thread gets its own
//     `Random _random` shadowing the shared member (thread-local RNG seed).
// These are design-accepted; re-enabling -Werror for them would fight the design.

#include "annoylib.h"
#include "annoy_type_support.h"
#include "kissrandom.h"

using namespace Annoy;

// Instantiate the templates across representative element types so the compiler
// actually checks the generic bodies (uninstantiated templates are not fully
// diagnosed).
template <typename T>
static void exercise() {
  AnnoyIndex<int32_t, T, Euclidean, Kiss64Random,
             AnnoyIndexSingleThreadedBuildPolicy> idx(4);
  T v[4] = {};
  idx.add_item(0, v);
  idx.add_item(1, v);
  idx.build(3);
  T out[4];
  idx.get_item(0, out);
  (void)idx.get_n_items();
  (void)idx.get_n_trees();
}

int main() {
  exercise<float>();
  exercise<double>();
  exercise<long double>();  // float80 path
  (void)annoy_support::report_json();
  return 0;
}
