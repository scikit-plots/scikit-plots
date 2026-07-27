// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Regression test for CY-004 (guide 28): the deallocation path must be proven
// no-fail. The concrete C++ destructors are declared `noexcept`, so no exception
// can be established while a native index is being destroyed (including from the
// Cython __dealloc__ slot). These static_asserts fail to compile if a destructor
// ever regresses to a potentially-throwing form.
//
// Build: g++ -std=c++17 -I../src test_dealloc_noexcept.cpp -o t && ./t

#include "annoylib.h"
#include "kissrandom.h"
#include <type_traits>
#include <cstdio>

using namespace Annoy;

using EuclF32 =
    AnnoyIndex<int32_t, float, Euclidean, Kiss64Random,
               AnnoyIndexSingleThreadedBuildPolicy>;
using AngF64 =
    AnnoyIndex<int64_t, double, Angular, Kiss64Random,
               AnnoyIndexSingleThreadedBuildPolicy>;

// AnnoyIndex is the primary owned type behind the Cython Index pointer. Its
// destructor must be noexcept so deallocation cannot establish an exception.
// (HammingWrapper's ~noexcept is enforced by the real module build.)
static_assert(std::is_nothrow_destructible<EuclF32>::value,
              "AnnoyIndex<int32,float,Euclidean> destructor must be noexcept");
static_assert(std::is_nothrow_destructible<AngF64>::value,
              "AnnoyIndex<int64,double,Angular> destructor must be noexcept");

int main() {
  // Exercise destruction through the base pointer (the ownership path used by
  // the Cython layer) under partial init (constructed, never built).
  AnnoyIndexInterfaceBase* p = new EuclF32(8);
  delete p;  // no-fail delete through the base pointer (Cython's ownership path)
  std::printf("0 failures\n");
  return 0;
}
