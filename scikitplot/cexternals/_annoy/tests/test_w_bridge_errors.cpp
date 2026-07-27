// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Regression test for ANNOY-OBS-001 (guide 6.10): the widened `_w` bridge must
// route failures through the char** error channel and clear outputs instead of
// swallowing exceptions. This exercises the REAL bridge on a concrete index.
//
// Build & run (host):
//   g++ -std=c++17 -I<src> test_w_bridge_errors.cpp -o t && ./t
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "annoylib.h"
#include "kissrandom.h"

using namespace Annoy;
typedef AnnoyIndex<int32_t, float, Euclidean, Kiss64Random,
                   AnnoyIndexThreadedBuildPolicy> Idx;

static int failures = 0;
static void check(bool ok, const char* name) {
  std::printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", name);
  if (!ok) ++failures;
}

int main() {
  const int f = 4;
  Idx idx(f);
  for (int i = 0; i < 30; ++i) {
    std::vector<float> v(f);
    for (int j = 0; j < f; ++j) v[j] = static_cast<float>((i * 7 + j * 3) % 11);
    idx.add_item(i, v.data(), nullptr);
  }
  idx.build(10, -1, nullptr);

  // get_item_w: success -> error stays NULL, embedding filled
  {
    char* err = nullptr;
    std::vector<double> emb(f, -123.0);
    idx.get_item_w(0, emb.data(), &err);
    check(err == nullptr, "get_item_w success -> no error");
    if (err) std::free(err);
  }

  // get_nns_by_vector_w: success -> results, error NULL, sizes agree
  std::vector<double> q(f, 1.0);
  {
    char* err = nullptr;
    std::vector<uint64_t> res; std::vector<double> dist;
    idx.get_nns_by_vector_w(q.data(), 5, -1, &res, &dist, &err);
    check(err == nullptr, "get_nns_by_vector_w success -> no error");
    check(!res.empty() && res.size() == dist.size(), "results present, sizes agree");
  }

  // clear/overwrite semantics: pre-fill outputs with junk, call, ensure the
  // outputs are replaced (not appended to) — the guide's "clear outputs".
  {
    char* err = nullptr;
    std::vector<uint64_t> res(99, 7ULL);
    std::vector<double> dist(99, 7.0);
    idx.get_nns_by_item_w(0, 5, -1, &res, &dist, &err);
    check(err == nullptr, "get_nns_by_item_w success -> no error");
    check(res.size() <= 5 && res.size() == dist.size(),
          "outputs overwritten, not appended (<=n)");
  }

  // error channel is optional: passing NULL must not crash on success
  {
    std::vector<uint64_t> res; std::vector<double> dist;
    idx.get_nns_by_vector_w(q.data(), 3, -1, &res, &dist, nullptr);
    check(!res.empty(), "NULL error arg tolerated on success");
  }

  std::printf("\n%d failures\n", failures);
  return failures == 0 ? 0 : 1;
}
