#!/usr/bin/env bash
# Guard (macOS-Intel link fix): the float16 path must NOT reference compiler-rt
# CPU-feature globals (__cpu_features2 / __cpu_indicator_init). Those come from
# __builtin_cpu_supports and cause an illegal text relocation under the classic
# macOS linker (ld64) — breaking the Python-extension link on x86 macOS. has_f16c()
# uses a direct CPUID probe instead, which references no such symbol, so this must
# stay empty. Run in CI (any x86 host) to catch reintroduction of the builtin.
#
#   bash check_no_cpu_feature_reloc.sh   # exit 0 = clean, 1 = symbol reintroduced
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../src"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/probe.cpp" <<'CPP'
#include "annoylib.h"
#include "kissrandom.h"
using namespace Annoy;
// instantiate the float16 template whose add_item_w failed to link on macOS-Intel
template struct Annoy::AnnoyIndex<int32_t, float16_t, Euclidean, Kiss64Random,
                                  AnnoyIndexSingleThreadedBuildPolicy>;
int main(){ return (int)float16_t::has_f16c(); }
CPP
g++ -std=c++17 -O2 -I"$src" -c "$tmp/probe.cpp" -o "$tmp/probe.o"
if nm "$tmp/probe.o" | grep -iE 'cpu_features|cpu_indicator|cpu_model' ; then
  echo "FAIL: float16 path references a compiler-rt CPU-feature global (macOS-Intel link risk)." >&2
  echo "      Use a direct CPUID probe in has_f16c(), not __builtin_cpu_supports." >&2
  exit 1
fi
echo "OK: float16 path references no compiler-rt CPU-feature global (links under any linker)."
