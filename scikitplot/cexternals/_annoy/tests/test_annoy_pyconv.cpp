// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Regression test for ANNOY-CONV-001 (guide 6.11): AnnoyCountToPy must preserve
// the full unsigned range of count/size values, unlike the signed
// PyLong_FromLongLong it replaced. Embeds CPython and exercises the REAL header.
//
// Build & run (host):
//   g++ -std=c++17 $(python3-config --includes) -I<src> test_annoy_pyconv.cpp \
//       $(python3-config --ldflags --embed) -o t && ./t
#include <Python.h>
#include <cstdint>
#include <cstdio>
#include <climits>

#include "annoy_pyconv.h"

static int failures = 0;
static void check(bool ok, const char* name) {
    std::printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", name);
    if (!ok) ++failures;
}

// round-trip a count through AnnoyCountToPy and back via the unsigned reader
static bool roundtrip(unsigned long long v) {
    PyObject* o = AnnoyCountToPy(v);
    if (!o) { PyErr_Clear(); return false; }
    unsigned long long back = PyLong_AsUnsignedLongLong(o);
    bool err = (back == (unsigned long long)-1) && PyErr_Occurred();
    if (err) PyErr_Clear();
    Py_DECREF(o);
    return !err && back == v;
}

int main() {
    Py_Initialize();

    const unsigned long long LLMAX = (unsigned long long)LLONG_MAX;
    check(roundtrip(0ULL),            "0 round-trips");
    check(roundtrip(1ULL),            "1 round-trips");
    check(roundtrip(4096ULL),         "4096 round-trips");
    check(roundtrip(LLMAX),           "LLONG_MAX round-trips");
    check(roundtrip(LLMAX + 1ULL),    "LLONG_MAX+1 round-trips (was truncated)");
    check(roundtrip(UINT64_MAX),      "UINT64_MAX round-trips (was truncated)");

    // Document the defect: the old signed path misrepresents a >LLONG_MAX count.
    {
        PyObject* bad = PyLong_FromLongLong((long long)UINT64_MAX);  // becomes -1
        long long as_signed = PyLong_AsLongLong(bad);
        Py_XDECREF(bad);
        check(as_signed == -1,
              "old signed PyLong_FromLongLong(UINT64_MAX) yields -1 (the bug)");
    }

    std::printf("\n%d failures\n", failures);
    Py_Finalize();
    return failures == 0 ? 0 : 1;
}
