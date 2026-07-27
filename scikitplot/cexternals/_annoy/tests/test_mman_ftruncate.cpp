// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Regression test for ANNOY-MMAN-001 (guide 6.1): the Windows file-resize
// adapter in mman_ftruncate_win.h. Runs on any host by providing a minimal
// mocked Win32 surface with fault injection, then compiling the REAL adapter
// header against it. Covers the seven cases the review requires plus the
// deterministic errno mapping.
//
// Build & run (host):
//   g++ -std=c++17 -I<src> test_mman_ftruncate.cpp -o t && ./t
#include <cstdint>
#include <cstdio>
#include <cerrno>

// ----------------------------- mocked Win32 surface -----------------------------
using HANDLE = void*;
using BOOL = int;
struct LARGE_INTEGER { int64_t QuadPart; };

static HANDLE INVALID_HANDLE_VALUE = reinterpret_cast<HANDLE>(-1);
enum { FILE_BEGIN = 0, FILE_CURRENT = 1 };
enum { ERROR_INVALID_HANDLE = 6, ERROR_IO_DEVICE = 1117 };

static int g_seek_calls = 0;               // running count of SetFilePointerEx calls
static int g_seek_fail_on = 0;             // 1 => fail 1st call, 2 => fail 2nd, 0 => none
static int g_setend_fail = 0;              // nonzero => SetEndOfFile fails
static unsigned long g_inject_error = ERROR_IO_DEVICE;  // error reported on failure
static unsigned long g_last_error = 0;
static HANDLE g_osfhandle = reinterpret_cast<HANDLE>(0x1);

static void reset_mocks() {
    g_seek_calls = 0; g_seek_fail_on = 0; g_setend_fail = 0;
    g_inject_error = ERROR_IO_DEVICE; g_last_error = 0;
    g_osfhandle = reinterpret_cast<HANDLE>(0x1); errno = 0;
}

BOOL SetFilePointerEx(HANDLE, LARGE_INTEGER, void*, int) {
    ++g_seek_calls;
    if (g_seek_fail_on != 0 && g_seek_calls == g_seek_fail_on) {
        g_last_error = g_inject_error;
        return 0;  // BOOL failure
    }
    return 1;      // BOOL success
}
BOOL SetEndOfFile(HANDLE) {
    if (g_setend_fail) { g_last_error = g_inject_error; return 0; }
    return 1;
}
unsigned long GetLastError() { return g_last_error; }
HANDLE _get_osfhandle(int) { return g_osfhandle; }

// ----------------------------- the real adapter under test ----------------------
#include "mman_ftruncate_win.h"

// ----------------------------- tiny test harness --------------------------------
static int failures = 0;
static void check(bool ok, const char* name) {
    std::printf("%s  %s\n", ok ? "[PASS]" : "[FAIL]", name);
    if (!ok) ++failures;
}

int main() {
    reset_mocks();
    check(annoy_win_ftruncate(3, 4096) == 0 && errno == 0, "success resize -> 0");

    reset_mocks();
    check(annoy_win_ftruncate(3, 0) == 0, "zero size -> 0");

    reset_mocks();
    check(annoy_win_ftruncate(3, int64_t(1) << 40) == 0, "large size -> 0");

    // core ~0 bug: a real failure (BOOL 0) must now hit the error path
    reset_mocks(); g_seek_fail_on = 1;
    check(annoy_win_ftruncate(3, 4096) == -1 && errno == EIO, "first seek fail -> -1/EIO");

    reset_mocks(); g_seek_fail_on = 2;
    check(annoy_win_ftruncate(3, 4096) == -1 && errno == EIO, "second seek fail -> -1/EIO");

    reset_mocks(); g_setend_fail = 1;
    check(annoy_win_ftruncate(3, 4096) == -1 && errno == EIO, "SetEndOfFile fail -> -1/EIO");

    reset_mocks();
    check(annoy_win_ftruncate(-1, 4096) == -1 && errno == EBADF, "fd<0 -> -1/EBADF");

    reset_mocks(); g_osfhandle = INVALID_HANDLE_VALUE;
    check(annoy_win_ftruncate(3, 4096) == -1 && errno == EBADF, "INVALID_HANDLE -> -1/EBADF");

    // deterministic mapping: ERROR_INVALID_HANDLE from a failing call -> EBADF
    reset_mocks(); g_seek_fail_on = 1; g_inject_error = ERROR_INVALID_HANDLE;
    check(annoy_win_ftruncate(3, 4096) == -1 && errno == EBADF, "ERROR_INVALID_HANDLE -> EBADF");

    std::printf("\n%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
