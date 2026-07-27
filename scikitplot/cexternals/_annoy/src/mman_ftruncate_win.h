// Authors: The scikit-plots developers
// SPDX-License-Identifier: BSD-3-Clause
//
// Windows file-resize adapter, extracted from mman.h so its control flow and
// error mapping can be unit-tested with mocked Win32 calls on any host
// (review finding ANNOY-MMAN-001, guide 6.1).
//
// Contract
// --------
// Returns 0 on success. On failure returns -1 and sets `errno` via a
// deterministic GetLastError translation. Performs NO direct I/O: it never
// writes to stderr, so failures propagate through errno / the subsystem error
// path rather than being printed.
//
// Minimal Win32 surface required (provided by <windows.h>/<io.h> in the real
// build, or by mocks in tests): HANDLE, LARGE_INTEGER, INVALID_HANDLE_VALUE,
// ERROR_INVALID_HANDLE, FILE_CURRENT, FILE_BEGIN, SetFilePointerEx,
// SetEndOfFile, GetLastError, _get_osfhandle.
#ifndef ANNOY_MMAN_FTRUNCATE_WIN_H
#define ANNOY_MMAN_FTRUNCATE_WIN_H

#include <cerrno>
#include <cstdint>

// Resize the file behind an already-resolved handle to `size` bytes.
inline int annoy_win_set_file_size(HANDLE h, const int64_t size) {
    LARGE_INTEGER li_start, li_size;
    li_start.QuadPart = static_cast<int64_t>(0);
    li_size.QuadPart = size;

    // SetFilePointerEx returns a Win32 BOOL: nonzero is success and ZERO is
    // failure. The previous code compared against ~0, so a real failure (0)
    // never matched and the error path was skipped (silent corruption).
    if (SetFilePointerEx(h, li_start, NULL, FILE_CURRENT) == 0 ||
        SetFilePointerEx(h, li_size, NULL, FILE_BEGIN) == 0 ||
        SetEndOfFile(h) == 0) {
        const unsigned long error = GetLastError();
        switch (error) {
            case ERROR_INVALID_HANDLE:
                errno = EBADF;
                break;
            default:
                errno = EIO;
                break;
        }
        return -1;
    }
    return 0;
}

// POSIX-compatible ftruncate shim: validate the descriptor, resolve and
// validate the OS handle, then delegate to annoy_win_set_file_size.
inline int annoy_win_ftruncate(const int fd, const int64_t size) {
    if (fd < 0) {
        errno = EBADF;
        return -1;
    }
    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) {
        errno = EBADF;
        return -1;
    }
    return annoy_win_set_file_size(h, size);
}

#endif  // ANNOY_MMAN_FTRUNCATE_WIN_H
