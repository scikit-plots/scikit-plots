// scikitplot/cexternals/_annoy/_mman/src/mman.h
// Cross-platform memory mapping support
//
// This header provides a unified interface for memory mapping across platforms:
// - Windows: Uses custom implementation wrapping CreateFileMapping/MapViewOfFile
// - Unix/Linux/macOS: Uses standard POSIX <sys/mman.h>
//
// Licensed under MIT
// Original Windows implementation from: https://code.google.com/p/mman-win32/

#pragma once
/*
 * sys/mman.h
 * mman-win32
 */
#include <sys/types.h>
#include <errno.h>
#include <stdint.h>

// ===========================================================================
// Platform Detection and Headers
// ===========================================================================

#if defined(_WIN32) || defined(_WIN64)

/* =============================== Windows =============================== */
#ifndef _MMAN_WIN32_H
#define _MMAN_WIN32_H

#ifndef _WIN32_WINNT		// Allow use of features specific to Windows XP or later.
#define _WIN32_WINNT 0x0501	// Change this to the appropriate value to target other versions of Windows.
#endif

/* All the headers include this file. */
#ifndef _MSC_VER
#include <_mingw.h>
#endif

#include <windows.h>
#include <io.h>

// POSIX-compatible constants for Windows
#define PROT_NONE       0
#define PROT_READ       1
#define PROT_WRITE      2
#define PROT_EXEC       4

#define MAP_FILE        0
#define MAP_SHARED      1
#define MAP_PRIVATE     2
#define MAP_TYPE        0xf
#define MAP_FIXED       0x10
#define MAP_ANONYMOUS   0x20
#define MAP_ANON        MAP_ANONYMOUS

#define MAP_FAILED      ((void *)-1)

// Flags for msync
#define MS_ASYNC        1
#define MS_SYNC         2
#define MS_INVALIDATE   4

#ifndef FILE_MAP_EXECUTE
#define FILE_MAP_EXECUTE    0x0020
#endif

// ===========================================================================
// Windows Helper Functions
// ===========================================================================
// void*   mmap(void *addr, size_t len, int prot, int flags, int fildes, off_t off);
// int     munmap(void *addr, size_t len);
// int     mprotect(void *addr, size_t len, int prot);
// int     msync(void *addr, size_t len, int flags);
// int     mlock(const void *addr, size_t len);
// int     munlock(const void *addr, size_t len);

static int __map_mman_error(const DWORD err, const int deferr)
{
    if (err == 0)
        return 0;

    // return err;  // implemented

    // Map Win32 error codes to POSIX errno values
    // Map Win32 GetLastError() values to POSIX errno codes.
    // For unknown errors, fall back to deferr (typically EPERM).
    switch (err) {
        case ERROR_INVALID_HANDLE:      return EBADF;
        case ERROR_TOO_MANY_OPEN_FILES: return EMFILE;
        case ERROR_FILE_NOT_FOUND:      return ENOENT;
        case ERROR_PATH_NOT_FOUND:      return ENOENT;
        case ERROR_ACCESS_DENIED:       return EACCES;
        case ERROR_SHARING_VIOLATION:   return EACCES;
        case ERROR_LOCK_VIOLATION:      return EACCES;
        case ERROR_ALREADY_EXISTS:      return EEXIST;
        case ERROR_FILE_EXISTS:         return EEXIST;
        case ERROR_NOT_ENOUGH_MEMORY:   return ENOMEM;
        case ERROR_OUTOFMEMORY:         return ENOMEM;
        case ERROR_INVALID_PARAMETER:   return EINVAL;
        case ERROR_INVALID_ADDRESS:     return EFAULT;
        case ERROR_DISK_FULL:           return ENOSPC;
        case ERROR_WRITE_PROTECT:       return EROFS;
        case ERROR_BROKEN_PIPE:         return EPIPE;
        case ERROR_NOT_SUPPORTED:       return ENOTSUP;
        case ERROR_CALL_NOT_IMPLEMENTED:return ENOSYS;
        default:                        return deferr;
    }
}

static DWORD __map_mmap_prot_page(const int prot)
{
    DWORD protect = 0;

    if (prot == PROT_NONE)
        return protect;

    if ((prot & PROT_EXEC) != 0)
    {
        protect = ((prot & PROT_WRITE) != 0) ?
                    PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    }
    else
    {
        protect = ((prot & PROT_WRITE) != 0) ?
                    PAGE_READWRITE : PAGE_READONLY;
    }

    return protect;
}

static DWORD __map_mmap_prot_file(const int prot)
{
    DWORD desiredAccess = 0;

    if (prot == PROT_NONE)
        return desiredAccess;

    if ((prot & PROT_READ) != 0)
        desiredAccess |= FILE_MAP_READ;
    if ((prot & PROT_WRITE) != 0)
        desiredAccess |= FILE_MAP_WRITE;
    if ((prot & PROT_EXEC) != 0)
        desiredAccess |= FILE_MAP_EXECUTE;

    return desiredAccess;
}

// ===========================================================================
// Windows Memory Mapping Functions
// ===========================================================================

inline void* mmap(void *addr, size_t len, int prot, int flags, int fildes, off_t off)
{
    HANDLE fm, h;

    void * map = MAP_FAILED;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4293)
#endif

    const DWORD dwFileOffsetLow = (sizeof(off_t) <= sizeof(DWORD)) ?
                    (DWORD)off : (DWORD)(off & 0xFFFFFFFFL);
    const DWORD dwFileOffsetHigh = (sizeof(off_t) <= sizeof(DWORD)) ?
                    (DWORD)0 : (DWORD)((off >> 32) & 0xFFFFFFFFL);
    const DWORD protect = __map_mmap_prot_page(prot);
    const DWORD desiredAccess = __map_mmap_prot_file(prot);

    const off_t maxSize = off + (off_t)len;

    const DWORD dwMaxSizeLow = (sizeof(off_t) <= sizeof(DWORD)) ?
                    (DWORD)maxSize : (DWORD)(maxSize & 0xFFFFFFFFL);
    const DWORD dwMaxSizeHigh = (sizeof(off_t) <= sizeof(DWORD)) ?
                    (DWORD)0 : (DWORD)((maxSize >> 32) & 0xFFFFFFFFL);

#ifdef _MSC_VER
#pragma warning(pop)
#endif

    errno = 0;

    if (len == 0
        /* Unsupported flag combinations */
        || (flags & MAP_FIXED) != 0
        /* Usupported protection combinations */
        || prot == PROT_EXEC)
    {
        errno = EINVAL;
        return MAP_FAILED;
    }

    h = ((flags & MAP_ANONYMOUS) == 0) ?
                    (HANDLE)_get_osfhandle(fildes) : INVALID_HANDLE_VALUE;

    if ((flags & MAP_ANONYMOUS) == 0 && h == INVALID_HANDLE_VALUE)
    {
        errno = EBADF;
        return MAP_FAILED;
    }

    fm = CreateFileMapping(h, NULL, protect, dwMaxSizeHigh, dwMaxSizeLow, NULL);

    if (fm == NULL)
    {
        errno = __map_mman_error(GetLastError(), EPERM);
        return MAP_FAILED;
    }

    map = MapViewOfFile(fm, desiredAccess, dwFileOffsetHigh, dwFileOffsetLow, len);

    CloseHandle(fm);

    if (map == NULL)
    {
        errno = __map_mman_error(GetLastError(), EPERM);
        return MAP_FAILED;
    }

    return map;
}

inline int munmap(void *addr, size_t len)
{
    if (UnmapViewOfFile(addr))
        return 0;

    errno =  __map_mman_error(GetLastError(), EPERM);

    return -1;
}

inline int mprotect(void *addr, size_t len, int prot)
{
    DWORD newProtect = __map_mmap_prot_page(prot);
    DWORD oldProtect = 0;

    if (VirtualProtect(addr, len, newProtect, &oldProtect))
        return 0;

    errno =  __map_mman_error(GetLastError(), EPERM);

    return -1;
}

inline int msync(void *addr, size_t len, int flags)
{
    if (FlushViewOfFile(addr, len))
        return 0;

    errno =  __map_mman_error(GetLastError(), EPERM);

    return -1;
}

inline int mlock(const void *addr, size_t len)
{
    if (VirtualLock((LPVOID)addr, len))
        return 0;

    errno =  __map_mman_error(GetLastError(), EPERM);

    return -1;
}

inline int munlock(const void *addr, size_t len)
{
    if (VirtualUnlock((LPVOID)addr, len))
        return 0;

    errno =  __map_mman_error(GetLastError(), EPERM);

    return -1;
}

#if !defined(__MINGW32__)
inline int ftruncate(const int fd, const int64_t size) {
    if (fd < 0) {
        errno = EBADF;
        return -1;
    }

    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    LARGE_INTEGER li_start, li_size;
    li_start.QuadPart = static_cast<int64_t>(0);
    li_size.QuadPart = size;

    if (SetFilePointerEx(h, li_start, NULL, FILE_CURRENT) == ~0 ||
        SetFilePointerEx(h, li_size, NULL, FILE_BEGIN) == ~0 ||
        !SetEndOfFile(h)) {
        unsigned long error = GetLastError();
        fprintf(stderr, "I/O error while truncating: %lu\n", error);
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
#endif

#endif  // _MMAN_WIN32_H

#else  // Not Windows

/* ============================== POSIX (Unix/Linux/macOS) =============================== */

// Use standard POSIX headers - all functions already defined
#include <sys/mman.h>
#include <unistd.h>

// POSIX already provides:
// - Functions: mmap, munmap, mprotect, msync, mlock, munlock
// - Constants: PROT_*, MAP_*, MS_*
// - Types: off_t, size_t

// Note: MAP_FAILED is defined as ((void *) -1) in POSIX
// Note: ftruncate is also standard in POSIX (in <unistd.h>)

#endif  // Platform detection
