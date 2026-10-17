/*
 * mem_map_lock.h - the lock of one MemoryMap, where the GIL does not exist.
 *
 * A free-threaded CPython (3.13+, Py_GIL_DISABLED) runs MemoryMap's methods
 * on several threads at once, so the export count and the open/closed state
 * change under a PyMutex. With the GIL these sections are already serial
 * (none of them releases the GIL), and the lock compiles to nothing: the
 * behaviour is the one the GIL build had.
 *
 * Rules for a section under the lock: plain C only. No Python call, no
 * exception, no GIL release, no second lock.
 *
 * Py_GIL_DISABLED comes from the interpreter's configuration (pyconfig.h,
 * or the build's defines on Windows), as for the rest of the extension's
 * ABI. A child forked while another thread holds the lock inherits it held,
 * as for any mutex; fork from a single thread, or exec in the child.
 */
#ifndef SKPLT_MEM_MAP_LOCK_H
#define SKPLT_MEM_MAP_LOCK_H

#include <Python.h>

#ifdef Py_GIL_DISABLED
typedef PyMutex skplt_mm_lock; /* zero-initialised by the object allocator */
#define skplt_mm_acquire(lock) PyMutex_Lock(lock)
#define skplt_mm_release(lock) PyMutex_Unlock(lock)
#else
typedef struct {
    char unused;
} skplt_mm_lock;
#define skplt_mm_acquire(lock) ((void)(lock))
#define skplt_mm_release(lock) ((void)(lock))
#endif

#endif /* SKPLT_MEM_MAP_LOCK_H */
