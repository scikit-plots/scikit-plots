# mman – Cross-Platform Memory Mapping (Cython)

**Status · v 1.0.1 · Production-ready ✅**

This single document covers everything: architecture, every fix that was
applied, the improvements that were implemented, the full public API,
step-by-step integration into your own project, runnable examples at three
difficulty levels, the roadmap for future work, and how to build and test.

---

## 1. Overview

`mman` is a Cython wrapper that exposes memory-mapping to Python through a
single, platform-agnostic API.

| Platform | What happens at the C level |
|---|---|
| **Windows** | `mman.h` translates to `CreateFileMapping` / `MapViewOfFile` |
| **Linux / macOS** | Standard POSIX `<sys/mman.h>` is used directly |

The Python programmer sees one class (`MemoryMap`) and one convenience
function (`mmap_region`) regardless of the OS.

---

## 2. Architecture – file map

```
mman.h                  ← C header (Windows emulation + POSIX passthrough)
  ↓  included by
mman.pxd                ← Cython declarations: extern funcs, inline helpers
  ↓  cimported by
mman.pyx                ← Python-facing wrapper: MemoryMap class + constants
  ↓  mirrors
mman.pyi                ← Type stubs for mypy / pyright / IDE autocomplete
  ↓  demonstrated by
plot_mman.py            ← Gallery example (Basic / Medium / Advanced)
```

Each layer has exactly one responsibility.  Nothing leaks across boundaries.

---

## 3. Fixes Applied

Every item here was a concrete defect found during review and fixed in
place.  Nothing was speculative.

### 3.1 Pointer-safety snapshot pattern (compilation error)

**Symptom** – Cython refused to compile:

```
Error: Storing unsafe C derivative of temporary Python reference
```

**Root cause** – pointer arithmetic was performed directly on
`self._addr`, a Python-object attribute.  Cython cannot guarantee the
object stays pinned during the arithmetic.

**Fix** – Every method that touches the raw pointer now *snapshots* it to a
local C variable first, then does all arithmetic on that variable:

```cython
cdef void* base_addr = self._addr          # snapshot (GIL held)
cdef uintptr_t addr  = <uintptr_t>base_addr
cdef char*    src    = <char*>(addr + <uintptr_t>offset)   # safe
```

Both `read()` and `write()` now use the identical two-step
`void* → uintptr_t → char*` derivation.  The inconsistency that existed
between them (read used `uintptr_t`, write did direct pointer addition)
has been removed.

### 3.2 `get_page_size()` – real system call

**Symptom** – the function returned a hard-coded `4096`.  macOS on Apple
Silicon uses 16 384-byte pages; any file-mapping offset that relied on
`4096` would silently misalign.

**Fix** – a `cdef extern` block now declares `sysconf` from `<unistd.h>`
and pulls the real `_SC_PAGESIZE` macro via a verbatim C snippet (with a
Windows dummy fallback).  `get_page_size()` calls `sysconf(_SC_PAGESIZE)`
and falls back to `4096` only when the call returns ≤ 0.

A new **`page_size`** read-only property on `MemoryMap` exposes this value
to Python so users can align offsets correctly without guessing.

### 3.3 Version mismatch

`mman.pxd` declared `VERSION_PATCH = 1`; `mman.pyx` had both
`DEF VERSION_PATCH = 0` and `__version__ = "1.0.0"`.  All three are now
`1` / `"1.0.1"`.

### 3.4 `FILE_MAP_EXECUTE` annotation

The constant was the only module-level value without a `Final[int]`
annotation.  It now matches every other constant.

### 3.5 Dead imports in `.pyi`

`Union` and `Literal` were imported but never used.  Removed.

### 3.6 Missing constants in `.pyi`

`MAP_FILE`, `MAP_TYPE`, and `FILE_MAP_EXECUTE` are exported by `.pyx` but
were absent from the stub file.  Added.

---

## 4. Improvements Implemented

These are features that were on the roadmap and have now landed.

### 4.1 `mlock()` / `munlock()` – page locking

`mman.h` and `mman.pxd` already declared these; the Python wrappers were
commented out.  They are now first-class methods on `MemoryMap`.

* `mlock()` pins pages in physical RAM (no page-faults / swapping).
* `munlock()` releases the lock.
* Both raise `MMapError` with a hint about `RLIMIT_MEMLOCK` /
  `SeLockmemoryPrivilege` when the kernel rejects the call.

### 4.2 `as_numpy_array()` – zero-copy NumPy view

Returns a 1-D `ndarray` that shares the mapped buffer with **no copy**.

* Accepts an optional `dtype`; defaults to `uint8`.
* `WRITEABLE` flag is set only when the mapping was opened with
  `PROT_WRITE`.
* A private `_mmap_owner` attribute on the array keeps the `MemoryMap`
  alive as long as the array exists, preventing use-after-free.
* Raises `ValueError` when `dtype.itemsize` does not evenly divide the
  mapping size.

### 4.3 `page_size` property

Exposes the kernel page size (see §3.2) as a read-only property so that
users can compute page-aligned offsets without hard-coding `4096`.

---

## 5. API Reference

### 5.1 Constants

| Group | Name | Value | Meaning |
|---|---|---|---|
| Protection | `PROT_NONE` | 0 | No access |
| | `PROT_READ` | 1 | Read |
| | `PROT_WRITE` | 2 | Write |
| | `PROT_EXEC` | 4 | Execute |
| Mapping | `MAP_FILE` | 0 | File-backed (default) |
| | `MAP_SHARED` | 1 | Shared across processes |
| | `MAP_PRIVATE` | 2 | Copy-on-write |
| | `MAP_ANONYMOUS` | 0x20 | Not backed by a file |
| | `MAP_ANON` | 0x20 | Alias for above |
| | `MAP_FIXED` | 0x10 | Require exact address |
| Sync | `MS_ASYNC` | 1 | Fire-and-forget flush |
| | `MS_SYNC` | 2 | Block until flushed |
| | `MS_INVALIDATE` | 4 | Discard cached pages |
| Windows | `FILE_MAP_EXECUTE` | 0x20 | Execute-map access bit |

### 5.2 Exceptions

```
OSError
 └── MMapError                  base for all mman errors
      ├── MMapAllocationError   mmap() / CreateFileMapping failed
      └── MMapInvalidParameterError   bad argument
```

### 5.3 `MemoryMap`

#### Factory methods (static)

| Method | Creates |
|---|---|
| `create_anonymous(size, prot, flags)` | RAM-only mapping |
| `create_file_mapping(fd, offset, size, prot, flags)` | File-backed mapping |

#### Properties (read-only)

| Property | Type | Description |
|---|---|---|
| `addr` | `int` | Virtual address |
| `size` | `int` | Byte count |
| `is_valid` | `bool` | `True` while open |
| `page_size` | `int` | OS page size |

#### Methods

| Method | Returns | Description |
|---|---|---|
| `read(size, offset=0)` | `bytes` | Copy bytes out |
| `write(data, offset=0)` | `int` | Copy bytes in; returns count |
| `mprotect(prot)` | `None` | Change protection flags |
| `msync(flags=MS_SYNC)` | `None` | Flush to backing file |
| `mlock()` | `None` | Pin pages in RAM |
| `munlock()` | `None` | Release pin |
| `as_numpy_array(dtype=None)` | `np.ndarray` | Zero-copy view |
| `close()` | `None` | Unmap (idempotent) |

Context-manager protocol (`with` statement) is supported; `close()` is
called automatically on block exit.

### 5.4 `mmap_region()`

Convenience function that dispatches to `create_anonymous` or
`create_file_mapping` depending on whether `MAP_ANONYMOUS` is set in
`flags`.

```python
mmap_region(size, prot=PROT_READ|PROT_WRITE,
            flags=MAP_PRIVATE|MAP_ANONYMOUS, fd=-1, offset=0)
```

---

## 6. How to Integrate Into Your Own Project

### 6.1 Prerequisites

* Python ≥ 3.7
* Cython ≥ 0.29 (3.0+ recommended)
* C++11 compiler (MSVC on Windows; g++ / clang++ on POSIX)
* NumPy (optional; required only for `as_numpy_array`)

### 6.2 Step-by-step

```
Step 1 – Locate the four source files in your tree:

    your_project/
    └── scikitplot/cexternals/_annoy/_mman/
        ├── src/
        │   └── mman.h          ← C header (do not modify)
        ├── mman.pxd            ← Cython declarations
        ├── mman.pyx            ← Cython implementation
        └── mman.pyi            ← Type stubs

Step 2 – Compile the extension:

    # Meson (preferred)
    meson setup builddir
    meson compile -C builddir

    # … or setuptools
    python setup.py build_ext --inplace

Step 3 – Import and use:

    from scikitplot.cexternals._annoy._mman import mman
    from scikitplot.cexternals._annoy._mman.mman import (
        MemoryMap, PROT_READ, PROT_WRITE,
    )

    with MemoryMap.create_anonymous(4096) as m:
        m.write(b"hello")
        print(m.read(5))        # b'hello'

Step 4 – Align file offsets using the page_size property:

    with MemoryMap.create_anonymous(4096) as m:
        page = m.page_size      # 4096 or 16384 depending on platform
        aligned_offset = (raw_offset // page) * page

Step 5 – Use NumPy (optional):

    import numpy as np
    with MemoryMap.create_anonymous(1 << 20) as m:
        arr = m.as_numpy_array(dtype=np.float64)
        arr[:] = np.linspace(0, 1, len(arr))
```

### 6.3 Common pitfalls

| Mistake | How to avoid |
|---|---|
| Using the NumPy array after `close()` | Keep the `with` block open while the array is in use |
| Passing an unaligned offset to `create_file_mapping` | Use `m.page_size` to compute the correct alignment |
| Calling `mlock()` without privileges | Wrap in `try/except MMapError`; log and continue |
| Forgetting `MAP_SHARED` for file writes | `MAP_PRIVATE` is copy-on-write; changes never reach disk |

---

## 7. Examples

The file `plot_mman.py` is a runnable gallery script.  The three sections
below mirror it exactly.

### 7.1 Basic – anonymous map

```python
from scikitplot.cexternals._annoy._mman.mman import (
    MemoryMap, PROT_READ, PROT_WRITE,
)

with MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE) as m:
    m.write(b"Hello from mman!")
    data = m.read(16)
    assert data == b"Hello from mman!"
    print(f"page_size = {m.page_size}")   # 4096 or 16384
```

### 7.2 Medium – file-backed map with visualisation

```python
import os, struct, tempfile
import numpy as np
import matplotlib.pyplot as plt
from scikitplot.cexternals._annoy._mman.mman import (
    MemoryMap, PROT_READ, PROT_WRITE, MAP_SHARED, MS_SYNC,
)

ROW  = struct.calcsize("<Id")   # 12 bytes: int32 + float64
ROWS = 16

# --- create and pre-allocate file ---
fd, path = tempfile.mkstemp()
os.write(fd, b"\\x00" * ROW * ROWS)
os.close(fd)

# --- map, write structured records, sync ---
with open(path, "r+b") as f:
    with MemoryMap.create_file_mapping(
        f.fileno(), 0, ROW * ROWS,
        PROT_READ | PROT_WRITE, MAP_SHARED
    ) as m:
        for i in range(ROWS):
            m.write(struct.pack("<Id", i, i * 1.5), offset=i * ROW)
        m.msync(MS_SYNC)

# --- verify via plain I/O and plot ---
with open(path, "rb") as f:
    raw = f.read()

byte_matrix = np.frombuffer(raw, dtype=np.uint8).reshape(ROWS, ROW)
plt.imshow(byte_matrix, cmap="viridis", aspect="auto")
plt.colorbar(label="byte value")
plt.title("File-backed mmap – raw byte layout")
plt.show()

os.unlink(path)
```

### 7.3 Advanced – zero-copy NumPy + mprotect + benchmark

```python
import time
import numpy as np
import matplotlib.pyplot as plt
from scikitplot.cexternals._annoy._mman.mman import (
    MemoryMap, PROT_READ, PROT_WRITE,
)

SIZE   = 1 << 20   # 1 MiB
ITERS  = 200

with MemoryMap.create_anonymous(SIZE, PROT_READ | PROT_WRITE) as m:
    arr = m.as_numpy_array(dtype=np.uint8)

    # fill via NumPy
    arr[:] = np.tile(np.arange(256, dtype=np.uint8), SIZE // 256)

    # --- mprotect lifecycle ---
    m.mprotect(PROT_READ)                  # read-only
    _ = m.read(64)                         # fine
    m.mprotect(PROT_READ | PROT_WRITE)     # restore

    # --- benchmark: .read() vs arr.copy() ---
    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = m.read(SIZE)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(ITERS):
        _ = arr.copy()
    t_np = time.perf_counter() - t0

# plot throughput
labels = [".read()", "arr.copy()"]
bw = [SIZE * ITERS / t / 1e9 for t in (t_read, t_np)]
plt.bar(labels, bw, color=["#4c72b0", "#dd8452"])
plt.ylabel("GB/s")
plt.title("Copy throughput – 1 MiB")
plt.show()
```

---

## 8. Future Enhancements

### Short-term (next release)

| # | Item | Notes |
|---|---|---|
| 1 | Named shared memory | `MemoryMap.create_named(name, size)` for cross-process IPC without explicit fd passing |
| 2 | Memory-mapped file iterator | Iterate over fixed-size records in a mapped file without loading the whole file |
| 3 | `memset()` helper | Efficiently zero or fill a mapped region in one C call |

### Long-term (roadmap)

| # | Item | Notes |
|---|---|---|
| 1 | Async / await support | `create_anonymous_async()` that yields back to the event loop during large allocations |
| 2 | Memory pool manager | Carve sub-regions out of a single large mapping; track free/used with a bitmap |
| 3 | Advanced IPC primitives | Futex-based locks inside shared mappings for low-latency producer/consumer |
| 4 | Cross-platform abstraction layer | Thin shim so that user code can switch between mman, Python `mmap`, and `multiprocessing.shared_memory` with one flag |

---

## 9. Build & Test

### Build

```bash
# Meson (recommended)
meson setup builddir
meson compile -C builddir
meson install -C builddir

# setuptools fallback
python setup.py build_ext --inplace
```

### Run the gallery example

```bash
python plot_mman.py
```

### Run the test suite

```bash
pytest test_mman.py -v                  # all tests
pytest test_mman.py -k "anonymous" -v   # subset
pytest test_mman.py --cov=mman --cov-report=html
```

### Quick smoke test (copy-paste into a REPL)

```python
from scikitplot.cexternals._annoy._mman.mman import MemoryMap, PROT_READ, PROT_WRITE

# anonymous read/write
with MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE) as m:
    m.write(b"smoke test", offset=100)
    assert m.read(10, offset=100) == b"smoke test"
    print("✅ anonymous OK")

# NumPy zero-copy
import numpy as np
with MemoryMap.create_anonymous(4096, PROT_READ | PROT_WRITE) as m:
    a = m.as_numpy_array(dtype=np.uint8)
    a[0] = 42
    assert m.read(1) == b"\x2a"
    print("✅ as_numpy_array OK")

print("✅ all smoke tests passed")
```

---

*Document version 1.0.1 · 2026-02-03*
