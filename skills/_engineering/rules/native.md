# Native code

Categories: C++, Native memory

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 19: C++ - a count is not stored in the id type without a bound
- **When:** a quantity derived from sizes (`bytes / sizeof(S)`) is cast to a narrow integer type.
- **Then:** clamp to `std::numeric_limits<S>::max()` first, and test at dimensions that put the quantity just below and above each limit.
- **Verified by:** `test_small_index_types_return_full_results_at_every_dimension`.
- **Added:** 2026-10-07. Root cause: `static_cast<int8_t>(136)` is -120; one comparison used it as `size_t`, the other signed.

### Rule 35: C++ - what is copied as bytes is initialised as bytes
- **When:** an object is built in uninitialised memory (`alloca`, `malloc`) and later copied whole (`memcpy`, `fwrite`).
- **Then:** zero the whole object first; padding is part of what is copied.
- **Verified by:** the saved file of one index is identical, byte for byte, from a wheel with threads and from one without (`test_one_thread_gives_the_same_file_however_it_was_asked_for`, and the cross-wheel measurement of round 5, `maintenances/_repo/_maintenance/history/r05_second_ci.md`).
- **Added:** 2026-10-07. Root cause: split nodes were assembled on the stack and copied into the index; four padding bytes per node carried stack contents into every saved file.

### Rule 80: Native memory - nothing is allocated by hand before a call that can raise
- **When:** Cython code allocates memory and then calls the Python API (`hash`, iteration, `__eq__`), or calls C++ that can throw.
- **Then:** hold the memory in `libcpp.vector`, whose destructor runs on every exit, or free it in `try/finally`. Declare every C++ function that can throw with `except +`, and check every remaining allocation for failure.
- **Verified by:** `test_a_raising_hash_does_not_leak`: 1,000 failing calls grow the glibc heap by less than 1 MB (`mallinfo2`). Before the fix the leak was 16 KB per call.
- **Added:** 2026-10-12. Root cause: `malloc`, then `hash()` raised, then return. Each failing call leaked `len(a) * 8` bytes, and `levenshtein.distance` reached this path with unhashable elements. `malloc` was never checked for `NULL`.

### Rule 86: Native memory - a view of native memory holds a buffer export, and closing defers the free
- **When:** an object owns native memory (`mmap`, `malloc`) and hands out views (NumPy arrays, `memoryview`), or releases the GIL while it uses that memory.
- **Then:**
  - Implement the buffer protocol with an export count.
  - `close()` with live exports ends the object's API and defers the free to the last `__releasebuffer__`.
  - Every method that releases the GIL around the memory holds the same count while it runs (pin before the first operation that can release the GIL, unpin in `finally`).
  - `close()` takes the address while it holds the GIL, so a second `close()` cannot free it twice.
  - Refuse to remove a right (a protection bit, a size) that a live export was handed out with: `BufferError`, as CPython does for a resize under exports.
  - Make NumPy views from `memoryview(self)`, because `np.frombuffer(obj)` releases its export at once.
- **Verified by:** `memmap/_memmap/tests/test_mman.py`.
  - Two subprocess reproductions exited with 139/-11 before the fix: a view returned from a `with` block, and `close()` during another thread's `read()`.
  - A `/proc/self/maps` check shows the region unmapped after the last view.
  - On the uploaded code 8 of the 9 lifetime tests fail; the ninth is a guard.
- **Added:** 2026-10-12.
- **Root cause:**
  - `as_numpy_array()` wrapped the raw address, and `close()` unmapped it underneath.
  - `read()` copied with the GIL released, so `close()` on another thread could unmap mid-copy.
  - `mprotect(PROT_READ)` left a writable array that faulted on its next write.
