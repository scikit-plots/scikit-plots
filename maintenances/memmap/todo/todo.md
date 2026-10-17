# `scikitplot.memmap` maintenance: active tasks

Round 33 work that changes `scikitplot.memmap`. The campaign's status and
open decisions are in [`../../corpus/todo/todo.md`](../../corpus/todo/todo.md).
Completed tasks are archived under [`../_maintenance/history/`](../_maintenance/history/).

Rules that apply to every increment: [`skills/_engineering/`](../../../skills/_engineering/SKILL.md).

## Task: round 33 / Run 6 / 6b - `MemoryMap` is safe without the GIL (R6-3)

### Context
- Review (`modules/14_memmap.md`, F-36): "On free-threaded CPython builds
  the export count is not atomic. Share one mapping between threads only
  with external synchronization." The class notes say the same.
- The meson build declares every Cython module
  `-Xfreethreading_compatible=True` when Cython is 3.1 or newer
  (`meson.build`), so on a free-threaded interpreter the module runs
  without the GIL. With an older Cython it is not declared, and importing
  it turns the GIL back on.
- Measured on 6c (`4eacea3`), the module built alone with Cython 3.3 for
  CPython 3.13.14 free-threaded (`r34/ft6b/build.sh`):
  - 8 threads x 20,000 `memoryview(m).release()`: after every view is
    released, `close()` still left the region mapped in 8 of 10 trials
    (lost updates of `_exports`); with the GIL on (`PYTHON_GIL=1`), 0 of 10;
  - the same, then a view held across `close()`: the process died with
    SIGSEGV in 1 of 5 runs of 10 trials (the count drifted down, `close()`
    unmapped under the live view).
- R6-2 (`spawn`) is done in 6a; the `random` generators already hold
  their `RLock` around every draw. This task is `memmap` only.
- Base: 6c (`4eacea3`).

### State and invariants
- Shared, mutable after creation: `_is_valid`, `_addr`, `_prot`,
  `_exports`, `_unmap_pending`. Fixed after creation: `_size`, `_flags`.
- I1: `_exports` equals the number of live exports and pins.
- I2: the region is unmapped only when `_is_valid` is false and
  `_exports` is 0, and only once.
- I3: no export or pin is taken on a closed mapping, or with a right the
  mapping does not have at that moment (`mprotect` cannot remove a right
  under an export, so a right checked with the pin taken stays).

### Design
- One lock per mapping, `mem_map_lock.h`: a `PyMutex` where the GIL does
  not exist (`Py_GIL_DISABLED`, CPython 3.13+), and nothing where it does
  (the GIL already serialises these sections: none releases it). So GIL
  builds compile to the same code, and the lock needs no Cython 3.1
  (`critical_section`), which the build's `cython>=3.0.8` does not promise.
- Under the lock, and only plain C inside (no Python call, no exception,
  no GIL release; errors are raised after the unlock):
  - `_acquire(need)`: check open and `need` rights, count the pin, all at
    once; the buffer export and every method that runs without the GIL
    use it, after their existing early checks (kept: messages and their
    order stay);
  - `_unpin()`: count down; at 0 with an unmap pending, take the address;
    the `munmap` runs after the unlock;
  - `close()`: mark closed, then defer or take the address;
  - `mprotect()`: check, system call and record together (as today with
    the GIL);
  - every read of `_is_valid`/`_addr` outside them (`addr`, `is_valid`,
    `__repr__`, the early checks): through the lock, so there is no data
    race to begin with.
- `__dealloc__` needs no lock: no other reference exists.
- The class notes: the external-synchronisation caveat goes; what holds
  is stated per operation.

### Steps
- [x] `mem_map_lock.h`; `_lock` field; `_acquire`, `_unpin`, `close`,
      `mprotect`, the reads
- [x] tests: the free-threaded probes as tests (run in a child process, on
      every build); the existing suite on the GIL build
- [x] free-threaded build of the module alone: probes 0 of N; the whole
      `test_mman.py` there
- [x] fragment; review; negative proof; suites; gates; harness

### Acceptance
- On CPython 3.13t (module built alone, GIL off): the lost-update probe
  leaves 0 regions mapped in 10 trials, and the held-view probe survives
  every run; `test_mman.py` passes there.
- On the GIL build: `test_mman.py` passes; the lock compiles to nothing
  and the behaviour (messages, their order, results) is unchanged.

### Results review - 6b (2026-10-17)
- GIL build (3.12, editable): `scikitplot/memmap` 65 passed, 1 skipped (the
  free-threaded-only check); the thread tests take about 1 s each.
- Free-threaded 3.13t, the module built alone (Cython 3.3, GIL off): the
  final thread tests on the base: at least one fails in every run (the
  count test 4 of 4, the held-view test 3 of 4, the mixed-operation test
  8 of 12, mostly SIGSEGV); on the change: 6 passed in 4 of 4 (and 4 of 4
  in the re-review); `test_mman.py` 60 passed. Built without
  `freethreading_compatible`, the GIL-off check fails (the tests would have
  run with the GIL).
- Review 1: no High or Medium; the three invariants hold, no section calls
  Python, raises, releases the GIL or takes a second lock. 11 Low, fixed:
  two race tests had no power on the base (a mixed-operation test added;
  the two are labelled smoke tests); `/proc/self/maps` matched entry starts
  only (merged entries: ranges now); the free-threaded check could only
  skip (now read from the build configuration, checked in a child, fails);
  the `mprotect` error and Raises named views only (running operations
  hold too); header and record wording; the `_addr` read rule written
  down; `close()`'s failure window documented; `fork` noted in the header;
  the first paragraph of the fragment ("waits for" -> "defers the unmap");
  `_pin` returns `int` with `except -1`; the child runs outside the source
  tree. Re-review: all fixed; 4 Low (the fragment's `mprotect` sentence,
  per-test rates in this record, the free-threaded check on Windows, the
  mixed test skipping its crash check without `/proc`), fixed.
- Header: g++ `-std=c++11/17 -Wall -Wextra -pedantic` against the 3.12 and
  3.13t headers, no warning (review). Not built here: MSVC, macOS, 3.14t.
- Gates against 6c: clean; `cython-lint` on `mem_map.pyx`: nothing (the
  base had 4).
- Suites (with `scikitplot/memmap` added): 10080 passed + the baseline 15.
- CI harness on an untouched snapshot (`snap6b`): 203 passed, 4 failed,
  3 skipped, the failing ids those of 6a. `memmap` is in no partial
  distribution, so the harness does not build it: the free-threaded runs
  above are the module built alone.
