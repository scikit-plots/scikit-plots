<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# CY-009 — thread-safety claim aligned to enforced synchronization (Policy A)

**Priority:** P1  **Area:** `annoy/_annoy/annoylib.pyx.in` (docstrings + policy)  **Gate tier:** ring

## Finding
`build()` and the query methods claimed "**thread-safe**, releases GIL". This
conflated two different things: releasing the GIL (so *other* Python threads run)
vs. making concurrent operations on the *same* index safe. The native calls run
under `with nogil` and are NOT mutually excluded, so query-while-build,
add-while-query, and dealloc-during-a-native-call are real races on one instance —
the claim exceeded the enforced synchronization.

## Fix (Policy A — instance not thread-safe for mutation)
- `build()`: dropped "thread-safe"; documented that it MUTATES and must not overlap
  any other operation on the same instance; GIL release ≠ synchronization.
- query: qualified to "safe for concurrent reads" — safe from multiple threads
  only on a FULLY-BUILT, non-mutated index (read-only path); unsafe concurrent with
  mutation.
- Added an explicit **Concurrency (CY-009)** policy block to the class docstring:
  safe = independent instances + concurrent read-only queries on a built index;
  unsafe = any mutator overlapping any other same-instance operation. The wrapper
  adds no locking around native calls; callers serialize mutation themselves.

## Guard (supported cases only — does NOT assert races are safe)
`tests/test_concurrency_policy.py` (3): independent instances built+queried across
threads; many threads issuing read-only queries against one shared built index all
match the single-threaded baseline; and the class docstring states the Policy-A
contract. Kept small for determinism / no OOM.

## Verification
Guard 3 passed; docstring carries the Concurrency section and "not thread-safe";
`build()` no longer claims thread-safe.

## Completeness note (all misleading claims corrected)
Beyond `build()` and `get_nns_by_item`, two more sites carried the same "thread-safe"
overclaim and were corrected: `get_nns_by_vector` (qualified to concurrent-reads),
and — most importantly — the `__enter__` context manager, whose docstring said
"Acquires lock for thread-safe operations". That RLock is acquired ONLY in
`__enter__`/`__exit__`; the actual methods do not take it, so the context manager
is a *cooperative* mutual-exclusion aid (it serializes only threads that also use
`with index:`), not automatic thread-safety. Docstring corrected to say exactly
that — matching the guide's requirement that docs not imply the context manager
makes operations safe. Final grep: 0 "thread-safe, releases GIL" remain.
