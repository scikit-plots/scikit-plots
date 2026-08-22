<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 7 — ANNOY-OBS-001 (guide 6.10) — `_w` bridge swallowed exceptions

**Priority:** P1  **Area:** `cexternals/_annoy/src/annoylib.h`  **Gate tier:** ring

## Finding (grounded in guide 6.10)
Three widened `_w` bridge read methods — `get_item_w`, `get_nns_by_item_w`,
`get_nns_by_vector_w` — had empty `catch (...) {}` and `void` signatures with no
error channel. On failure (notably `std::bad_alloc`) a caller received empty,
partial, or stale outputs with no indication of failure. The block comment even
*claimed* all bridges route errors through the `char** error` convention — which
was true only for `add_item_w`, not these read methods.

## Root cause
"void + silent catch": the type-erased boundary erased failure information.

## Fix (contained; matches the established convention)
Per the review decision — explicit status/result channel, clear outputs, preserve
the error — and consistent with the sibling `add_item_w`:
- Added `char** error = NULL` to the three methods (base pure-virtual
  declarations + the single `final` bridge implementations).
- Replaced `catch (...) {}` with `catch (const std::exception& e)` /
  `catch (...)` that **clear the outputs** (zero the embedding; `clear()` the
  result/distances vectors) and set `*error` via `dup_cstr`. `noexcept` still
  prevents escape, but failure is no longer erased.
- Updated the block comment to match the now-true contract. Diff:
  `out/annoylib.h.run7.diff`.

Blast radius: the entire `_w` widened API has **0 internal callers** (latent
external interface), and the bridge is `final` with a single implementation, so
the signature change is contained; the default `= NULL` keeps it source-compatible.

## Verification
- **C++ bridge test** `tests/test_w_bridge_errors.cpp`: instantiates a concrete
  `AnnoyIndex<int32_t,float,Euclidean,…>`, builds a small index, and exercises
  all three `_w` methods — success leaves `error == NULL`, results are correct,
  outputs are **overwritten not appended** (clear semantics), and a `NULL` error
  argument is tolerated. 6 checks pass.
- **Ring green** (both `annoylib` extensions rebuilt): kiss 111, memmap 49,
  annoy/_annoy cython 27, portable-blob 5, fd-sentinel 1, euclidean 12. The
  success path is behavior-preserving (typed path unchanged; `_w` had no callers).

## Deferred to CI (honesty)
Forcing the exception path (`std::bad_alloc`) deterministically needs allocator
fault injection; the clear-outputs + error-report logic is implemented per the
guide and covered by inspection + the success/plumbing test. Fault injection is a
CI concern (consistent with prior deferrals). Related CY-017 (mirroring no-throw
declarations across the Cython layer) remains a separate future run.
