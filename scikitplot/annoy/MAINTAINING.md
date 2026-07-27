<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Maintaining the `annoy` subsystem

Single entry point for anyone changing the `annoy` subsystem and its native
siblings. It records **how** to make a safe change, **why** the design is the way
it is, and **what** has been reviewed — so hard-won knowledge lives with the code.

The subsystem spans four in-tree areas (co-reviewed because the guide dissects
their C sources):

| Area | Role |
|---|---|
| `annoy/` | Cython `Index` bindings (`_annoy/annoylib.pyx.in`), user-facing mixins, base |
| `cexternals/_annoy/` | vendored C++ core (`src/annoylib.h`, `annoymodule.cc`, `kissrandom.h`, `mman.h`) |
| `memmap/_memmap/` | memory-map lifecycle (`mem_map.pyx`, uses `mman.h`) |
| `random/_kiss/` | KISS RNG (`kiss_random.pyx`, uses `kissrandom.h`) |

## Document map

| Document | Audience | Purpose |
|---|---|---|
| `MAINTAINING.md` (this file) | maintainers | Workflow, prevention rules, review log. |
| `_maintenance/ANNOY_REVIEW_PLAYBOOK.md` | maintainers | Ordered finding ledger (TIER 0–3). |
| `_maintenance/RUN0_BASELINE.md`, `RUN0b_MESON_GATE.md` | maintainers | Build/gate baseline + reproducible recipe. |
| `_maintenance/lessons.md`, `todo.md` | maintainers | Live lessons and run tracker. |
| `_maintenance/gate.py` | dev/CI | Scoped-gate runner (see L-GATE-SCOPE). |
| `_maintenance/CONTINUATION.md` | dev | **Resume the review in a fresh chat/sandbox** + the drop-in export procedure. |

`_maintenance/` is a **dev/review hub**; exclude it from the installed wheel
(like the cython campaign playbook lived at repo root, not in the wheel).

## Change workflow (reusable, per finding/contract)

1. **Reproduce / ground.** Read the finding in the guide; reproduce the defect or
   write the failing test first. For a lifecycle/concurrency claim, prove it with
   a real schedule, not a mock (carried rule L-CON).
2. **Root-cause, minimal fix.** Fewest lines/files; keep externed C signatures
   (e.g. `ftruncate(fd,size)`) stable so the Cython bindings don't break.
3. **Permanent regression test.** Encode the corrected contract. If a test
   codified old/broken behavior, rewrite it — don't delete it. Native/Windows
   logic is tested with mocked calls compiled on the host (see ANNOY-MMAN-001).
4. **Always-green tiered gate.** annoy → (memmap+random+cexternals) → import
   smoke; a ring failure blocks the run. Build via meson (`_maintenance/RUN0b`).
5. **Public-surface parity.** New public symbol → update the `.pyi` the same run.
6. **Document the "why."** Design decision → ADR. Recurring trap → prevention rule.
7. **Export the drop-in.** At end of run, regenerate the cumulative source-only
   `annoy_submodules_through_run<N>.zip` (procedure in
   `_maintenance/CONTINUATION.md` §5) and round-trip verify it.

If the working chat is lost, resume from `_maintenance/CONTINUATION.md` — it lists
what to re-upload, the standing prompt, and the build/gate recipe, so no chat
memory is needed.

## Prevention rules

Carried verbatim from `scikitplot.cython` (`_maintenance/lessons.md` has full
text + per-finding mappings): **L-SEC, L-CACHE, L-CON, L-BATCH, L-VALIDATE,
L-OBS, L-ABI, L-PERF, L-DOC**. New to this subsystem:

- **L-HARNESS** — build under a real `scikitplot`-named root so compiled classes
  get correct `__module__` (else pickle/`__reduce__` silently break).
- **L-GATE-SCOPE** — a scoped gate needs `pytest --noconftest -o addopts=`, a
  pre-stubbed `scikitplot.api`, the real `get_config` bound, and the generated
  `config/__config__.py` / `_citation.py` / `version.py` staged.
- **L-BUILD-ROOT** — the package is not standalone-buildable; assemble the repo
  root (`pyproject.toml`, root `meson.build`, `meson.options`, `meson_cpu/`,
  `_build_utils/`) and scope compilation at the `ninja <target>` level.

## Review log — deep semantic review

Findings closed here. Full roadmap: `_maintenance/ANNOY_REVIEW_PLAYBOOK.md`.

| Finding | Area | Priority | Correction |
|---|---|---:|---|
| ANNOY-MMAN-001 (guide 6.1) | `cexternals/_annoy/src/mman.h` | P0 | Windows `ftruncate` checked `SetFilePointerEx == ~0` (a BOOL whose failure value is 0), so real failures were skipped and the error was printed to stderr. Extracted a unit-testable adapter (`mman_ftruncate_win.h`): correct BOOL failure checks, resolved-handle validation, deterministic errno mapping, no direct I/O. Regression: 9-case mocked-Win32 host test (`tests/test_mman_ftruncate.cpp`). Ring green. |
| ANNOY-CONV-001 (guide 6.11) | `cexternals/_annoy/src/annoymodule.cc` | P1 | `n_items64`/`n_trees64` (`uint64_t` counts) were returned via signed `PyLong_FromLongLong`, misrepresenting values above `LLONG_MAX`. Added a checked count/size helper `AnnoyCountToPy` (`annoy_pyconv.h`) mirroring `AnnoyIdxToPy`, preserving full unsigned range; repointed both sites. Regression: embed-Python test (`tests/test_annoy_pyconv.cpp`, 7 checks incl. `UINT64_MAX`). Ring green. |
| ANNOY-CMP-001 (guide 6.13) | `cexternals/_annoy/src/annoylib.h` | P1 | `safe_numeric_cast` bounds-checked by casting T's bounds into the source type S, which wraps for signed↔unsigned / width-mismatched pairs (e.g. widening `int32→int64` wrongly clamped). Added `annoy_int_cmp.h` (sign-safe `annoy_cmp_*` + `annoy_clamp_cast_int`, a C++17 backport of `std::cmp_*`); integer branch now clamps without bound casts. Regression: full T×S matrix vs `__int128` oracle (`tests/test_annoy_int_cmp.cpp`). Dead `annoylib_review.h` duplicate noted for pruning (CY-001). Ring green. |
| ANNOY-RNG-001 (guide 6.15) | `random/_kiss/kiss_random.pyx` | P1 | `random()`/`next_double` divided a 64-bit draw by `2**64-1`, which rounds so a maximal draw returns exactly `1.0`, breaking the documented `[0,1)`. Switched to the canonical top-mantissa-bits × `2**-53` scale (float32: 24 bits × `2**-24`); integer stream untouched (index reproducibility preserved). Regression: `tests/test_kiss_random_unit_interval.py` (grid + interval + numpy callback + reference vectors). §6.14 modulo bias deferred (stream-changing; needs public-guarantee decision). Ring green. |
| BUILD-WARN-001 (build) | `cexternals/_annoy/src/annoymodule.cc` | P2 | GCC 13 emitted a spurious `-Wstringop-overflow` on `vector::insert(end(), it, it)` in `annoy_build_portable_blob` (safe by construction; false positive). Rewrote the payload append as equivalent `resize()+memcpy` — clears the warning at root without a global `-Wno-*` or pragma. Regression: `tests/test_portable_blob.py` (pickle round-trip byte-equivalence, 4 metrics + determinism). Warning-free rebuild; ring green. |
| ANNOY-FD-001 (guide 6.8) | `cexternals/_annoy/src/annoylib.h` | P1 | On-disk `_fd` used `0` as the "not open" sentinel and tested it by truthiness; fd `0` is valid (closed stdin), so an index on fd 0 was treated as closed and `unload()` leaked it. Sentinel changed to `-1` (init, 16 resets, 3 checks). Regression: `tests/test_fd_sentinel.py` (subprocess closes fd 0, asserts `unload()` closes it). Dead `annoylib_v0.h`/`annoylib_review.h` duplicates noted (CY-001). Ring green. |
| ANNOY-OBS-001 (guide 6.10) | `cexternals/_annoy/src/annoylib.h` | P1 | The `_w` bridge read methods (`get_item_w`, `get_nns_by_item_w`, `get_nns_by_vector_w`) had empty `catch (...) {}` and no error channel, returning empty/stale outputs silently. Added the `char** error` channel (matching `add_item_w`) and, on failure, clear outputs + set the error. Regression: `tests/test_w_bridge_errors.cpp` (concrete-index bridge test). 0 internal callers (contained). Ring green. |
| ANNOY-STD-001 (guide 6.9) | `cexternals/_annoy/src/annoylib.h` | P1 | Declared `#error` minimum was C++11 with a dead C++11 shared-mutex fallback, but the build is `cpp_std=c++17` and the code uses `if constexpr`. Set the true minimum to C++17 via an early accurate `#error`; removed the dead fallback and stale C++11 comments. Verified by a focused compile check (c++11/c++14 rejected, c++17 compiles). Ring green. |
| ANNOY-RNG-002 (guide 6.16) | `random/_kiss/kiss_random.{pyx,pxd}` | P1 | State/pickle saved only the seed, so restore RESTARTED the stream. Exposed the KISS `x,y,z,c` words in the pxd; `get_state`/`set_state`/`__reduce__` now save/restore full state (legacy seed-only still loads). 13 tests that encoded the old restart contract were rewritten to the continuation oracle; new `tests/test_kiss_state_continuation.py`. Full suite 123 passed. |
| ANNOY-SAVE-001 (guide 6.5) | `cexternals/_annoy/src/annoylib.h` | P1 | `save()` unlinked the target then wrote in place then unload+reload — a partial write lost the old file and a failed reload lost the in-memory index. Made it transactional: write to a same-dir temp, fsync, atomically rename, and only unload/reload after commit (target + in-memory preserved on failure). Regression: `tests/test_save_atomicity.py`. ENOSPC/crash injection -> CI. Ring green. |
| ANNOY-SAVE-002 (guide 6.7) | `cexternals/_annoy/src/annoylib.h` | P1 | A failed final truncate in `on_disk_build` finalization left a partial, header-less file exposed as loadable (the `TODO ... corrupt state`). Now removes the file on failure (added `_on_disk_path`; unlink is open-file-safe, no lifecycle change). Regression: `tests/test_on_disk_finalize.py` (happy-path completeness). Truncate-injection -> CI. Ring green. |
| CY-008 (guide Part II 31) | `annoy/_annoy/annoylib.pyx.in` | P0/P1 | `get_item`/`get_nns_by_item`/`get_distance` checked negative + dtype-capacity but not `>= n_items`, so out-of-range ids returned garbage from unchecked native reads. Added one shared `_check_item_in_range` validator (negative, capacity, not-constructed, `>= get_n_items_w()`); all three route through it. Holes (`< n_items`) still allowed. Regression: `tests/test_item_bounds.py`. Ring green. |
| CY-006 (guide Part II 29.3) | `annoy/_annoy/annoylib.pyx.in` | P1 | `set_params` let structural type params (`index_dtype`/`dtype`/`wrapper_dtype`/`random_dtype`) mutate reported metadata while the concrete int32 backing stayed — divergence; unknown keys were ignored. Made them immutable-after-construction (like f/metric) and reject unknown keys (`get_params` keys stay round-trip-safe). Regression: `tests/test_set_params_immutable.py`. Ring green. |
| CY-005 (guide Part II 28) | `annoy/_annoy/annoylib.pyx.in` | P1 | Public `on_disk_build` decoded a native `char*` error and raised without `free()`, leaking one alloc per failure (every other native-error site frees). Routed it through the existing `ScopedError` RAII owner so the free is automatic on every exit incl. the raise. Regression: `tests/test_on_disk_error_ownership.py`; RSS proxy 0 KB over 40k failures. Ring green. |
| CY-007 (guide Part II 30) | `annoy/_annoy/annoylib.pyx.in` | P1 | `__len__`/`__contains__`/`__iter__` docstrings promised dense-occupancy semantics over a sparse extent-based core (gap false-positive membership, gap iteration, len=extent). Adopted Model C (low-level extent/array): corrected the contract to match the self-consistent extent behavior (occupancy Model B deferred — needs format change, cannot survive native load). Regression: `tests/test_sparse_sequence_contract.py`. Ring green. |
| CY-010 (guide Part II 34) | `annoy/_annoy/annoylib.pyx.in` | P1 | Pickle of a non-default `index_dtype` (int64/uint64) failed to deserialize (`n_nodes < n_items`): `set_state` restored the dtype STRINGS but not the dispatch ENUMS (`index_type_id`/`data_type_id`) that `_ensure_index` uses, so it rebuilt an int32 backing under a non-int32 blob. Now derives the enums from the strings like `__init__`. Regression: `tests/test_typed_state_roundtrip.py`. Ring green. |
| CY-011 (guide Part II 35) | `annoy/_annoy/annoylib.pyx.in` | P1/P2 | `__sklearn_tags__` called `super().__sklearn_tags__()` but the class isn't a `BaseEstimator` subclass -> `AttributeError`. Now delegates to sklearn's root tag builder `BaseEstimator.__sklearn_tags__(self)` (version-safe, sklearn stays optional). Regression: `tests/test_sklearn_tags.py`. Ring green. |
| CY-017 (guide Part II 27.2) | `annoy/_annoy/annoylib.pxd.in` | P1/P2 | `Kiss64Random` operational methods declared `except +` while the RNG core is non-allocating no-throw arithmetic (inconsistent with the `noexcept` interface base, R7). Changed them to `noexcept` (allocating constructors keep `except +`), making the per-method choice explicit. Regression: `tests/test_rng_noexcept_decls.py`. Ring green. |
| CY-018 (guide Part II 39) | `annoy/_annoy/annoylib.pyx.in` | P2 | Native save/load validate f+metric but don't persist wrapper query params (`n_neighbors`/`seed`), which silently stay at the loader's config; docstrings omitted this and mis-declared the mismatch error as RuntimeError (actually IOError). Documented the persistence contract + corrected the exception type (no format change). Regression: `tests/test_save_load_contract.py`. CY-014 checked — no false JSON-portable label, no fix needed. Ring green. |
| CY-013 (guide Part II 39) | `annoy/_annoy/annoylib.pyx.in` | P2 | `wrapper_dtype`/`random_dtype` (advisory, not used for dispatch) accepted arbitrary strings silently while `index_dtype`/`dtype` were validated. Now validated against `{uint32,uint64}` at construction and documented as advisory. Regression: `tests/test_advisory_dtype_validation.py`. Ring green. |
| CY-019 (guide Part II 39) | `annoy/_annoy/annoylib.pyx.in` | P2 | `Index.__init__` docstring drift: `index_dtype` documented default `int64` (actual `int32`) and false "only int32,int64/future" + `dtype` false "only float32,float64/future float16,float128" — all 8 int and 4 float types work. Corrected both to match reality (+ class docstring). Regression: `tests/test_doc_parity.py` (parity-enforced). Ring green. |
| CY-012 (guide Part II 39) | `annoy/_annoy/annoylib.pyx.in` | P2 | `float128` dtype implies 128-bit precision, but all I/O flows through the `double` `_w` bridge — float128 I/O == float64 I/O (no extra input precision, only internal-arithmetic precision). Documented the double-bridge precision contract on `add_item` (class docstring already noted it). Regression: `tests/test_float128_precision_contract.py`. Ring green. |
| CY-004 (guide Part II 28) | `cexternals/_annoy/src/annoylib.h`, `annoy/_annoy/annoylib.pyx.in` | P1 | Dealloc fault model ambiguous: `_destroy_index` docstring falsely claimed `unload()` is `except +` (it's `noexcept` since R7), and concrete C++ destructors were only implicitly noexcept. Declared `~AnnoyIndex`/`~HammingWrapper` `noexcept` (compiler-enforced no-fail) + corrected the docstring. Regression: `tests/test_dealloc_noexcept.cpp` (static_assert) + `tests/test_dealloc_fault_model.py`. Ring green. |
| CY-017 (guide Part II 27.2) | `annoy/_annoy/annoylib.pxd.in` | P1/P2 | Every C++ interface virtual is `noexcept` + `char** error`, but the `.pxd` declared them `except +` (dead C++-exception translation, dual fault model). Converted 39 interface-virtual decls to `noexcept`; kept `except +` on allocating constructors and (non-noexcept) KISS methods. Single fault model = char** error -> wrapper raises. Regression: `tests/test_error_channel_contract.py`. Ring green. |

## Deferred (CI / repo, intentionally not rushed here)
Real Windows `mman.h` semantics; full-package meson build; fault injection
(ENOSPC / crash mid-truncate); large-index & concurrency suites that OOM in a
sandbox (`index_test`, `accuracy`, `memory_leak`, `multithreaded_build`,
`on_disk_build`, `threading`); the mmap lifetime state machine (guide 6.2/6.3)
and same-object GIL lock policy (CY-009) — architectural, staged with a
documented invariant per L-PERF rather than rushed in a hardening run.
