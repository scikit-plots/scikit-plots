# Active Tasks — annoy subsystem hardening

## Run 0 — baseline & playbook  [DONE]
- [x] Environment measured (Py 3.12.3, numpy 2.4.4, g++ 13.3, cython/meson/ninja).
- [x] Build reality established: meson unusable from package-only zip;
      direct `cython + g++` per-extension gate proven on `kiss_random`.
- [x] Baseline number: `random/_kiss` 99 pass / 6 fail (6 = harness naming
      artifact, cause identified; expected 105/105 under real `scikitplot` root).
- [x] Blast radius mapped: downstream contained; ring = annoy+cexternals+memmap+random.
- [x] Ordered playbook produced (TIER 0–3, R1–R35+, grounded in §6/§7/§39).
- [x] Tiered gate commands proposed.

### Results Review — Run 0
- Files added: `out/ANNOY_REVIEW_PLAYBOOK.md`, `out/RUN0_BASELINE.md`,
  `tasks/todo.md`, `tasks/lessons.md`.
- No source changed this run (baseline/plan only).
- Evidence: kiss_random built + `pytest --noconftest` → 99 passed, 6 failed
  (classified). Deviations: none.

## Run 0b — real scoped meson build + ring gate  [DONE]
- [x] Repo-root files integrated (pyproject, root meson, meson.options, meson_cpu, _build_utils).
- [x] `meson setup` + scoped `ninja` build of all 4 extensions — real pipeline.
- [x] Ring baseline (real build): **561 passed** (kiss 105, memmap 49,
      annoy/_annoy 333, annoy metric 42, annoy core 32). `random/_kiss` = 105/105
      (L-HARNESS confirmed; Run-0 pickle failures gone).
- [x] Documented scaffolds: CITATION.bib, tools/f2py stub, gate.py (L-GATE-SCOPE).
- [x] Surfaced BUILD-WARN-001 (annoymodule.cc:4051 -Wstringop-overflow) → TIER 1.
- [x] Deferred to CI: index/accuracy/memory_leak/multithreaded/on_disk/threading (sandbox OOM).

## Run 1 — ANNOY-MMAN-001 (guide 6.1)  [DONE]
- [x] Reproduced: `SetFilePointerEx == ~0` on a BOOL (failure=0) → error path skipped.
- [x] Root-cause fix: extracted `mman_ftruncate_win.h` (correct BOOL checks,
      handle validation, deterministic errno, no stderr); externed `ftruncate`
      signature preserved.
- [x] Permanent regression: 9-case mocked-Win32 host test — all pass.
- [x] Ring green after rebuild: memmap 49, kiss 105.
- [x] Centralized docs: `annoy/MAINTAINING.md` + `annoy/_maintenance/` hub.
- [x] Evidence: `RUN1_ANNOY_MMAN_001.md`, `mman.h.run1.diff`.

## Run 2 — ANNOY-CONV-001 (guide 6.11)  [DONE]
- [x] Reproduced: `n_items64`/`n_trees64` (uint64) via signed `PyLong_FromLongLong`.
- [x] Fix: added checked `AnnoyCountToPy` (`annoy_pyconv.h`), repointed both sites;
      audited all count/size/index conversions (sentinel-bearing ones left signed).
- [x] Permanent regression: embed-Python test, 7 checks incl. `UINT64_MAX` — pass.
- [x] Ring green after rebuild: kiss 105, memmap 49, annoy types/euclidean/cython.
- [x] Lesson L-UNSIGNED-FIDELITY; review-log ANNOY-CONV-001; evidence RUN2 doc.

## Run 3 — ANNOY-CMP-001 (guide 6.13)  [DONE]
- [x] Reproduced: safe_numeric_cast casts T bounds into S (wraps on signed/width mismatch).
- [x] Fix: annoy_int_cmp.h (sign-safe cmp + clamp, C++17 std::cmp_* backport);
      integer branch clamps without bound casts. Dead annoylib_review.h noted (CY-001).
- [x] Permanent regression: full T x S matrix vs __int128 oracle — pass.
- [x] Ring green (both annoylib rebuilt): kiss 105, memmap 49, annoy/_annoy 333.
- [x] Lesson L-RANGE-SAFE-CAST; review-log ANNOY-CMP-001; evidence RUN3 doc.

## Run 4 — ANNOY-RNG-001 (guide 6.15)  [DONE]
- [x] Reproduced: random()/next_double / (2**64-1) can return exactly 1.0.
- [x] Fix: canonical top-53-bits * 2**-53 (float32: 24 bits); integer stream untouched.
- [x] Permanent regression: test_kiss_random_unit_interval.py (6 tests) — pass.
- [x] Ring green: kiss 111; cross-ring memmap 49, annoy cython 27.
- [x] Lesson L-HALF-OPEN-FLOAT; review-log ANNOY-RNG-001; evidence RUN4 doc.
- [~] §6.14 KISS modulo bias DEFERRED (integer-stream change; needs public-guarantee decision).

## Run 5 — BUILD-WARN-001  [DONE]
- [x] Classified GCC 13 -Wstringop-overflow on vector::insert as spurious (safe by construction).
- [x] Fix: payload append via resize()+memcpy (root-cause form; no global -Wno-/pragma).
- [x] Permanent regression: test_portable_blob.py (pickle byte-equivalence, 4 metrics + determinism).
- [x] Warning-free rebuild (0 warnings); ring green (kiss 111, memmap 49, annoy suites).
- [x] Lesson L-WARN-ROOT; review-log BUILD-WARN-001; evidence RUN5 doc.

## Run 6 — ANNOY-FD-001 (guide 6.8)  [DONE]
- [x] Reproduced: _fd uses 0 as "not open" + truthiness checks; fd 0 valid -> leak.
- [x] Fix: sentinel -> -1 (init, 16 resets, 3 checks). Dead v0/review duplicates noted (CY-001).
- [x] Discriminating regression: test_fd_sentinel.py (subprocess forces fd 0, asserts unload closes it).
- [x] Ring green (both annoylib rebuilt): kiss 111, memmap 49, annoy cython 27, blob 5, euclidean 12.
- [x] Lesson L-FD-SENTINEL; review-log ANNOY-FD-001; evidence RUN6 doc.

## Run 7 — ANNOY-OBS-001 (guide 6.10)  [DONE]
- [x] Reproduced: _w bridge read methods catch(...) {} swallow -> empty/stale outputs.
- [x] Fix: added char** error channel (matches add_item_w); on failure clear outputs + set error.
- [x] Regression: test_w_bridge_errors.cpp (concrete-index bridge; success/plumbing/clear). 6 checks.
- [x] Ring green (both annoylib rebuilt): kiss 111, memmap 49, annoy cython 27, blob 5, fd 1, euclidean 12.
- [x] Lesson L-NO-SWALLOW; review-log ANNOY-OBS-001; evidence RUN7 doc.
- [~] bad_alloc fault-injection deferred to CI; CY-017 (Cython no-throw decls) separate run.

## Run 8 — ANNOY-STD-001 (guide 6.9)  [DONE]
- [x] Reproduced: #error declared C++11 min + dead C++11 mutex fallback; real min is C++17.
- [x] Fix: early accurate C++17 #error; removed dead fallback + stale comments.
- [x] Verified: focused compile check (c++11/c++14 rejected, c++17 compiles); ring green.
- [x] Lesson L-STD-TRUTH; review-log ANNOY-STD-001; evidence RUN8 doc.

## Run 9 — ANNOY-RNG-002 (guide 6.16)  [DONE]
- [x] Reproduced (oracle): all 5 wrappers restarted instead of resuming.
- [x] Fix: exposed x,y,z,c in pxd; full get_state/set_state/__reduce__; legacy fallback.
- [x] Rewrote 13 restart-contract tests -> continuation oracle; new test_kiss_state_continuation.py.
- [x] Full kiss suite 123 passed; cross-ring memmap 49, annoy cython 27.
- [x] Lesson L-FULL-RNG-STATE; review-log ANNOY-RNG-002; evidence RUN9 doc.

## Run 10 — ANNOY-SAVE-001 (guide 6.5)  [DONE]
- [x] Reproduced: save() unlinks target + writes in place + unload/reload -> non-atomic.
- [x] Fix: temp + fsync + atomic rename; unload/reload only after commit.
- [x] Regression: test_save_atomicity.py (complete/atomic-replace/clean-failure). Ring green.
- [x] Lesson L-ATOMIC-SAVE; review-log ANNOY-SAVE-001; evidence RUN10 doc.
- [~] ENOSPC/crash injection -> CI; §6.6 format-matrix decision + §6.7 on_disk truncate -> separate.

## Run 11 — ANNOY-SAVE-002 (guide 6.7)  [DONE]
- [x] Reproduced: failed finalize-truncate left a corrupt header-less file (the TODO).
- [x] Fix: added _on_disk_path; unlink the corrupt file on failure (open-file-safe).
- [x] Regression: test_on_disk_finalize.py (happy-path completeness). Ring green.
- [x] Lesson L-NO-CORRUPT-EXPOSE; review-log ANNOY-SAVE-002; evidence RUN11 doc.
- [~] truncate fault-injection -> CI; crash-during-build (temp+rename/header flag) -> follow-up w/ 6.6.

## Run 12 — CY-008 (guide Part II 31)  [DONE]
- [x] Reproduced: get_item/get_nns_by_item/get_distance returned garbage for id >= n_items.
- [x] Fix: one shared _check_item_in_range validator; all three route through it; holes preserved.
- [x] Regression: test_item_bounds.py (12). Ring green (holes_test 5 confirms gap handling).
- [x] Lesson L-ONE-EXISTENCE-VALIDATOR; review-log CY-008; evidence RUN12 doc.

## Run 13 — CY-006 (guide Part II 29.3)  [DONE]
- [x] Reproduced: set_params(index_dtype=..) mutated metadata while int32 backing stayed.
- [x] Fix: structural dtype params immutable-after-construction (like f/metric); reject unknown keys.
- [x] Regression: test_set_params_immutable.py (10). Ring green (annoy/_annoy 345).
- [x] Lesson L-STRUCTURAL-IMMUTABLE; review-log CY-006; evidence RUN13 doc.

## Run 14 — CY-005 (guide Part II 28)  [DONE]
- [x] Reproduced: public on_disk_build leaked native error string on every failure.
- [x] Fix: route on_disk_build through ScopedError RAII owner (auto-free on raise).
- [x] Regression: test_on_disk_error_ownership.py (3); RSS proxy 0 KB over 40k failures.
- [x] Lesson L-NATIVE-ERR-OWNER; review-log CY-005; evidence RUN14 doc.

## Run 15 — CY-007 (guide Part II 30)  [DONE]
- [x] Finding: dense-occupancy docstrings over sparse extent core (gap false-positive/iteration).
- [x] Decision: Model C (extent/array) — contract matched to self-consistent behavior; Model B deferred.
- [x] Regression: test_sparse_sequence_contract.py (5). Ring green (annoy/_annoy 363).
- [x] Lesson L-CONTRACT-MATCHES-CORE; review-log CY-007; evidence RUN15 doc.

## Run 16 — CY-010 (guide Part II 34)  [DONE]
- [x] Reproduced: pickle of index_dtype=int64/uint64 failed to deserialize (n_nodes < n_items).
- [x] Root cause: set_state restored dtype strings but not index_type_id/data_type_id enums.
- [x] Fix: derive dispatch enums from strings in set_state (mirrors __init__).
- [x] Regression: test_typed_state_roundtrip.py (12). Ring green (annoy/_annoy 363).
- [x] Lesson L-RESTORE-DISPATCH-ENUMS; review-log CY-010; evidence RUN16 doc.

## Run 17 — CY-011 (guide Part II 35)  [DONE]
- [x] Reproduced: __sklearn_tags__() raised AttributeError (super has no such method).
- [x] Fix: delegate to sklearn root builder BaseEstimator.__sklearn_tags__(self) (version-safe).
- [x] Regression: test_sklearn_tags.py (4). Ring green (annoy/_annoy 375).
- [x] Lesson L-VERSION-SAFE-TAGS; review-log CY-011; evidence RUN17 doc.

## Run 18 — CY-017 (guide Part II 27.2)  [DONE]
- [x] Finding: C++ interface all noexcept+char**error but .pxd declared except+ (dual fault model).
- [x] Fix: 39 interface-virtual decls -> noexcept; kept except+ on ctors (bad_alloc) + KISS.
- [x] Verified error propagation intact (except+ was dead); regression test_error_channel_contract.py (5).
- [x] Ring green (annoy/_annoy 384). Lesson L-MIRROR-NOEXCEPT; review-log CY-017; evidence RUN18.

## Run 19 — [PENDING USER "continue"]
Options:
- [ ] CY-013 wrapper/random dtype params informational/unchecked (honesty).
- [ ] CY-014 native state labeled JSON-portable too broadly (persistence docs).
- [ ] CY-018 save/load wrapper metadata may not mirror backing state.
- [ ] CY-019 docs/signatures/defaults/type-count drift (doc parity).
- [ ] KISS exception-decl follow-up (from R18); __getitem__ neg-index vs __contains__ (from R15).
- [ ] Staged: §6.2/§6.3 mmap state machine, CY-009 (thread-stress/ASan CI), §6.14, §6.6.

---

## Completed Tasks Archive
(none yet)

## Deferred (see _maintenance/DEFERRED_FUTURE_WORK.md)
- [ ] (A) Multiprecision backend (boost/MPFR) -> float256/512 usable dtypes.
- [ ] (B) Widened public I/O bridge (>float64 through add_item/get_item; ABI+format change, ties §6.6).
