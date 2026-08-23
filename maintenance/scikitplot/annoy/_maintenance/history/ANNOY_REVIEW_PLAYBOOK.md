<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# `annoy` subsystem — deep semantic review playbook (ordered ledger)

Run-by-run hardening ledger for the `annoy` subsystem, mirroring the
`scikitplot.cython` R1–R30 campaign. This is a **review roadmap, not a patch
queue**: every item is expanded into the standard finding format (reproduce →
root-cause → regression test → always-green gate → parity → document) at the
start of its run.

**Source of truth:** `ANNOY_DEEP_SEMANTIC_REVIEW_GUIDE.md` (Part I §6/§7,
Part II §39). Finding IDs are quoted from the guide; nothing here is invented.

**Scope (confirmed):** `annoy/`, `cexternals/_annoy/`, `memmap/_memmap/`,
`random/_kiss/`. `neighbors/` + `nc/` come **last**, after the four are green.

**Restructuring policy (confirmed):** in-place readability only — granular
sections, logical grouping, visual separation *within* files. **No file
moves/splits** during hardening; structural moves are considered at the very end
(rule L-PERF / defer-risky).

**Run granularity:** one contract per run (several tightly-coupled findings on a
single file/contract), each independently green.

---

## Subsystem → guide-file map

| Submodule (in-tree) | Guide files | Domain |
|---|---|---|
| `cexternals/_annoy/src/` | §7.1 `annoylib.h`, §7.2 `annoymodule.cc`, §7.3 `mman.h`, §7.4 `kissrandom.h` | vendored C++ core + POSIX/Windows mmap adapter + RNG |
| `annoy/_annoy/` | §7.5 `mem_map.*`, §7.7 `.pyi`, §7.8 `__init__.py`, §7.9 `meson.build`; Part II | Cython `Index` bindings (`annoylib.pyx.in`/`.pxd.in`), stubs |
| `memmap/_memmap/` | §7.3 `mman.h`, §7.5 `mem_map.pyx/.pxd/.pyi` | memory-map lifecycle (MMAN) |
| `random/_kiss/` | §7.4 `kissrandom.h`, §7.6 `kiss_random.pyx/.pxd/.pxi/.pyi` | KISS RNG |

---

## Tier ordering rationale

Ordered by (1) priority P0→P3, (2) dependency: shared/foundational contracts
before dependents, (3) contract grouping. Carried prevention rules from
`MAINTAINING.md` are named per cluster.

### TIER 0 — safety boundary (P0). Gate: ring (A+B).
Establish memory-safety, concurrency, and lifecycle invariants first; everything
else builds on these. Matches guide §17 "P0 — establish safety boundaries".

| Order | ID | Finding (guide) | Submodule | Carried rule |
|---|---|---|---|---|
| R1 | §6.1 | Windows `ftruncate` failure test is incorrect — rewrite to correct contract, don't delete | memmap / tests | (test-contract) |
| R2 | §6.2 | mmap operations can race with explicit `close` | memmap | L-CON |
| R3 | §6.3 | exported NumPy view may outlive the open mapping | memmap / annoy | L-CON, ownership |
| R4 | §6.4 | worker exceptions and raw locks can terminate or deadlock | cexternals core | L-CON |
| R5 | CY-008 | out-of-range reads reach unchecked native access (P0/P1) | annoy Cython | L-VALIDATE |
| R6 | CY-009 | GIL release lacks a complete same-object lock policy (P0/P1) | annoy Cython | L-CON, L-ABI |

### TIER 1 — persistence & error propagation (P1). Gate: submodule (A), ring (B) if cross-boundary.

Persistence/atomicity contract:
| Order | ID | Finding | Submodule | Carried rule |
|---|---|---|---|---|
| R7 | §6.5 | `save()` is not failure-atomic | annoy / cexternals | L-PERF (atomic publish) |
| R8 | §6.6 | on-disk build and normal save do not share one format contract | annoy / cexternals | L-VALIDATE |
| R9 | §6.7 | failed final truncate can leave a corrupt on-disk index | memmap / cexternals | L-CON |
| R10 | CY-005 | native error string can leak in primary on-disk path | annoy Cython | L-OBS |
| R11 | CY-010 | non-default typed state restore is inconsistent | annoy Cython | L-VALIDATE |
| R12 | CY-014/018 | native state labeled portable too broadly; wrapper metadata may not mirror backing state | annoy Cython | L-DOC, L-VALIDATE |

Error-propagation contract:
| Order | ID | Finding | Submodule | Carried rule |
|---|---|---|---|---|
| R13 | §6.10 | widened `noexcept` bridge methods swallow exceptions | annoy Cython | L-OBS |
| R14 | CY-017 | Cython exception declarations do not mirror no-throw core | annoy Cython | L-OBS |
| R15 | §6.8 | file descriptor zero is used as "not open" | cexternals / memmap | L-VALIDATE |

Numeric / type-honesty contract:
| Order | ID | Finding | Submodule | Carried rule |
|---|---|---|---|---|
| R16 | §6.11 | unsigned Python values converted through signed constructors | annoy Cython | L-VALIDATE |
| R17 | §6.13 | integer conversion helper can make unsafe comparisons | cexternals core | L-VALIDATE |
| R18 | §6.15 | a floating random method may return exactly `1.0` | random / kiss | L-VALIDATE |
| R19 | §6.14 | KISS bounded indexing uses modulo reduction (bias) | random / kiss | L-VALIDATE |
| R20 | §6.16 | RNG "state" appears seed-only rather than full continuation state | random / kiss | L-VALIDATE, L-ABI |
| R21 | CY-012/013 | float128 public precision exceeds double bridge; wrapper/random dtype params unchecked | annoy Cython | L-DOC |

Estimator / protocol contract:
| Order | ID | Finding | Submodule | Carried rule |
|---|---|---|---|---|
| R22 | CY-006 | `set_params` can diverge metadata and concrete type | annoy Cython | L-VALIDATE |
| R23 | CY-007 | sparse native IDs conflict with dense sequence protocols | annoy Cython | L-VALIDATE |
| R24 | CY-011 | sklearn tag delegation fails | annoy Cython | (integration) |

Cross-platform (partial in-sandbox; real Windows → CI):
| Order | ID | Finding | Submodule | Carried rule |
|---|---|---|---|---|
| R25 | §6.12 | Windows mmap adapter is not a complete POSIX semantic adapter | memmap / mman.h | L-ABI (declare+guard) |
| R26 | §6.9 | declared C++11 minimum does not match the snapshot | cexternals / meson | L-CACHE (toolchain) |

### TIER 2 — architecture, reviewability, build (P2). Gate: submodule (A).
In-place readability only; structural splits **deferred** with documented invariant.
| Order | ID | Finding | Submodule | Carried rule |
|---|---|---|---|---|
| R27 | §6.17 | `annoymodule.cc` has excessive responsibility → in-place sectioning; split deferred | cexternals | L-PERF (defer) |
| R28 | §6.18 | `annoylib.h` combines unrelated core domains → in-place sectioning; split deferred | cexternals | L-PERF (defer) |
| R29 | CY-015 | rich HTML/CSS/link logic inflates compiled boundary | cexternals `_repr_html` | L-PERF (defer) |
| R30 | §6.19 | Meson compiles the binding into both a static library and the extension | annoy / meson | L-CACHE |
| R31 | §6.20 / CY-019 | exports & generated metadata can drift; docs/signatures/defaults drift | annoy stubs | L-DOC, stub-parity |
| R32 | CY-001/002/003 | multiple implementation templates hide fixes; `.pxi` source-of-truth; installed `.pxd` ABI | annoy Cython | L-CACHE, L-ABI |
| R33 | CY-016 | 160 specializations lack an explicit support-tier/test budget | annoy Cython | L-ABI (tiers) |
| R34 | CY-020 | generated compile-warning volume obscures actionable conversions | annoy / build | L-OBS |
| R35 | CY-004 | deallocation path fault model is ambiguous | annoy Cython | L-ABI, ADR |

### TIER 3 — optimization & innovation (P3). Gate: submodule (A) + benchmark.
Guide §17 P3. Deferred until a benchmark justifies a dedicated design (L-PERF).

---

## Deferred to CI / repo (named, not faked — honesty rule)
- Real Windows `mman.h` / `ftruncate` semantics (§6.1, §6.12) → Windows CI job.
- Full-package meson build (repo root `project()`, `_build_utils`, `pyproject`
  are **not** in the package-only zip) → CI with full checkout.
- Fault injection (ENOSPC, crash mid-truncate) for §6.5/§6.7 → CI fault matrix.
- Free-threaded / subinterpreter / fork lifecycle (CY-004, CY-009) → special
  CPython matrix + ADR (mirror `ADR-0001`).
- Fuzzing roadmap (guide §13, §40) → CI fuzz targets.

## Structural moves (deferred to end, per confirmed policy)
Candidate in-place → move later: split `annoylib.h` core domains (§6.18), extract
`_repr_html` assets out of the compiled boundary (CY-015), consolidate the
multiple `.pyx.in`/`backup_template` sources into one source-of-truth (CY-001).
Each requires a documented invariant + green ring before any move.
