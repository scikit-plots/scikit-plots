<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Maintaining `scikitplot.cython`

The single entry point for anyone changing this subsystem. It records **how** to
make a safe change, **why** the current design is the way it is, and **what** has
already been reviewed — so that hard-won knowledge lives with the code and is
verified against it rather than rotting in a scratch file.

This file is guarded by `tests/test__maintainer_docs.py`: the test fails if a
referenced document is missing or if a reviewed finding is dropped from the log.

## Document map

| Document | Audience | Purpose |
|---|---|---|
| `DEV_NOTES.md` | maintainers | Design goals, packaging, reuse-after-restart. |
| `OPERATIONS.md` | operators | Trust model, cache recovery, concurrency, batch, unsupported platforms. Snippets are doctested (`_operations_examples.py`). |
| `ADR-0001-runtime-lifecycle.md` | maintainers | Decision record: native-extension lifecycle (unload / free-threaded / subinterpreter / fork). |
| `MAINTAINING.md` (this file) | maintainers | Workflow, prevention rules, review log, ADR index. |

## Change workflow (reusable)

Every non-trivial change to this subsystem follows the same loop. It is
deliberately the same rhythm used for the R1–R30 hardening campaign:

1. **Reproduce / ground.** Reproduce the defect or write the failing test first.
   For a lifecycle/concurrency claim, prove it with a real process/thread
   schedule, not a single-process mock (see rule L-29).
2. **Root-cause, minimal fix.** Change the fewest lines in the fewest files.
   Prefer a localized fix over a broad refactor.
3. **Permanent regression test.** Encode the corrected contract. If a test
   codified the *old, broken* behavior, rewrite it to the correct contract —
   do not delete it.
4. **Always-green gate.** The full suite must pass before packaging:
   `PYTHONPATH=. python -m pytest -q scikitplot/cython/tests`.
5. **Public-surface parity.** New public symbol? Update `__init__.pyi` the same
   change. The AST guard (`tests/test__stub_parity.py`) enforces this.
6. **Document the "why."** A design decision → a new ADR. A trap that could
   recur → a prevention rule below.

## Prevention rules (lessons that must not recur)

Each rule is *specific, actionable, and verifiable*. Read these before touching
the named area.

- **L-SEC (strict must be operative).** A security master-switch (`strict`) must
  actually govern the guards it implies. Verify by asserting a permissive value
  flips the derived flags and an explicit per-flag value overrides the switch.
  *Area: `_security.py`.*
- **L-CACHE (key from the real toolchain).** Cache keys must fold the toolchain
  the build backend *actually* selects (resolved compiler + ABI + free-threaded
  flag), never a `PATH`/`sysconfig` guess. Verify by asserting a different
  resolved compiler yields a different fingerprint. *Area: `_cache.py`,
  `_profiles.py`.*
- **L-CON (exclusive means exclusive).** A build lock must exclude across
  *processes*. The staleness grace period must be **decoupled** from the wait
  timeout — a non-blocking probe (`timeout_s=0`) must never reclaim a live lock.
  Verify with a real two-process schedule (held → timeout, released → acquired).
  *Area: `_lock.py`. See `ADR`-adjacent test `test__interprocess_exclusivity.py`.*
- **L-BATCH (report, don't vanish).** A batch that commits irreversible side
  effects (imported modules) must return a structured partial result with a
  resume token on failure, not fail-fast-and-vanish. Never fake rollback of
  something that cannot be undone. *Area: `_public.py`, `_result.py`.*
- **L-VALIDATE (validate, don't coerce).** External metadata that drives builds
  must be validated strictly against a schema (reject unknown versions,
  wrong-typed entries, uncontained paths) — silent coercion hides malformed
  input. Keep a lenient default only as explicit opt-out. *Area:
  `_templates_api.py`.*
- **L-OBS (typed diagnostics, bounded logs).** Capture tool output in a *bounded*
  buffer and attach a typed diagnostic (phase, module, versions, status, log
  tail) to failures; preserve the human-readable message for compatibility.
  *Area: `_builder.py`, `_budget.py`.*
- **L-ABI (declare, then guard).** For behavior that cannot be universally
  guaranteed (unload, free-threaded, subinterpreter, fork), publish an explicit
  capability record and a guard that errors by default with an opt-in — never a
  false promise (`supports_unload` is `False`). *Area: `_profiles.py`,
  `ADR-0001`.*
- **L-PERF (remove redundant work; defer risky indexes).** Optimize by
  eliminating duplicated/unbounded work (dedupe normalized paths *after*
  validation; give traversal a budget). Defer a consistency-critical cache
  index until its benchmark justifies a dedicated design — a manifest must never
  become a second source of truth. *Area: `_public.py`, `_gc.py`.*
- **L-DOC (tested docs).** Operational docs must be executable: pair prose with
  doctested examples and a test that validates the Markdown, so claims cannot
  drift from the implementation. *Area: `OPERATIONS.md`,
  `_operations_examples.py`.*

## Review log — deep semantic review (R1–R30)

All 30 findings from the deep semantic review are closed. Each was a
self-contained change with a regression test and an always-green gate; all fixes
are cumulative in the shipped package.

| Finding | Area | Correction |
|---|---|---|
| CYTHON-CON-001 | lock | Exclusive mkdir + owner token. |
| CYTHON-CACHE-001 | cache | Staged build + atomic publish. |
| CYTHON-GC-001 | gc | GC under lock; skip held keys. |
| CYTHON-LOAD-002 | loader | Atomic artifact staging. |
| CYTHON-SEC-001 | security | Single validation choke point. |
| CYTHON-API-001 | public | Source `.pyx` via trusted include dirs. |
| CYTHON-CACHE-002 | cache | Basename containment + artifact hash. |
| CYTHON-PIN-001 | pins | Typed error + atomic pin writes. |
| CYTHON-WASM-001 | profiles | Platform capabilities + asset check. |
| CYTHON-LOAD-001 | loader | Import transaction (sys.modules). |
| CYTHON-CACHE-003 | cache | Toolchain fingerprint. |
| CYTHON-TPL-001 | templates | Path containment. |
| CYTHON-PKG-001 | builder | Dotted-name validation. |
| CYTHON-CON-002 | builder | setuptools / registry thread safety. |
| CYTHON-RES-001 | budget | Build budget + bounded buffer + timeout. |
| CYTHON-CACHE-004 | public | Transactional export. |
| CYTHON-COMP-001 | compiler | Compiler capabilities + spec version. |
| CYTHON-TYP-001 | stub | Stub parity + AST guard. |
| CYTHON-SEC-002 | security | `allow_*` default None; strict operative. |
| CYTHON-SCH-001 | cache | Meta schema version + reuse gate. |
| CYTHON-API-002 | api | Stability tiers. |
| CYTHON-API-003 | utils | ASCII-only sanitize + collision suffix. |
| CYTHON-BATCH-001 | public | Batch partial result + resume token. |
| CYTHON-DOC-001 | docs | Tested operational guide. |
| CYTHON-PORT-001 | profiles | Resolved compiler keys the cache. |
| CYTHON-TPL-002 | templates | Strict metadata validation. |
| CYTHON-OBS-001 | builder | Typed diagnostics + bounded capture. |
| CYTHON-ABI-001 | profiles | Runtime lifecycle contract + ADR-0001. |
| CYTHON-TEST-001 | lock | Decouple staleness; true 2-process test (fixed a real exclusivity bug). |
| CYTHON-PERF-001 | public/gc | Normalized-path dedup + bounded traversal. |

## Adding an ADR

Record any decision that is expensive to reverse or non-obvious. Copy the shape
of `ADR-0001-runtime-lifecycle.md`: **Status · Context · Decision · Consequences
· Alternatives considered**. Number sequentially (`ADR-0002-...`) and add a row
to the document map above.

## Deferred to repo / CI (intentionally not in the installed package)

The following are contributor-process or environment concerns and live at the
repository root, not in the wheel: the full review-campaign playbook; Windows CI
jobs (rather than skips); crash / `ENOSPC` fault-injection; a free-threaded /
subinterpreter subprocess matrix on special CPython builds; a live JupyterLite
test; `package_data` globs that ship `OPERATIONS.md` / `ADR-*` / `_templates/**`
in the wheel; a `mypy`/`pyright` type-check gate on the stub; and a future
indexed cache manifest with a 100k-entry benchmark (see rule L-PERF).
