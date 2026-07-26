<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Operational guide — `scikitplot.cython`

This guide documents the **operational semantics** the user guide's happy-path
examples do not: the trust model, cache recovery, concurrency, batch recovery,
and behavior on unsupported platforms (CYTHON-DOC-001).

Every code claim below has an executable counterpart in
`_operations_examples.py`, exercised as a doctest so this document cannot drift
from the implementation. Run:

```
python -m doctest scikitplot/cython/_operations_examples.py -v
```

---

## 1. Security / trust model

Runtime compilation executes a toolchain on caller-supplied sources and flags.
The `SecurityPolicy` is the trust boundary.

- The **default policy is strict**: dangerous inputs (shell metacharacters,
  reserved macros, dangerous compiler args, absolute include dirs) are rejected.
- `strict=` is a **master switch and is operative** (CYTHON-SEC-002): an unset
  `allow_*` flag follows `strict` (`True` → restrictive, `False` → permissive).
- An **explicit** `allow_*` value always overrides `strict`, so you can relax
  everything except one guard: `SecurityPolicy(strict=False,
  allow_shell_metacharacters=False)`.
- All build entry points route through one validation choke point
  (CYTHON-SEC-001); source size is capped by `max_source_bytes`.

Treat any non-default relaxation as granting the source author the ability to
run arbitrary build commands. Keep `strict=True` for untrusted input.

## 2. Cache integrity & recovery

The cache is content-addressed by a key that includes the full toolchain
fingerprint (Python/Cython/NumPy, compiler, ABI — CYTHON-CACHE-003 /
CYTHON-CACHE-001), so a toolchain change re-keys rather than reusing a stale
artifact.

- **Schema versioning** (CYTHON-SCH-001): `meta.json` carries
  `meta_schema_version`. Legacy (unversioned) or newer-unknown entries are
  **incompatible** and are rebuilt, never misread. Check with
  `is_meta_schema_compatible(meta)`.
- **Atomic publish** (CYTHON-CACHE-001/002): artifacts are staged and atomically
  swapped into place; a crash mid-build leaves the prior entry intact.
- **Transactional export** (CYTHON-CACHE-004): `export_cached` stages then
  atomically swaps, preserving the previous export if the copy fails.
- **Recovery protocol**: to recover a suspect cache, delete the offending entry
  directory (or run the GC); the next build re-creates and re-stamps it. Pinned
  entries are never GC'd (CYTHON-PIN-001), and GC refuses entries with an active
  build lock (CYTHON-GC-001).

## 3. Concurrency matrix

| Scenario | Behavior |
|---|---|
| Two processes, same cache key | Serialised by an advisory, owner-tokened file lock (`build_lock`, CYTHON-CON-001); the second waits up to `lock_timeout_s`. |
| Two threads, first-time setuptools init | Exactly-once via double-checked locking (CYTHON-CON-002). |
| Concurrent compiler-registry register/get | Guarded by an internal re-entrant lock (CYTHON-CON-002). |
| GC while a build holds a lock | GC skips the active key rather than deleting under it (CYTHON-GC-001). |
| Build exceeds a deadline | `build_timeout_s` raises `BuildTimeoutError` (caller-observed; CYTHON-RES-001). |

## 4. Batch builds & recovery

`cython_import_all_result(directory, collect=..., only=...)` returns a
`BatchBuildResult` (CYTHON-BATCH-001):

- `collect=False` (fail-fast): stops at the first failure and raises
  `BatchBuildError`, whose `.result` lists the **committed** items and whose
  `.resume_token` names the items still to attempt.
- `collect=True`: attempts every item and records all `failures`.
- Resume by passing the token back via `only=`.

Already-committed native side effects (imported modules) are **reported, not
rolled back**: imported modules cannot be safely unloaded, so the API gives you
the committed set and a resume path instead of pretending to undo them.

`cython_import_all()` remains the fail-fast, dict-returning convenience wrapper.

## 5. Unsupported-capability behavior

Not every environment can compile at runtime. Rather than fail opaquely, query
`platform_capabilities()` (CYTHON-WASM-001):

- `can_compile_at_runtime` — `False` where no toolchain is available (e.g.
  browser WASM / Pyodide).
- `is_browser_wasm`, `prebuilt_only`, `wasm_package_suffix` — describe the
  fallback path (ship prebuilt artifacts and import them via
  `import_artifact_bytes` / `import_artifact_path`).
- `verify_template_assets()` guards that packaged `.pxi`/`.pxd` assets are
  present before a template build is attempted.

Branch on these flags to keep behavior deterministic across desktop, CI,
Docker, notebook, and browser targets.

## 6. API stability

Every public symbol carries a stability tier (CYTHON-API-002): query
`api_stability(name)` and list a tier with `list_api(tier)`. Depend on
`Stability.STABLE`; treat `EXPERIMENTAL` as subject to change.

---

### Verification

The claims above are checked by `_operations_examples.py` (doctests) and by the
regression suites for each cited finding. The raw Markdown of this file is
validated by `tests/test__operations_docs.py`.
