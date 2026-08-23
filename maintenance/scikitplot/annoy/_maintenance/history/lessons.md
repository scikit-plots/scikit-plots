# Lessons Learned — annoy subsystem

## Carried-over rules from `scikitplot.cython` MAINTAINING.md (apply as-is)
Read before touching the named area; these are verified, not aspirational.
- **L-SEC** strict must be operative (a master switch must govern its guards).
- **L-CACHE** cache key from the *resolved* toolchain (compiler + ABI + ft flag),
  never a PATH/sysconfig guess.
- **L-CON** exclusive means exclusive across *processes*; decouple staleness grace
  from wait timeout; prove with a real 2-process/thread schedule, not a mock.
- **L-BATCH** report partial results with a resume token; never fake rollback of
  irreversible side effects.
- **L-VALIDATE** validate external metadata against a schema; never silently coerce.
- **L-OBS** typed diagnostics + bounded log capture; preserve human message.
- **L-ABI** declare capability, then guard: error-by-default with opt-in for
  behavior that can't be universally guaranteed (unload/free-threaded/subinterp/fork).
- **L-PERF** remove redundant work; **defer** consistency-critical/architectural
  changes until a benchmark/invariant justifies a dedicated design.
- **L-DOC** operational docs must be executable (doctested) so claims can't drift.

### Direct mappings into this subsystem
- L-CON → §6.2 mmap/close race, §6.4 worker-lock deadlock, CY-009 same-object lock.
- L-ABI → CY-004 dealloc fault model, CY-009, §6.16 RNG continuation state.
- L-VALIDATE → §6.11 unsigned→signed, §6.13 unsafe compare, CY-008 unchecked reads.
- L-OBS → §6.10 noexcept swallow, CY-005 error-string leak, CY-017 exc declarations.
- L-PERF → §6.17/§6.18 file responsibility, CY-015 HTML boundary (defer splits).

## Active Rules (new, this subsystem)

### Rule L-HARNESS — build under the real package name
- **When:** building a Cython extension in-sandbox for the gate.
- **Then:** the package root on Cython `-I` and on `sys.path` must be a real
  directory named `scikitplot` (not a `work`/other symlink), so compiled classes
  get `__module__ == "scikitplot.…"`.
- **Verified by:** a pickle round-trip test passes (no
  `Can't pickle <class 'work.…'>`); `<Class>.__module__` starts with `scikitplot`.
- **Root cause:** Cython bakes the module qualname from the import path at build
  time; a mis-named root silently breaks pickle/`__reduce__` without touching logic.

### Rule L-GATE-SCOPE — scoped gate needs three bypasses (refined in Run 0b)
- **When:** running a Tier-A/B gate against a SCOPED build (not the full package).
- **Then:** (1) `pytest --noconftest -o addopts=` — the package-root `conftest.py`
  pulls the whole compiled package and `pyproject` addopts add uninstalled
  plugins; (2) pre-stub `sys.modules["scikitplot.api"]` so the lazy `__getattr__`
  short-circuits; (3) bind the real `get_config` onto `scikitplot` and stage the
  meson-generated `config/__config__.py`, `config/_citation.py`, `version.py`,
  because the annoy extension does `from scikitplot import get_config` at import.
- **Verified by:** target suite collects and runs; no `ImportError: cannot import
  name get_config` and no `scikitplot.api`/plotters chain error.

### Rule L-BUILD-ROOT — the package is not standalone-buildable
- **When:** building any extension in this subsystem in-sandbox.
- **Then:** assemble the repo root: `pyproject.toml`, root `meson.build`,
  `meson.options`, `meson_cpu/`, and `scikitplot/_build_utils/`. Scope compilation
  at the `ninja <target>` level (no per-subpackage meson option exists). Provide
  scaffolds `CITATION.bib` + `tools/f2py/generate_f2pymod.py` for configure.
- **Verified by:** `meson setup builddir -Dallow-noblas=true` reports build
  targets and `scikit-plots <version>`; the 4 scoped `.so` link.
- **Root cause:** submodule `meson.build` files reference parent-scope variables;
  the top `meson.build` has no `project()`.

### Rule L-NATIVE-SENTINEL — correct sentinels + no I/O in native error paths
- **When:** wrapping an OS/native API in a cross-platform shim (esp. Win32).
- **Then:** check the API's **documented** failure sentinel (e.g. a Win32 `BOOL`
  fails on `0`, not `~0`), validate resolved handles, map errors deterministically
  to `errno`/typed diagnostics, and never `fprintf(stderr, …)` from a library path.
- **Verified by:** a host unit test with mocked calls covering each failure sentinel
  and the success path (see `cexternals/_annoy/tests/test_mman_ftruncate.cpp`).
- **Root cause of ANNOY-MMAN-001:** wrong BOOL sentinel (`~0`) silently skipped the
  error path; direct stderr bypassed structured propagation.

### Rule L-UNSIGNED-FIDELITY — non-negative wide values use unsigned Python conversion
- **When:** returning a count/size/index held as an unsigned/wide type
  (`IndexDtype`/`uint64_t`/`size_t`) to Python.
- **Then:** use the centralized unsigned helper (`AnnoyIdxToPy` for indices,
  `AnnoyCountToPy` for counts) — never signed `PyLong_FromLongLong`, which
  truncates above `LLONG_MAX`. Values that legitimately carry a negative sentinel
  (e.g. `search_k == -1`) stay signed by design.
- **Verified by:** an embed-Python round-trip test at the boundaries `LLONG_MAX`,
  `LLONG_MAX+1`, `UINT64_MAX` (see `tests/test_annoy_pyconv.cpp`).

### Rule L-RANGE-SAFE-CAST — never cast a bound into the other operand's type
- **When:** range-checking an integer before narrowing (any signed/unsigned or
  width-mismatched pair).
- **Then:** compare in a common domain with sign-safe helpers
  (`annoy_cmp_*` / C++20 `std::cmp_*`); never `static_cast<S>(numeric_limits<T>::max())`
  — that wraps when T's bound is unrepresentable in S.
- **Verified by:** a full T×S property matrix vs an `__int128` oracle
  (`tests/test_annoy_int_cmp.cpp`).
- **Also:** when a fix lives in a header with a dead duplicate (e.g.
  `annoylib_review.h`), note the duplicate (CY-001) so the fix isn't shadowed.

### Rule L-HALF-OPEN-FLOAT — unit floats via power-of-two scale, never /(2**k - 1)
- **When:** converting an integer draw to a float in `[0, 1)`.
- **Then:** take the top mantissa bits and multiply by an exact power-of-two
  scale (`(draw >> 11) * 2**-53` for float64; `>> 40, * 2**-24` for float32).
  Never divide by `2**64 - 1` / `2**64` — rounding maps the max draw to `1.0`.
- **Verified by:** values lie on the `2**-53` grid and `max < 1.0` over large
  samples, incl. the NumPy `Generator` callback path.
- **Reproducibility:** changing the float stream is a sanctioned defect fix; do
  NOT change the integer stream (tree splitting) without a public-guarantee
  decision + legacy mode (see deferred §6.14).

### Rule L-WARN-ROOT — clear compiler warnings at the source, never silence
- **When:** a compiler warning fires (even a suspected false positive).
- **Then:** classify it (real vs. provability gap). If spurious, rewrite the
  construct into a form the compiler can prove safe (e.g. `vector::insert(it,it)`
  → `resize()+memcpy`); do NOT use a global `-Wno-*` or a blanket pragma that
  would also hide future real defects. Document the classification.
- **Verified by:** a warning-free rebuild AND a behavior-equivalence test
  (e.g. `tests/test_portable_blob.py` byte-for-byte pickle round-trip).

### Rule L-FD-SENTINEL — 0 is a valid descriptor; never use it as "absent"
- **When:** tracking an OS resource id (fd, handle, index) that has a valid 0.
- **Then:** use `-1` (or a typed/RAII wrapper) as the "not held" sentinel, and
  compare against it explicitly (`fd != -1`) — never truthiness (`if (fd)`),
  which misclassifies a valid 0 as absent and skips cleanup (leak).
- **Verified by:** a test that forces the resource onto id 0 (close fd 0 first)
  and asserts cleanup still runs (see `tests/test_fd_sentinel.py`, subprocess).

### Rule L-NO-SWALLOW — noexcept prevents escape, it must not erase failure
- **When:** a `noexcept` / type-erased boundary method catches to avoid escape.
- **Then:** never `catch (...) {}`. Provide a status/error channel, clear the
  outputs on failure (so no stale/partial data leaks), and preserve the first
  error (`dup_cstr(e.what())`). Match the module's existing error convention.
- **Verified by:** a test that the success path leaves error NULL and outputs
  correct/overwritten; fault injection for the throw path is a CI concern.

### Rule L-STD-TRUTH — enforce the real language standard, early and accurately
- **When:** a header declares a minimum C++ standard (`#error` / version guards).
- **Then:** enforce the standard the code actually needs and the build configures
  (here C++17: `if constexpr` + `cpp_std=c++17`). Place the guard BEFORE the first
  include/construct that needs it, so a wrong-standard build fails with the
  accurate message. Delete dead lower-standard fallback branches/comments.
- **Verified by:** a focused compile check — lower standards rejected with the
  accurate message, the target standard compiles.

### Rule L-FULL-RNG-STATE — serialize the whole generator, verify continuation
- **When:** adding or reviewing get_state/set_state/pickle for a stateful RNG.
- **Then:** persist ALL live state words (not just the seed) + version; make
  `__reduce__` carry `get_state()` (not `None`); keep a legacy fallback for
  seed-only states. The contract is RESUME, not restart (numpy-compatible).
- **Verified by:** the oracle — draw N, serialize, draw M from the original,
  restore, require the restored M to match (independent of the suite's own tests).
- **Also:** if existing tests encode restart, REWRITE them to the resume contract
  (don't delete); a test that serializes AFTER drawing and expects reproduction of
  those same values is testing the bug.

### Rule L-ATOMIC-SAVE — stage, fsync, atomic-rename; never mutate before commit
- **When:** persisting data by overwriting an existing file.
- **Then:** write to a same-filesystem temp, flush (`fflush`+`fsync` / `_commit`),
  then atomically replace (`rename` / `MoveFileEx REPLACE_EXISTING`). Never
  `unlink`/truncate the target first, and only switch in-memory backing after the
  rename commits. Remove the temp on any failure.
- **Verified by:** happy-path completeness, atomic replace over an existing file,
  no temp litter, and a clean failure that preserves target + in-memory state.
  Crash/ENOSPC injection is a CI concern.

### Rule L-NO-CORRUPT-EXPOSE — a failed finalize must not leave a loadable partial
- **When:** finalization of an in-place on-disk artifact can fail (truncate, flush).
- **Then:** on failure, remove (or mark invalid) the partial file so it is never
  accepted as valid. Unlinking an open file is POSIX-safe (fd/mmap persist).
  Prefer build-at-temp + atomic-rename for full crash safety when feasible.
- **Verified by:** the happy path keeps a complete loadable file; the failure
  branch (removal) is verified by construction + a CI fault-injection matrix.

### Rule L-ONE-EXISTENCE-VALIDATOR — validate ids once, before native access
- **When:** a wrapper exposes several operations that take an id/index into a
  native container.
- **Then:** route every such operation through ONE existence validator that
  rejects negative, over-capacity, not-constructed, and `>= size` ids before any
  native read. Do not open-code partial checks per method (they drift — here the
  `>= n_items` bound was missing from all but `__getitem__`).
- **Verified by:** each op raises for out-of-range/negative; valid ids and
  legitimate gaps/holes (`< size`) still work.

### Rule L-STRUCTURAL-IMMUTABLE — structural params can't change after construction
- **When:** a wrapper exposes sklearn-style `set_params` and some params fix the
  concrete native type/dispatch at construction (dtype, metric, dimension).
- **Then:** reject changes to those params once constructed (raise, matching the
  existing pattern) — never `setattr` metadata only, which diverges the report
  from the backing. Also reject unknown keys; keep `get_params()` keys valid so
  `set_params(**get_params())` round-trips. `clone()` uses `__init__`, so this is
  cloning-safe.
- **Verified by:** each structural param raises after construction; a rejected
  change leaves metadata unchanged; unknown keys raise; mutable params still work.

### Rule L-NATIVE-ERR-OWNER — own native error strings with one RAII wrapper
- **When:** a native call returns a heap-allocated (`strdup`/`malloc`) error
  string via a `char**` out-parameter that the caller must free.
- **Then:** bind it to a single RAII owner (here `ScopedError`, freeing in
  `__dealloc__`) at EVERY call site, so the free happens on all exits including
  the exception path. Do not hand-roll `free()` per site — one site
  (`on_disk_build`) forgot it and leaked on every failure.
- **Verified by:** failure raises the decoded message; repeated failures show no
  RSS growth (proxy) / no ASan leak (CI); success path unchanged.

### Rule L-CONTRACT-MATCHES-CORE — document the contract the core can back
- **When:** a Python protocol (sequence/mapping/estimator) wraps a lower-level
  core with different domain semantics (e.g. a sparse/extent core under dense
  sequence protocols).
- **Then:** make the documented contract match what the core actually
  guarantees. Do not promise semantics (occupancy, ordering, counts) the core
  cannot back — especially ones that cannot survive save/load. If richer
  semantics are wanted, they need explicit state + a persistence story (a
  separate decision), not a misleading docstring.
- **Verified by:** a contract test asserting the protocols are mutually
  consistent under the documented model (here: len == #iterations == valid `in`
  range == valid `__getitem__` range, on a sparse index).

### Rule L-RESTORE-DISPATCH-ENUMS — restore the dispatch type, not just its label
- **When:** an object selects a concrete implementation via an internal enum/id
  derived from a user-facing type string, and it has a state/pickle restore path.
- **Then:** the restore path must reconstruct the DISPATCH enum/id (the thing the
  factory actually branches on), not only the display string. Mirror exactly what
  `__init__` does. Restoring the label alone leaves the factory on its default
  and can silently build the wrong concrete type (here: int32 backing under an
  int64 blob).
- **Verified by:** round-trip EVERY supported structural combination and assert
  type metadata + data + query semantics are byte-for-byte preserved.

### Rule L-VERSION-SAFE-TAGS — integrate via the framework's own builder, not a fake super()
- **When:** a class exposes an optional-framework integration hook (e.g. sklearn
  `__sklearn_tags__`) but is not a subclass of that framework's base.
- **Then:** do not `super().<hook>()` — the parent chain has no such method.
  Delegate to the framework's ROOT builder unbound (e.g.
  `BaseEstimator.__sklearn_tags__(self)`) inside a lazy import, so the installed
  framework version produces its own correct defaults and the dependency stays
  optional. Alternatively remove the hook if the integration isn't claimed.
- **Verified by:** the hook returns the framework's expected type without raising,
  and the framework's public accessor (`get_tags`) works.

### Rule L-MIRROR-NOEXCEPT — Cython extern decls must mirror the C++ throw contract
- **When:** declaring `cdef extern` C++ methods in a `.pxd` whose C++ signatures
  carry `noexcept` and use a `bool`/`char** error` failure channel.
- **Then:** declare those methods `noexcept` in Cython (NOT `except +`). `except +`
  on a `noexcept` method is dead translation and advertises a second fault model
  that can never fire. Reserve `except +` for members that can genuinely throw —
  e.g. allocating constructors (`new` -> `std::bad_alloc` -> `MemoryError`).
- **Verified by:** rebuild is clean and every failing call still raises a Python
  exception via the error channel (proving the removed `except +` was dead), with
  success paths unchanged.

### Rule L-DECL-MIRRORS-CONTRACT — exception specs must match the real throw behaviour
- **When:** declaring an extern C/C++ function/method for Cython.
- **Then:** the exception spec must mirror the ACTUAL contract — `noexcept` for
  non-allocating, provably no-throw operations (mirroring a no-throw core);
  `except +` only where the callee can throw a C++ exception (e.g. allocating
  constructors -> bad_alloc). Do not leave it implicit or blanket-apply `except +`;
  a spurious `except +` asserts a boundary that doesn't exist, and a wrong
  `noexcept` on a throwing call terminates the process.
- **Verified by:** the module compiles clean and the wrapped path behaves
  correctly at runtime; a structural check that the decls match the chosen model.

### Rule L-PERSIST-CONTRACT-EXPLICIT — state what a format does and doesn't persist
- **When:** an object has a serialization path (native file save/load) that
  covers only part of its observable state.
- **Then:** document precisely what round-trips (and is validated on load) versus
  what does NOT (and therefore stays at the loading instance's config). Validate
  the structural metadata that IS stored; point users to the full-state path
  (pickle) for the rest. Keep the "Raises" section accurate to the real
  exception type.
- **Verified by:** load rejects structural mismatches; stored state mirrors
  exactly; non-persisted params demonstrably keep the loader's values; docstring
  names the correct exception.

### Rule L-VALIDATE-ADVISORY-PARAMS — validate even informational parameters
- **When:** a public constructor accepts a parameter that is advisory /
  informational (recorded and reported but not driving behaviour).
- **Then:** still validate it against its allowed set and raise on invalid input,
  the same as behavioural params. Reported metadata must be trustworthy; silently
  accepting garbage makes `get_params()` meaningless. Document the param as
  advisory so users don't expect it to change behaviour.
- **Verified by:** invalid values raise; valid ones round-trip in get_params();
  the advisory value provably does not alter results.

### Rule L-DOC-PARITY-TESTED — lock documented defaults/sets to real behaviour
- **When:** a docstring states a default value or an enumerated set of supported
  values, especially when a second (class-level) docstring documents the same
  thing.
- **Then:** make the two agree with each other AND with the code, and add a
  parity test that (1) asserts the documented default equals the real constructed
  default and (2) asserts every value the docstring calls "supported" actually
  works. Drop stale "future"/"only X" claims for things that already work.
- **Verified by:** the parity test fails if docs and behaviour diverge again.

### Rule L-PRECISION-CONTRACT-HONEST — document the precision the I/O path delivers
- **When:** a public dtype/type option nominally implies a precision wider than
  the actual data path (e.g. a float128 option over a double-precision bridge).
- **Then:** document, at the I/O boundary where conversion happens (not only in a
  class docstring), the real precision delivered — what is narrowed, what gains
  nothing, and what (if anything) the wide type actually improves. Don't let the
  nominal type width imply a guarantee the path can't keep.
- **Verified by:** a test that the wide-type I/O is identical to the bridge-width
  type's I/O, and that the contract is documented at the boundary.

### Rule L-NOFAIL-DESTRUCTOR — prove the deallocation path cannot throw
- **When:** a wrapper's destructor / __dealloc__ path releases a native resource
  (delete a C++ object, unmap, free).
- **Then:** make the no-fail guarantee explicit and enforced: declare the C++
  destructors `noexcept`, ensure every call they make is no-throw, and back it
  with a `static_assert(std::is_nothrow_destructible<T>)`. Keep the Cython dealloc
  helper's docstring accurate about WHY the GIL is needed (attribute access, not
  exception conversion) — a stale "except +" claim hides the real fault model.
- **Verified by:** the static_assert compiles; runtime del under partial init,
  double-release, and mmap-backed state never crashes.

### Rule L-ERROR-LISTS-ACCEPTED — invalid-value errors must reflect ALL accepted values
- **When:** raising an error for an invalid enum-like argument that accepts
  aliases/synonyms.
- **Then:** the message must list the canonical values AND make clear that
  documented aliases are accepted (with examples or a pointer to the full list),
  and all error sites for the same argument must agree. Don't imply a narrower
  accepted set than the code actually allows.
- **Verified by:** every accepted value/alias works; the error lists the
  canonical set and references aliases; all error sites share identical text.

### Rule L-NO-COMPILE-TIME-ISA-LANDMINE — CPU-feature fast paths must be runtime-chosen
- **When:** a header selects a hardware-accelerated implementation (F16C, AVX,
  AVX512) behind a compile-time macro (`#if defined(__F16C__)`), AND the build
  uses numpy-style `cpu-dispatch` that compiles high-ISA variants.
- **Then:** the accelerated code can be baked into dispatch variants and SIGILL on
  hosts (incl. virtualized) that report but don't honor the sub-ISA. Compile BOTH
  the portable and accelerated forms and choose at RUNTIME via a CPU probe
  (`__builtin_cpu_supports`), keeping the portable path proven-equivalent; and
  constrain `cpu-dispatch` for the extension until that lands.
- **Verified by:** the portable path matches the accelerated path bit-for-bit; a
  capability probe reports the active tier + real precision width.

### Rule L-RUNTIME-CPU-DISPATCH (implements L-NO-COMPILE-TIME-ISA-LANDMINE)
- **When:** a header offers a CPU-feature fast path (F16C/AVX/…) and the build
  uses cpu-dispatch that compiles high-ISA variants.
- **Then:** compile the accelerated function with `__attribute__((target("...")))`
  ALONGSIDE a portable fallback, and select at runtime via a cached
  `__builtin_cpu_supports(...)`. Never gate the fast path on a bare
  `#if defined(__FEATURE__)`, which bakes it into dispatch variants and SIGILLs on
  hosts that don't honor the sub-ISA.
- **Verified by:** header compiles with AND without the `-mFEATURE` flag; the
  fallback matches the accelerated path bit-for-bit; the previously-crashing suite
  passes.

### Rule L-HONEST-CAPABILITY-REPORT — never over-report a type's support
- **When:** exposing which numeric types/dtypes a build supports.
- **Then:** report the ACTUAL compiled tier + real precision width; an emulated
  type must self-identify (float128-as-long-double), an unavailable one must say so
  with zero width (never a silent alias), and `usable_as_dtype` must match what the
  constructor actually accepts (cross-check in tests).
- **Verified by:** report's usable set == set the Index accepts; unavailable tiers
  have size 0; the gate compiles with the optional-backend flag both present/absent.

### Rule L-DUAL-CONFIG-DATA-TYPES — pxd.in and pyx.in data_types must match
- **When:** adding/removing an element type in the generated dtype dispatch.
- **Then:** update BOTH `annoylib.pyx.in` AND `annoylib.pxd.in` `data_types`
  lists (the pyx generates typedef cimports the pxd must declare), AND every hard
  static_assert whitelist that lists T (`is_valid_data_type`, the HammingWrapper
  interface assert). A mismatch fails as "<Typedef>.pxd not found" or a
  static_assert on the new type.
- **Verified by:** the new dtype constructs across all metrics; supported_dtypes()
  usable set matches Index acceptance; ring green.

### Rule L-CAPABILITY-MATCHES-DISPATCH — report usability from the dispatch, not a proxy
- **When:** a capability report says whether a dtype is usable.
- **Then:** base `usable_as_dtype` on what the dispatch actually accepts (here the
  data_types matrix always includes float80, so it is always usable). Do not gate
  it on an unrelated proxy macro (e.g. distinctness from another type), which
  desyncs the report from reality on some platforms.
- **Verified by:** reported-usable set == set the constructor accepts, on every
  target platform.

### Rule L-NO-ORPHAN-INCLUDE — every .pxi must be consumed or removed
- **When:** a `.pxi` (or similar static include) is present and/or `fs.copyfile`'d
  in meson.
- **Then:** confirm a Cython `include "<file>.pxi"` directive actually pulls it in.
  An un-included `.pxi` is dead: its constants drift from the real code and mislead
  maintainers. Remove it from the source AND the build surface, or add + test a
  real inclusion path.
- **Verified by:** `tests/test_no_orphan_pxi.py` (no `.pxi` without a matching
  `include`); the module builds after removal.

### Rule L-NATIVE-ERR-OWNER — native char* errors go through ScopedError
- **When:** calling a native `bool fn(..., char** error)` bridge from Cython.
- **Then:** pass `&err.err` from a `ScopedError` (RAII) rather than a raw
  `char* error` freed by hand. Manual `free()` is bypassed by any early-return or
  exception on the path, leaking the strdup'd string. Applies to save/load/build/
  unbuild/serialize/add_item — migrate raw sites when touched.
- **Verified by:** no `free(error)` in the migrated region; repeated-failure test
  stays crash-free; error messages unchanged.

### Rule L-STRICT-COMPILE-OWN-CODE — check hand-written C++ under strict warnings separately
- **When:** a hand-written header is compiled inside a generated TU that carries
  blanket `-Wno-*` suppressions (e.g. Cython-generated `.pyx.cpp`).
- **Then:** also compile the hand-written headers ALONE (instantiating templates)
  under strict warnings with `-Werror` on the actionable categories (unused-result,
  return-type, uninitialized). Blanket suppression on the shared TU otherwise hides
  real defects like ignored syscall results. Note: `(void)fn()` does NOT silence
  GCC `warn_unused_result` — consume the value (`if (fn()!=0){}`).
- **Verified by:** `tests/test_warning_budget.cpp` compiles -Werror-clean on the
  actionable categories; documented benign budget for the rest.

### Rule L-PRIVATE-CIMPORT-SURFACE — auto-generated .pxd is not a stable API
- **When:** a generated `.pxd` exposing template-dispatch typedefs is installed.
- **Then:** state explicitly in the template header that the cimport surface is a
  PRIVATE implementation detail with no ABI-stability guarantee, point downstream
  users at the Python API, and scope its install (devel tag). Auto-generated
  dispatch names change whenever the type matrix changes, so they must not be
  presented as a stable public cimport API.
- **Verified by:** the generated/installed `.pxd` carries the policy statement
  (`test_pxd_abi_policy.py`).

### Rule L-SINGLE-CANONICAL-TEMPLATE — one built template, guarded
- **When:** a generated source has a `.in` template that the build selects.
- **Then:** ensure NO near-duplicate sibling template exists in the built package
  dir. A non-built twin lets a fix land where it never ships (and drifts stale).
  Delete duplicates; keep only quarantined copies clearly outside the build dir.
- **Verified by:** `test_single_canonical_template.py`.

### Rule L-GIL-RELEASE-NOT-THREAD-SAFE — "releases GIL" != "thread-safe"
- **When:** documenting a method that runs native work under `with nogil`.
- **Then:** do NOT call it "thread-safe" on that basis. Releasing the GIL lets
  OTHER threads run; it does not synchronize access to the same object. State the
  concurrency policy explicitly (which same-instance overlaps are safe vs UB) and
  only test the supported cases (independent instances; concurrent reads on a
  built, non-mutated index).
- **Verified by:** `test_concurrency_policy.py`; docstrings state the policy.
