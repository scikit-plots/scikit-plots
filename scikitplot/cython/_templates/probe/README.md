# `_templates/probe/` — developer & AI diagnostic probes

Reusable, standalone scripts that **reproduce** a defect or **verify** an
invariant of `scikitplot.cython`. They are intentionally shipped with the
package so both human maintainers and AI assistants can re-run them during the
run-by-run upgrade without rewriting them each time.

These are **not** templates and are excluded from `list_templates()` (the
`_templates_api` enumerator skips any path containing a `probe` part).

Each script is self-bootstrapping: it imports `scikitplot.cython` if already
installed/importable, otherwise walks up to the repo root that contains the
`scikitplot` package. Run any of them directly:

```bash
python -m scikitplot.cython._templates.probe.repro_con001      # if importable as a module
# or, as a plain script:
python path/to/scikitplot/cython/_templates/probe/repro_con001.py
```

Exit code `0` means the probe's success condition held.

| Script | Finding | What it checks | Pass condition |
|---|---|---|---|
| `repro_con001.py` | CYTHON-CON-001 | Two processes entering `build_lock` on the same path | Exit 1 = **exclusive** (no overlap). Exit 0 = bug reproduced (overlap) |
| `compile_probe.py` | CYTHON-CACHE-001 | Real single-module Cython build + cache reuse via staging/publish | `add(2,3)=5`, `used_cache` flips `False→True` |
| `probe_cache001_concurrency.py` | CYTHON-CACHE-001 | Two processes building the **same key** concurrently | Exactly 1 published entry, `used_cache=[False,True]`, 0 staging leftovers |
| `probe_gc001_race.py` | CYTHON-GC-001 | GC running while a build holds the per-key lock | Entry survives; reported in `skipped_active`, not `deleted` |
| `probe_load002_symlink.py` | CYTHON-LOAD-002 | Raw native-byte staging: symlink refusal, private 0700 dir, atomic publish | All properties hold (exit 0) |
| `repro_api001.py` | CYTHON-API-001 | Importing a normal absolute `.pyx` under the default policy | Prints "imported OK" on a fixed tree; `SecurityError` on the unpatched tree |
| `probe_cache002_containment.py` | CYTHON-CACHE-002 | Cache artifact selection is contained to the entry + integrity-checked | Absolute/traversal refused; mismatched hash raises (exit 0) |
| `repro_pin001.py` | CYTHON-PIN-001 | Corrupt `pins.json` handling | Raises `PinRegistryError`; `pin()` refuses to clobber |
| `probe_wasm001_capabilities.py` | CYTHON-WASM-001 | Platform capability contract + template .pxi/support asset guard | Contract coherent; browser=prebuilt-only; no missing assets; guard detects drops (exit 0) |
| `repro_load001.py` | CYTHON-LOAD-001 | Failed reload vs prior `sys.modules` entry | Prints "PRESERVED" on a fixed tree; "DESTROYED" on the unpatched tree |
| `probe_cache003_fingerprint.py` | CYTHON-CACHE-003 | Cache fingerprint includes toolchain/ABI; compiler change re-keys | Fingerprint complete; different compiler → different key (exit 0) |
| `repro_tpl001.py` | CYTHON-TPL-001 | Template/workflow/example resolver path containment | On a fixed tree, absolute/traversal names raise `ValueError` |
| `repro_pkg001.py` | CYTHON-PKG-001 | Package/module name validation | Malformed dotted names raise `ValueError` before build |
| `probe_con002_threadsafety.py` | CYTHON-CON-002 | Thread-safety of setuptools singleton + compiler registry | Exactly-once init; concurrent registry ops clean (exit 0) |
| `probe_res001_budget.py` | CYTHON-RES-001 | Build deadline + bounded output (in-process) | Deadline raises; results propagate; output keeps tail (exit 0) |
| `repro_cache004.py` | CYTHON-CACHE-004 | Transactional cache export | Interrupted export preserves the prior export (no partial dest) |
| `probe_comp001_capabilities.py` | CYTHON-COMP-001 | Versioned compiler capability descriptor | Built-ins declare caps; legacy default conservative (exit 0) |
| `probe_api002_stability.py` | CYTHON-API-002 | API stability tiers cover the whole public surface | Total coverage; tiers partition; accessors behave (exit 0) |
| `probe_typ001_stub_parity.py` | CYTHON-TYP-001 | Runtime `__all__` vs packaged `.pyi` parity | Every public symbol declared in the stub (exit 0) |
| `probe_sec002_strict.py` | CYTHON-SEC-002 | `SecurityPolicy.strict` is operative | strict flips unset allow_*; explicit override wins (exit 0) |
| `probe_sch001_schema.py` | CYTHON-SCH-001 | Versioned cache metadata schema | write stamps version; legacy/future incompatible (exit 0) |
| `repro_api003_sanitize.py` | CYTHON-API-003 | sanitize() collisions + non-ASCII | Distinct inputs stay distinct; output pure ASCII (exit 0) |
| `probe_batch001_partial.py` | CYTHON-BATCH-001 | Batch build partial report + resume | fail-fast carries committed+token; collect records failures (exit 0) |
| `probe_port001_toolchain.py` | CYTHON-PORT-001 | Resolved compiler keys the cache | reports effective compiler; different compiler → different key (exit 0) |
| `probe_tpl002_validation.py` | CYTHON-TPL-002 | Strict template-metadata validation | schema/typed-entry/containment checks reject bad metadata (exit 0) |
| `probe_obs001_diagnostics.py` | CYTHON-OBS-001 | Bounded capture + typed BuildDiagnostic | bounded log; failure attaches phase/module/versions (exit 0) |
| `probe_abi001_runtime.py` | CYTHON-ABI-001 | Runtime lifecycle contract + safe defaults | unload=False; free-threaded/subinterp rejected by default (exit 0) |
| `probe_test001_exclusivity.py` | CYTHON-TEST-001 | Interprocess build-lock exclusivity | two-process: held→timeout, released→acquired (exit 0) |
| `probe_perf001_dedup.py` | CYTHON-PERF-001 | Normalized-path dedup + bounded traversal | duplicate includes collapse; parent not duplicated; budgeted scan (exit 0) |

> Note on `repro_con001.py`: its exit-code polarity is inverted relative to the
> others — it is a *reproduction* of the original bug, so on a **fixed** tree it
> exits `1` (lock is exclusive, defect absent). The others are *verifications*
> and exit `0` on success.

## Adding a new probe

1. Name it `repro_<finding>.py` (reproduction) or `probe_<finding>_<aspect>.py`
   (invariant verification).
2. Keep it standalone: no test framework, no non-stdlib deps beyond the package.
3. Print a clear `VERDICT: OK|CHECK` line and return a matching exit code.
4. Add a row to the table above.
