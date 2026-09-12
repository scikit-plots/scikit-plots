---
name: cython-maintainer
description: Maintain scikitplot.cython as the independent runtime compilation service for caller-supplied Python/Cython code. Use for its public compile/import facade, security policy, Cython/setuptools/compiler integration, cache fingerprints, build locks, pins/GC, artifact loading, runtime capability contracts, shipped templates/probes, tests, and release evidence. Do not use it for project build-time Cython owners such as scikitplot.annoy.
---

# `scikitplot.cython` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and
`skills/`. This skill owns the **runtime compiler service** in
`scikitplot.cython`; it does not own every Cython-using submodule in the project.

## Read first

1. `maintenances/cython/MAINTAINING.md`
2. `maintenances/cython/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/cython/_maintenance/STATE.json`
4. `maintenances/cython/_maintenance/FAMILY.md`
5. `maintenances/cython/_maintenance/VERIFICATION.md`

Then run:

```sh
python -B maintenances/cython/_maintenance/check_trackers.py --json
python -B maintenances/cython/_maintenance/review_subsystem.py --json
python -B -m pytest maintenances/cython/_maintenance/tests -q -p no:cacheprovider
```

## Do not confuse runtime Cython with build-time Cython

`scikitplot.cython` accepts caller programs and may invoke Cython/setuptools and
a host compiler at runtime. `scikitplot.annoy` owns its own Tempita/Cython/Meson
build chain. A defect in Annoy's `.pyx.in`/`.pxd.in`, generated C++, or shared
Annoy headers belongs to the Annoy owners, not here.

Production `scikitplot.cython` intentionally has no sibling-submodule imports.
Adding one changes a valuable independence property and requires explicit review.

## Preserve the trust path

Treat the build pipeline as a transaction:

`facade -> security validation -> deterministic fingerprint -> per-key lock ->
staging build -> validate artifact -> atomic publish -> import`.

Never bypass `_public._validate_build_security` for a new public build entry
point. Do not duplicate security policy in the builder; keep one choke point.
Never make untrusted compiler/linker arguments executable through shell parsing.
Optional toolchain packages must stay lazy so importing the package does not
require Cython, setuptools, NumPy, or pybind11.

Cache keys must encode every input that can change a binary: source/support
content, directives/flags, relevant package/toolchain versions, interpreter ABI,
and the resolved compiler identity. If an input is omitted, do not compensate by
weakening cache reuse checks.

## State is part of correctness

Build locks are interprocess state, not just convenience. A probe must never
reclaim a fresh live lock; stale recovery and wait timeout remain separate
concepts. Release only the lock token this process owns.

Pins, cache metadata, GC, staging directories, exports, and raw-artifact staging
must remain atomic/fail-closed. Interrupted work must leave either the old valid
state or no authoritative new state—never a half-published entry.

## Templates and probes are executable assets

`_templates/` is not ordinary documentation. Validate metadata containment and
source pairing whenever assets change. Keep probe scripts standalone and update
the probe README when adding/removing one. `repro_con001.py` is special: exit 1
means exclusivity is fixed, unlike the normal exit-0 verification probes.

## Evidence rules

Run runtime tests from a complete package root. This supplied partial snapshot
has no `scikitplot/__init__.py`; direct pytest can therefore import the submodule
as top-level `cython` and shadow Cython's own `cython.py`.

Always inspect skip reasons. Current full-suite order can pollute
setuptools/distutils global state and self-skip three real-compile checks even
though they pass alone. Do not report those as green compiler coverage.

Current source hygiene is also red while `scikitplot/cython/__pycache__` ships in
the tree. Maintenance tooling may be PASS while runtime/release status remains
FAIL/BLOCKED.

For release, supplement Linux evidence with the supported compiler/platform
matrix, especially Windows/MSVC behavior and any browser/WASM prebuilt-only
claims. Keep runtime, maintenance, and skill planes separate; historical files
are provenance, never current authority.
