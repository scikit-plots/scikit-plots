<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 0b — real scoped meson build + ring gate (supersedes Run 0 build note)

With the repo-root files provided (`pyproject.toml`, root `meson.build`,
`meson.options`, `meson_cpu/`, `scikitplot/_build_utils/`), the **real meson
build now works**, scoped to the four related submodules. This replaces the
Run-0 `cython + g++` workaround with the authoritative pipeline
(`.pyx.in` → tempita → cython → C++ → link).

## Reproducible recipe (verified)

### Layout
```
repo/
  pyproject.toml  meson.build  meson.options  CITATION.bib(*)  tools/f2py/generate_f2pymod.py(*)
  meson_cpu/
  scikitplot/            <- real dir named 'scikitplot' (rule L-HARNESS)
    _build_utils/        <- overlaid from _build_utils.zip
```
(*) sandbox scaffolds — see "Scaffolds" below.

### Build deps
`pip install --break-system-packages cython meson ninja meson-python tempita pybind11 pytest hypothesis`

### Configure + scoped compile
```
meson setup builddir -Dallow-noblas=true
ninja -C builddir \
  scikitplot/random/_kiss/kiss_random.cpython-312-x86_64-linux-gnu.so \
  scikitplot/memmap/_memmap/mem_map.cpython-312-x86_64-linux-gnu.so \
  scikitplot/cexternals/_annoy/annoylib.cpython-312-x86_64-linux-gnu.so \
  scikitplot/annoy/_annoy/annoylib.cpython-312-x86_64-linux-gnu.so
```
Only these 4 targets compile — not the full 69-target tree. (`meson.options`
has no per-subpackage flags, so scoping is done at the `ninja` target level.)

### Stage generated artifacts into the source tree (editable-style)
Copy from `builddir/scikitplot/…` into `scikitplot/…`: the 4 `.so`, plus the
meson-generated `config/__config__.py`, `config/_citation.py`, and `version.py`.
(The `config` subpackage imports these at load; the annoy extension does
`from scikitplot import get_config` at module init.)

### Tiered gate (verified commands)
Run via `gate.py` (documented harness — pre-stubs the unbuilt `scikitplot.api`
namespace and binds the real `get_config`, so the package's lazy loader
short-circuits in a scoped build; rule L-GATE-SCOPE):
```
PYTHONPATH=repo python3 gate.py scikitplot/<sub>/tests
```

## Ring baseline (measured, real meson build)

| Suite | Result |
|---|---|
| `random/_kiss/tests` | **105 passed** |
| `memmap/_memmap/tests` | **49 passed** |
| `annoy/_annoy/tests` (Cython `Index`, all dtype/metric combos) | **333 passed** |
| `annoy/tests` metric (euclidean/manhattan/angular) | **42 passed** |
| `annoy/tests` core (seed/types/holes/dot/hamming/serialize) | **32 passed** |
| **Confirmed green total** | **561 passed** |

- `random/_kiss` is now **105/105** — the 6 Run-0 pickle failures were purely the
  symlink-naming artifact and are gone under the real `scikitplot` root
  (confirms rule L-HARNESS).
- `annoy/_mixins/tests` and `cexternals/_annoy/_annoy_test.py`: no tests collected
  (empty dir / non-default filename pattern) — not failures.

## Deferred to CI — sandbox resource limits (honesty rule; not failures)
OOM-killed by the sandbox on large-index builds / concurrency, to run on CI:
`annoy/tests/index_test.py`, `accuracy_test.py`, `memory_leak_test.py`,
`multithreaded_build_test.py`, `on_disk_build_test.py`, `threading_test.py`.
These are exactly the concurrency/lifetime suites relevant to TIER-0 findings
(§6.2/6.4, CY-009) — their real verification belongs on a CI runner with memory
headroom, per the playbook deferrals.

## New finding surfaced by the real build (fold into playbook)
- **BUILD-WARN-001** — `cexternals/_annoy/src/annoymodule.cc:4051`
  (`annoy_build_portable_blob`) emits `-Wstringop-overflow` on a
  `std::vector<unsigned char>::insert`. Maps to the persistence/on-disk cluster
  (near §6.5–6.7 / CY-005). Confidence: compile-verified. Triage in TIER 1;
  it is a candidate real defect in portable-blob assembly, not just noise
  (relates to CY-020 warning-volume, but this one is actionable).

## Scaffolds used (documented; not source changes)
- `CITATION.bib` — placeholder; `config/meson.build` reads it via `fs.read`.
- `tools/f2py/generate_f2pymod.py` — executable stub; resolved by `find_program`
  at configure but never invoked (no Fortran/`.pyf` target in scope; exits non-zero
  if ever called so a missing real tool is loud, not silent).
- `gate.py` — scoped-gate runner (L-GATE-SCOPE).
Each is a harness artifact for in-sandbox verification; on CI the real repo
provides all three.

## Heavy-compile timeout workaround (annoy/_annoy)
The generated annoylib C++ compile can exceed one sandbox command's timeout.
For a `.pyx.in` edit: `touch annoylib.pyx.in`; build the `annoylib.pyx` target
(tempita regen); then build the `.so` target across two resumes (Cython->cpp,
then cpp->link — ninja caches each completed stage). Only `cp` the `.so` after the
final `[..] Linking target` line. Confirm freshness by asserting a compiled
docstring/behaviour (a stale `.so` silently passes older behaviour).
