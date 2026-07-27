<!--
Authors: The scikit-plots developers
SPDX-License-Identifier: BSD-3-Clause
-->
# Run 0 — baseline & tiered gate (annoy subsystem)

## Environment (measured)
- Python 3.12.3; numpy 2.4.4; g++ (Ubuntu 13.3.0) present; Cython 3.2.9,
  meson 1.11.2, ninja 1.13 installed via pip (`--break-system-packages`).
- No compiled extensions shipped in the zip (0 `.so`).

## Build reality (measured, honest)
- **Meson cannot run from the package-only zip.** The top `meson.build` has no
  `project()` and every submodule `meson.build` references parent-scope
  variables (`_root_cython_tree`, `py`, `dep_list`, `inc_dir_py`,
  `cython_cpp_args`, …). The repo root (`pyproject.toml`, root `meson.build`,
  `_build_utils`) is **not** in this zip. → Full meson build is a **CI/full-checkout**
  concern (see playbook deferrals).
- **Viable in-sandbox gate: direct per-extension `cython + g++`**, replicating
  what meson does. Proven on `kiss_random`:
  1. `cython --cplus -3 -I <pkg-root> kiss_random.pyx -o kiss_random.cpp`
  2. `g++ -O3 -fPIC -shared -std=c++17 -I. -I<py-include> -I<numpy-include> kiss_random.cpp -o kiss_random<EXT>.so`
  3. import + `pytest --noconftest <sub>/tests`
  - The absolute cimport `from scikitplot.random._kiss...` requires a
    **`scikitplot`-named package root** on Cython's `-I` path (meson provides
    this via its copyfile tree).

## Baseline gate number (measured)
- `random/_kiss`: **99 passed, 6 failed** with the current sandbox harness.
  - The 6 failures are **all** `TestPickleSupport` and are a **harness artifact**,
    not a defect: the symlinked root imports the package as `work.*`, so
    `pickle` cannot re-import the class by its baked-in qualified name
    (`Can't pickle <class 'work.random._kiss.kiss_random.Kiss32Random'>`).
  - **Fix for a clean baseline (Run 1 setup):** build/run under a real
    `scikitplot`-named directory (not a `work` symlink) so `__module__` is
    `scikitplot.random._kiss.kiss_random`. Expected: **105/105 green**.
- `annoy/`, `cexternals/_annoy`, `memmap/_memmap`: not yet built this run
  (Tier-A build per submodule is the first action of their respective runs).

## Tiered gate (proposed commands)
Let `ROOT` be a real `scikitplot`-named package root (dir, not symlink).

- **Tier A — submodule** (the run's own contract):
  `build_ext <sub>` then
  `PYTHONPATH=$ROOT/.. python -m pytest -q --noconftest scikitplot/<sub>/tests`
- **Tier B — integration ring** (annoy + cexternals + memmap + random):
  build all four, run all four test dirs; **an ring failure BLOCKS the run**
  (per confirmed policy), it is not merely logged.
- **Tier C — repo import smoke:** import every built extension under the real
  `scikitplot` name; assert no import error.

`--noconftest` is required because the package-root `conftest.py` pulls in the
whole compiled package (`scikitplot._lib._ccallback_c`, `hypothesis`, …), which
is out of scope for a per-submodule gate.

## Blast radius (measured)
- **Downstream (who imports annoy): contained.** The wider repo does not broadly
  import annoy; the only cross-boundary consumers found are within
  `cexternals/_annoy/*`. The feared whole-repo blast radius does **not**
  materialize.
- **Upstream (what annoy imports):** `from .._utils` (×6), `from ..annoylib`
  (×4), `from ..cexternals` (×2), `from .._mixins` (×1). Related native siblings
  `memmap/_memmap` (mem_map) and `random/_kiss` (kiss_random) are co-reviewed
  because the guide's Part I dissects their C sources (`mman.h`, `kissrandom.h`).
- Integration ring for gating = **annoy + cexternals + memmap + random**.

## Clutter inventory (safe to prune for reviewability; no logic change)
Non-Python upstream bindings & dead variants (out of the Python subsystem):
`annoy/tests/annoy_test.go`, `annoy/tests/annoy_test.lua`,
`cexternals/_annoy/annoy-dev-1.rockspec`, `cexternals/_annoy/src/annoygomodule.i`,
`cexternals/_annoy/src/annoylib_review.h`, `cexternals/_annoy/src/annoylib_v0.h`,
`cexternals/_annoy/src/annoymodule.cu`, and `annoy/_annoy/backup_template/*` (6 files).
Pruning is a **presentation** change; sequence it with R31/R32 and keep a green
ring. (Confirmed policy: no moves mid-hardening — this is deletion of dead/foreign
artifacts, to be batched, not file relocation.)
