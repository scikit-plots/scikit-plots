# Fresh-chat handoff — `scikitplot.annoy`

## Authority

1. The user's current workspace, build logs and requested edit boundary.
2. Current runtime files under `scikitplot/annoy/` and the exact upstream headers
   they declare from `scikitplot/cexternals/_annoy/src/`.
3. `STATE.json`, `MAINTENANCE.json`, `REVIEW.json`, `TRACKER.json`, and
   `EVIDENCE.json`.
4. Current tests and Meson definitions.
5. `history/` and `_backup/` only for provenance.

Never inherit a historical campaign status or old clean-build claim as current.

## The architecture to keep separate

`cexternals/_annoy` is a distinct native C++/pybind11 subsystem. `annoy` owns a
separate Cython extension whose source chain is:

```text
annoylib.pyx.in + annoylib.pxd.in
        | Tempita
        v
build/annoylib.pyx + build/annoylib.pxd
        | Cython --cplus
        v
build-generated C++
        | C++ compiler, directly including cexternals/_annoy headers
        v
scikitplot.annoy._annoy.annoylib
```

Separately, public `scikitplot.annoy.Index` subclasses
`scikitplot.cexternals._annoy.Annoy`. That native Python backend is not the same
type as `scikitplot.annoy._annoy.Index`.

## Current snapshot truth

The maintenance tooling should report `maintenance=PASS`, but **runtime `FAIL`**
because the Meson generation program `scikitplot/_build_utils/tempita.py` is
missing from this snapshot. Release must remain `BLOCKED` until generation,
Cython/native compilation, module tests and platform evidence are current.

`scikitplot/annoy/_annoy/annoymodule.cpp` is a checked-in fastannoy/pybind11
source but is not referenced by the current Annoy Meson target. Treat it as
inactive legacy evidence unless the build is intentionally redesigned; it is
not the generated Cython C++ source of truth.

## First commands

```sh
python -B maintenances/annoy/_maintenance/check_trackers.py --json
python -B maintenances/annoy/_maintenance/review_subsystem.py --json
python -B maintenances/annoy/_maintenance/tests/test_contract.py
```

Read `EVIDENCE.json` before claiming any compile or runtime success. `--release`
is intentionally fail-closed. `--update` must not be used to bless a broken
generation/build contract.
