# Maintaining `scikitplot.annoy`

This is the current entry point for the **Annoy Cython/Python submodule**. It is
not the maintenance entry point for `scikitplot.cexternals._annoy`.

## Two different Annoy compiled domains

```text
cexternals/_annoy
    shared C/C++ headers + separately compiled native/pybind11 Annoy backend
             |                         |
             | direct header ABI       | Python inheritance/import
             v                         v
annoy/_annoy (Cython)           annoy.Index (high-level Python facade)
Tempita -> .pyx/.pxd -> Cython       subclasses cexternals._annoy.Annoy
-> generated C++ -> extension
```

`scikitplot.annoy` therefore has **two compiled relationships** with the
independent native subsystem:

1. its private Cython extension compiles directly against
   `cexternals/_annoy/src/{annoylib.h,kissrandom.h,annoy_type_support.h}`;
2. its public high-level `Index` inherits from the separately compiled
   `scikitplot.cexternals._annoy.Annoy` Python type.

Do not collapse those into one owner and do not copy native headers into
`scikitplot/annoy`.

## Start here

```sh
python -B maintenances/annoy/_maintenance/check_trackers.py --json
python -B maintenances/annoy/_maintenance/review_subsystem.py --json
python -B maintenances/annoy/_maintenance/tests/test_contract.py
```

The supplied snapshot currently has a maintenance framework that can be green,
but the runtime/build structural lane is expected to fail because
`scikitplot/annoy/_annoy/meson.build` names
`scikitplot/_build_utils/tempita.py` and that generator is absent. Do not turn
that into PASS by changing evidence or inventory metadata.

Read `_maintenance/FRESH_CHAT_HANDOFF.md` next. Historical A00-A21 campaign
notes remain under `_maintenance/history/`; they are evidence/provenance, not
current authority.
