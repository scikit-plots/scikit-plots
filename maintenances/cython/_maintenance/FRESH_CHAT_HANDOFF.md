# Fresh-chat handoff — `scikitplot.cython`

Read `MAINTENANCE.json`, `STATE.json`, `FAMILY.md`, and `VERIFICATION.md` before
editing. Then run the maintenance tests and both JSON review commands.

## Mental model

The critical path is:

`public facade -> security validation -> deterministic fingerprint -> per-key
lock -> staging build -> artifact validation -> atomic publish -> import`.

Every edit must preserve fail-closed input validation and leave cache/lock/pin
state understandable to a later process.  Never patch cache metadata to make a
bad artifact reusable.

Optional toolchain packages (`Cython`, `setuptools`, NumPy, pybind11) must not
become unconditional module-scope dependencies.  This subsystem must not import
sibling `scikitplot` services.

## Current snapshot

Maintenance tooling is healthy, but release is blocked.  The source archive
contains a bytecode cache file.  The supplied partial package also lacks
`scikitplot/__init__.py`, so running pytest directly mispackages this directory
as top-level `cython` and shadows Cython's own `cython.py`; use a complete
checkout for release evidence.

A temporary package-root harness proves the runtime suite reaches 1240 passed / 8
skipped. Three skips are specifically suite-order setuptools/distutils pollution
and pass when run alone. Treat those as missing full-run evidence, not green.

Special probe note: `repro_con001.py` intentionally exits **1** when the old lock
overlap defect is absent; its polarity is inverted relative to verification
probes.
