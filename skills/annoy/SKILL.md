---
name: annoy-maintainer
description: Maintain scikitplot.annoy as the distinct Python/Cython Annoy subsystem. Use for the high-level Index facade, mixins and typing, Tempita/Cython generation, Cython ABI declarations, Annoy-specific build/tests, and cross-boundary review against the independently owned cexternals/_annoy native C++ subsystem.
---

# Annoy Cython/Python maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`, and
`skills/`. This skill owns `scikitplot.annoy`; it does **not** make
`scikitplot.cexternals._annoy` part of the same maintenance plane.

## Read first

1. `maintenances/annoy/MAINTAINING.md`
2. `maintenances/annoy/_maintenance/FRESH_CHAT_HANDOFF.md`
3. `maintenances/annoy/_maintenance/STATE.json`
4. `maintenances/annoy/_maintenance/FAMILY.md`
5. `maintenances/annoy/_maintenance/VERIFICATION.md`

Then run:

```sh
python -B maintenances/annoy/_maintenance/check_trackers.py --json
python -B maintenances/annoy/_maintenance/review_subsystem.py --json
```

## Never merge the two Annoy owners mentally

`cexternals/_annoy` owns the shared hand-written C/C++ headers and the separately
compiled native/pybind11 `Annoy` type. `annoy` has two relationships to it:

- public `scikitplot.annoy.Index` subclasses/imports that native `Annoy` type;
- private `scikitplot.annoy._annoy.Index` is a different Cython extension built
  from `annoylib.pyx.in` + `annoylib.pxd.in` and directly declares against
  `annoylib.h`, `kissrandom.h`, and `annoy_type_support.h`.

For shared-header/native implementation defects, route the edit to
`skills/cexternals/_annoy/SKILL.md` and verify affected consumers. Do not fork a
header into `annoy`.

## Generation is part of correctness

The authoritative Cython sources are the `.in` templates. Meson must generate
`annoylib.pyx` and `annoylib.pxd`, then Cython generates C++ in the build tree.
Do not patch generated build outputs. The checked-in
`scikitplot/annoy/_annoy/annoymodule.cpp` is currently inactive legacy
fastannoy/pybind11 source, not the generated Cython C++ source.

The supplied snapshot has a known structural blocker: Meson names
`scikitplot/_build_utils/tempita.py`, which is absent. Keep runtime status red
until the generation path is genuinely repaired; do not bless it via tracker or
evidence updates.

## Verify the right layer

For template/declaration changes, regenerate in a clean build and compile the
Cython extension. For public `Index` changes, also test against the independently
compiled native `cexternals._annoy.Annoy` backend. For shared C++ changes, use
the native-source maintainer and rebuild Annoy plus direct sibling consumers as
appropriate. Static header existence checks never prove Cython/C++ ABI parity.

Keep runtime, maintenance and skill planes separate. Treat JSON metadata as
data, never as command input. Historical run files are provenance only; current
PASS claims require current evidence.


The active repository build requires **C++17**. Template-local C++14 comments/directives are historical/stale and must not be treated as the current compiler contract.
