---
title: "Public API spells 'colour map' 3 ways (cmap, color_map, palette)"
status: open
kind: "api"
area: "scikitplot"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot"
towncrier_type: "api"
towncrier_fragment: ""
---

# Public API spells 'colour map' 3 ways (cmap, color_map, palette)

## Summary

Across public callables, the idea 'colour map' is named `cmap`, `color_map`, `palette`. `cmap` is matplotlib's name; `palette` (seaborn) and `color_map` appear next to it.

## Why it matters

A user who learned one submodule guesses wrong in the next; wrappers and configuration files have to map names; documentation and generated skills repeat the inconsistency.

## Current evidence

- `python -B tools/maint_tools/sweep_runtime.py --json out.json` (main cba9383), cluster report:
- `cmap` in 11 public callable(s): `api.plot_calibration`, `api.plot_classifier_eval`, `api.plot_confusion_matrix`, `api.plot_feature_importances`, `api.plot_pca_2d_projection`, `api.plot_precision_recall` ...
- `color_map` in 2 public callable(s): `visualkeras.graph_view`, `visualkeras.layered_view`
- `palette` in 4 public callable(s): `seaborn.aucplot`, `seaborn.decileplot`, `seaborn.evalplot`, `seaborn.modelplot`

## Root cause / current understanding

Submodules were written at different times against different upstream conventions; there is no naming register.

## Expected behavior

One canonical name per idea (proposed: `cmap`), documented in a naming register in the engineering rules and used by every new or changed public callable; where two scopes differ (a per-file and a per-archive limit) the name says the scope with one pattern.

## Affected paths and ownership

- the public callables listed above
- `plugins/_shared/skills/scikitplot-engineering-rules/` (naming register)
- `tools/maint_tools/sweep_runtime.py` (`CLUSTERS`, the check)

## Constraints and non-goals

Renaming public parameters is an API decision; the campaign rule is no backward compatibility, so a rename lands with its fragment and usage-skill update (Rule 172).

## Edge cases to cover

- Classes whose `__init__` takes the old name positionally
- Configuration dataclasses serialised to JSON with the old key
- Names that look alike but differ in meaning (keep them, document why)

## Proposed direction

Decide the canonical name, rename in one change per submodule, add the cluster to a `sweep_runtime` gate so a new divergent spelling fails review.

## Verification / acceptance criteria

- `sweep_runtime.py` reports no cluster for this idea, or only the documented exceptions

## Documentation impact

API reference, usage skills, the naming register.

## Release-note promotion

Required: `scikitplot` / `api`, one bullet, when the change lands.
