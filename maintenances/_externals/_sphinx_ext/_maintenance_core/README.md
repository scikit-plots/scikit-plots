# `_sphinx_ext` maintenance core

This directory is the maintenance-only control plane shared by Sphinx-extension
subsystems. Runtime packages under `scikitplot/_externals/_sphinx_ext/` must never
import it.

## Manifest v2 and v3

Each maintained subsystem owns a `MAINTENANCE.json`. Version 2 replaces the ambiguous
pair of `runtime_requires` / `family_related` arrays with typed `dependency_edges`
and explicit `capability_ownership`.

A dependency edge has four required fields:

- `target`: sibling runtime package or related family package;
- `kind`: `runtime_required`, `runtime_optional`, `test_only`, `maintenance_only`, or
  `family_related`;
- `evidence`: `python_import`, `source_reference`, `architectural_relation`,
  `test_import`, or `maintenance_reference`;
- `capabilities`: one or more stable capability IDs that explain why the edge exists.

`python_import` is proven from Python AST imports. `source_reference` is intended for
runtime relationships such as dynamically resolved Sphinx extensions and is proven
from executable string syntax, not comments. This prevents prose from satisfying a
runtime dependency declaration.

Version-1 manifests remain readable so historical standalone maintenance bundles can
still be inspected. Version 2 introduced typed edges/capabilities. Version 3 keeps that
architecture model and additionally requires a subsystem-owned `review_profile` pointing
to `_maintenance/REVIEW.json`. Do not maintain legacy dependency arrays beside v2/v3
edges; that creates two authorities.

## Capability ownership

Capability IDs describe *what* a package owns rather than merely which package imports
which other package. Every capability has exactly one family-wide ownership declaration.
Dependency edges cite the capabilities they consume.

The family gate rejects:

- duplicate capability owners;
- unknown capabilities referenced by an edge;
- a capability cited through a dependency whose target is not the capability owner;
- ownership declarations outside the declaring subsystem's own runtime/dependency
  boundary;
- non-runtime edges that silently become runtime imports.

This makes architectural intent executable. A future maintainer should fix behavior in
the capability owner rather than duplicating shared logic in a consumer.

## Effective runtime graph

The architecture graph combines:

1. AST-observed sibling runtime imports; and
2. declared `runtime_required` / `runtime_optional` edges, including dynamic
   `source_reference` relationships.

Cycles are checked against the combined graph. This matters for extension systems where
one direction can be dynamically resolved rather than expressed as a Python import.

Inspect the current graph without importing runtime extensions:

```bash
python maintenances/_externals/_sphinx_ext/_maintenance_core/tools/report_architecture.py
python maintenances/_externals/_sphinx_ext/_maintenance_core/tools/report_architecture.py --format json
python maintenances/_externals/_sphinx_ext/_maintenance_core/tools/report_architecture.py --format mermaid
```

The report is generated evidence, not committed source of truth. Manifests plus current
runtime source remain authoritative.

## Independent review plane

Manifest v3 adds a deterministic review profile. Review profiles describe package targets,
independent lenses, and PR/release policy; they cannot embed arbitrary executable commands.
The registered review engine can run one subsystem's lenses/packages concurrently or review
multiple subsystems independently and reconcile afterward.

```bash
python maintenances/_externals/_sphinx_ext/_maintenance_core/tools/review_subsystem.py \
  maintenances/_externals/_sphinx_ext/_sphinx_youtube_gallery/MAINTENANCE.json --jobs 4
python maintenances/_externals/_sphinx_ext/_maintenance_core/tools/review_all.py --jobs 4
```

`PR_READY` and release promotion are deliberately separate. Optional verification that is
`UNAVAILABLE` may leave a subsystem reviewable while still blocking release promotion.
Use `--require-release` for a fail-closed release command. See `REVIEWING.md` for the
finding contract and agent-safety boundary.

## Adding a subsystem

A new maintained subsystem should normally need only its local `MAINTENANCE.json`,
`STATE.json`, `TRACKER.json`, `REVIEW.json`, fresh-chat handoff, skill entry, and optional
domain checker. `check_all.py` auto-discovers subsystem manifests; no central subsystem
list needs editing.

Run the common gate with bytecode disabled when freezing a candidate:

```bash
PYTHONDONTWRITEBYTECODE=1 python \
  maintenances/_externals/_sphinx_ext/_maintenance_core/tools/check_all.py
```
