# YTG-M002 — executable architecture graph and capability ownership

## Goal

Turn the Sphinx-extension maintenance family from a dependency-list checker into an
architecture authority that understands both **how packages relate** and **which package
owns each shared capability**.

## Design

`MAINTENANCE.json` schema version 2 uses typed `dependency_edges`. Each edge records its
target, relationship kind, evidence kind, and the capabilities that justify the edge.
The common core remains backward-readable for version-1 manifests so historical bundles
can still be inspected, but the wide repository has one v2 authority per maintained
subsystem.

The common architecture engine combines AST-observed imports with declared dynamic
runtime edges. This is essential for `_sphinx_youtube_gallery -> _sphinxcontrib_youtube`,
which is resolved dynamically. A reverse leaf import therefore participates in cycle
detection even though the forward edge is not a normal Python import.

Capability ownership is family-wide and unique. For the YouTube/UI family the current
owners are:

```text
youtube.reference-grammar          -> _sphinx_youtube_core
youtube.player-options             -> _sphinx_youtube_core
youtube.catalog-query-sync         -> _sphinx_youtube_gallery
sphinx.collection-controls         -> _sphinx_collection
sphinx.gallery-grid-presentation   -> _sphinx_gallery_grid
youtube.iframe-player              -> _sphinxcontrib_youtube
sphinx.pydata-component-inventory  -> _pydata_component_list
```

The AI assistant also declares its high-level runtime capability ownership without
creating any runtime dependency on the maintenance plane.

## Negative evidence

The maintenance-core suite now contains 21 tests. In addition to the previous structural
and security cases, it proves that the architecture layer turns red for:

- a dynamic runtime edge that participates in a cycle;
- a comment pretending to be source-reference evidence;
- a `family_related` edge that becomes an actual runtime import;
- capability/target owner mismatch;
- unknown capability references;
- duplicate family-wide capability ownership;
- undeclared imports using multiple supported relative/absolute import spellings;
- schema-v2 manifests that retain legacy dependency arrays as a second authority;
- dependency kinds paired with semantically incompatible evidence classes.

## Diagnostics

`_maintenance_core/tools/report_architecture.py` emits deterministic text, JSON, or
Mermaid diagnostics from current manifests plus runtime AST evidence. Generated reports
are diagnostics, not a second committed source of truth.

## Evidence — 2026-09-10

- maintenance-core unit tests: **21/21 passed**;
- common family gate: **GREEN, 2/2 maintained subsystems**;
- AI-specific composed maintenance checker: **GREEN (repository)**;
- YouTube-specific maintenance checker: **GREEN**;
- effective runtime graph: **7 packages, no cycle**;
- capability registry: **10 unique capability IDs across current maintained subsystems**;
- repository syntax/data audit: **295 Python AST parses + 27 JSON parses, zero errors**;
- architecture slice diff from the previous candidate: **6 added, 14 modified, 0 deleted**;
- runtime byte changes in this slice: **0**;
- cache/compiled residue: **0**;
- four principal historical standalone-v10 evidence files: **byte-identical**.

This checkpoint changes maintenance metadata/tooling only. The runtime provider-core
architecture established by YTG-M001 is unchanged.

## Rollback

Rollback may remove the v2 architecture tooling/manifests and restore the previous v1
manifest representation without touching runtime packages. Do not roll back
`_sphinx_youtube_core` as part of this checkpoint; that runtime change belongs to
YTG-M001.
