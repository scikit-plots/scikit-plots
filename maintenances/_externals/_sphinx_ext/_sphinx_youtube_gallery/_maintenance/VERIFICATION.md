# Verification map

Verification is layered so a fresh maintainer can distinguish source defects from
unavailable optional build dependencies.

## Layer 0 — maintenance architecture

`python ../_maintenance_core/tools/check_all.py` validates runtime / maintenance /
skill plane separation, manifest/state/tracker shape, source-anchor SHA shape,
checkpoint ownership, subsystem and family-wide contract-ID uniqueness, typed dependency edges, capability ownership, declared/effective runtime graph consistency,
`family_related` separation, undeclared cross-stack imports, dynamic-edge dependency cycles,
fresh-chat handoff/skill routing, JSON schema syntax, credential-key hygiene, and
cache/compiled residue.

The subsystem wrappers compose this common gate rather than reimplement it. The AI
assistant retains its deeper security/test-ownership checks; the YouTube wrapper adds
provider-core ownership, core dependency-freedom, compatibility-facade identity, and
leaf-player directionality checks.

The current architecture can be inspected without importing runtime extensions:

```bash
python maintenances/_externals/_sphinx_ext/_maintenance_core/tools/report_architecture.py
python maintenances/_externals/_sphinx_ext/_maintenance_core/tools/report_architecture.py --format json
```

The report combines AST-observed imports with declared dynamic runtime edges; it is
diagnostic output, not a committed replacement for `MAINTENANCE.json`.

## Layer 0b — independent review / PR reconciliation

`_maintenance_core/tools/review_subsystem.py` reviews one subsystem's declared runtime
package targets and review lenses independently. `review_all.py` can run subsystems in
parallel and reconciles only after each subsystem report completes. Review profiles are
declarative: they select registered deterministic checks and cannot embed shell/Python/env
execution fields.

Current result: 2/2 subsystems are `PR_READY`; all seven reviewed runtime package lanes
are GREEN; reconciliation has 0 ERROR, 0 WARNING, 4 UNAVAILABLE, 0 INFO. The four
UNAVAILABLE findings are the already-known Sphinx/jsdom layers, so YouTube release
promotion remains blocked while PR review remains valid. `--require-release` converts
that release status into a nonzero command result for release CI.

The first baseline review found one real maintenance drift: the AI skill's fresh-chat
sequence omitted `TRACKER.json` although its handoff required it. That routing entry was
repaired. The reviewer's initial handoff/active-checkpoint false positives were fixed in
the reviewer itself rather than mutating valid historical state.

## Layer 1 — dependency-free YouTube contracts

Run capability, sync, static parity, latest-source, documentation, canonical
provider-core doctest and compatibility-identity checks. These must work without
importing Sphinx/docutils where the test is explicitly dependency-free.

Current wide-repository result (2026-09-10):

- maintenance core: 35/35;
- family gate: 2/2 subsystems GREEN;
- maintenance manifests: v3 for AI + YouTube (v2 typed-edge/capability architecture retained);
- effective runtime graph: 7 packages, no cycle;
- capability ownership: 10 unique capability IDs;
- dependency-free doctests: 67/67;
- capability / sync / parity / latest-source: GREEN;
- documentation: 14 dependency-free public docstrings;
- `_sphinx_youtube_core`: identity GREEN and 2-layout namespace smoke GREEN;
- JavaScript parser/syntax: 3/3 files GREEN.

## Layer 2 — Sphinx integration

`_gallery_revision/verify_extension_layouts.py` first proves `_sphinx_youtube_core`
imports in both namespace shapes, then exercises extension ordering, dependency
auto-loading and mixed-namespace rejection when Sphinx is installed.
`verify_gallery_rendering.py` covers RST/MyST rendering, escaping, nesting, links and
empty results.

In this workspace the core namespace smoke is GREEN for both layouts, while the
Sphinx layer is **UNAVAILABLE**: `sphinx`, `docutils`, `sphinx_design`, and (for MyST
rendering) `myst_parser` are absent. Do not reinterpret that as GREEN or product
failure.

Several older live-control Python checks also transitively require Sphinx/docutils.
The Node browser-behavior checks require `jsdom`, which is absent here; their three
JavaScript files do pass `node --check`.

## Historical evidence

`_live_controls/sha256.json`, `_live_controls/validation.json`, and JSON evidence under
`_gallery_revision` describe the older standalone lineage. They are provenance, not a
mutable cache. The current review compared the four principal historical JSON/SHA
artifacts with `scikit-plots.zip` and found them byte-for-byte unchanged.

Current wide-repository evidence belongs in `STATE.json` / checkpoints, not by silently
refreshing old hashes.
