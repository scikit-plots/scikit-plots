# Fresh-chat handoff — `_sphinx_youtube_gallery`

You are maintaining `scikitplot._externals._sphinx_ext._sphinx_youtube_gallery`
inside the wider Sphinx-extension family. Do not rely on earlier chat history.

Read, in order:

1. `skills/_externals/_sphinx_ext/_sphinx_youtube_gallery/SKILL.md`
2. `maintenances/_externals/_sphinx_ext/_sphinx_youtube_gallery/MAINTAINING.md`
3. this file
4. `STATE.json`
5. `TRACKER.json`
6. `../todo/todo.md`
7. `../todo/lessons.md` when relevant
8. current source and executable evidence

For an independent review or PR pass, run `_maintenance_core/tools/review_subsystem.py`
against this subsystem's `MAINTENANCE.json`, or `review_all.py` for family reconciliation.
`REVIEW.json` is declarative and may select registered deterministic checks only.

## Current candidate status — 2026-09-10

All locally available maintenance and dependency-free gates are green: 35/35 common-core
tests, 2/2 family subsystems, a 7-package effective runtime graph with no cycle, 10
unique capability ownership declarations, 67/67 dependency-free doctests, capability/sync/parity/
latest-source checks, 14 dependency-free public docstrings, provider-core identity, and
two-layout provider-core namespace smoke. JavaScript syntax is also green for all three
live-control scripts.

Do **not** call the candidate release-closed yet. Sphinx rendering/layout verification is
unavailable here because the Sphinx/docutils stack is absent, and jsdom-backed browser
behavior is unavailable because `jsdom` is absent. Run those layers downstream before
promotion. Historical standalone validation JSON/SHA files remain immutable provenance.

## Current architecture

The provider grammar and leaf option validators live in dependency-free
`_sphinx_youtube_core`. `_sphinx_youtube_gallery.reference` and
`_sphinx_youtube_gallery._video_options` are compatibility facades only.

Runtime DAG:

```text
_sphinx_youtube_gallery -> _sphinx_collection
                        -> _sphinx_gallery_grid -> _sphinx_collection
                        -> _sphinxcontrib_youtube -> _sphinx_youtube_core
                        -> _sphinx_youtube_core

_pydata_component_list is family-related, not a gallery dependency.
```

The maintenance graph itself is also executable: schema-v3 `MAINTENANCE.json` files retain the v2 typed architecture model and use
typed dependency edges (`runtime_required`, `runtime_optional`, `test_only`,
`maintenance_only`, `family_related`) and capability IDs.
`_maintenance_core/tools/report_architecture.py` derives the effective graph from those
declarations plus current AST import evidence. Dynamic extension references participate
in cycle detection.

## Evidence rule

Historical `validation.json`, `verification.json`, `validation_summary.json`, and
`_live_controls/sha256.json` are provenance from older standalone revisions. Never
rewrite them merely because the wide repository changed. Record new wide-repository
observations in `STATE.json` and current verification notes instead.

## Environment rule

Missing `sphinx`, `docutils`, `sphinx_design`, or `myst_parser` means the associated
integration layer is unavailable in that environment. It is neither a product
failure nor an inferred pass.
