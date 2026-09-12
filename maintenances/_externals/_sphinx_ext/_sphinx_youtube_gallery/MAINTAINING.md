# Maintaining `_sphinx_youtube_gallery`

This is the human and fresh-chat entry point for the YouTube extension family.

## Authority order

1. current user-supplied workspace and failure logs;
2. `_maintenance/STATE.json` and `_maintenance/TRACKER.json`;
3. current runtime source plus executable evidence;
4. current checkpoints / todo / lessons;
5. historical standalone validation artifacts.

Previous chat history is never an authority when these files are present.

## Three repository planes

```text
scikitplot/_externals/_sphinx_ext/        runtime plane
maintenances/_externals/_sphinx_ext/      maintenance/state/evidence plane
skills/_externals/_sphinx_ext/            fresh-chat routing plane
```

Runtime code must not import `maintenances/` or `skills/`.

## Ownership map

```text
YouTube URL grammar + player option validation -> _sphinx_youtube_core
catalog schema/query/sync                  -> _sphinx_youtube_gallery
bounded YAML/browser metadata/controls     -> _sphinx_collection
cards/grid/source-format delegation        -> _sphinx_gallery_grid
leaf iframe/player + thumbnail fallback    -> _sphinxcontrib_youtube
PyData component inventory                 -> _pydata_component_list
                                               (family-related, not runtime dependency)
```

Before changing code, determine which owner is broken. Do not fix a shared
contract by duplicating logic in the gallery.

## Executable architecture ownership

`MAINTENANCE.json` version 2 classifies each family relationship with a typed edge and
ties it to one or more capability IDs. The common checker merges those declarations with
AST-observed imports, so dynamic extension loading and normal Python imports share one
effective DAG. Capability IDs have exactly one family-wide owner.

Inspect the current graph before moving shared logic between packages:

```bash
python maintenances/_externals/_sphinx_ext/_maintenance_core/tools/report_architecture.py
```

Do not hand-edit a generated graph snapshot into authority. Update the owning manifest or
runtime source, then regenerate diagnostics.

## Fresh-chat sequence

Read `skills/.../_sphinx_youtube_gallery/SKILL.md`, this file,
`_maintenance/FRESH_CHAT_HANDOFF.md`, `STATE.json`, `TRACKER.json`, then
`todo/todo.md` and relevant `todo/lessons.md` before editing.

## Verification entry points

Common family gate:

```bash
python maintenances/_externals/_sphinx_ext/_maintenance_core/tools/check_all.py
```

YouTube wrapper:

```bash
python maintenances/_externals/_sphinx_ext/_sphinx_youtube_gallery/_maintenance/tools/check_trackers.py
```

See `_maintenance/VERIFICATION.md` for behavior gates and environment boundaries.
