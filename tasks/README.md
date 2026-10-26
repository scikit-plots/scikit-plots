# Work records: where they live

Since increment 4R each work record lives with what it is about, so a
session reads only what its work needs. This file is the map; nothing else
belongs in `tasks/` (the checker below fails on any other file). The
repository-wide procedure (read order, planes, evidence) is
[`docs/maintenances/MAINTAINING.md`](../docs/maintenances/MAINTAINING.md).

**Instructions that name `tasks/todo.md` or `tasks/lessons.md`** (older
handoffs, standing preferences) mean, today: plans and results go to the
owner's active `todo/todo.md`; lessons go to `skills/scikitplot-engineering-rules/` as its
`SKILL.md` describes.

| Record | Location |
|---|---|
| Active plan and results of an area | `maintenances/<area>/todo/todo.md` (e.g. [corpus](../maintenances/corpus/todo/todo.md)) |
| Completed tasks, verbatim | `maintenances/<area>/_maintenance/history/` |
| Repository-wide work: CI, `libs/`, labeling, release tooling | [`maintenances/_repo/`](../maintenances/_repo/MAINTAINING.md) |
| Documentation site as a whole: Sphinx build, `conf.py`, API reference | [`docs/tasks/`](../docs/tasks/README.md) |
| Engineering rules, numbered, for every area | [`skills/scikitplot-engineering-rules/`](../skills/scikitplot-engineering-rules/SKILL.md) |
| A fact only one area needs | `maintenances/<area>/todo/lessons.md` |
| Every archive and the areas it names | [`HISTORY_INDEX.md`](../maintenances/_repo/_maintenance/HISTORY_INDEX.md) |

`<area>` mirrors `scikitplot/<area>/` (nested where the package is, e.g.
`_externals/_sphinx_ext/_sphinx_feedback`). Runtime code never imports
`maintenances/` or `skills/`.

## Starting a session

1. This file, then [`skills/scikitplot-engineering-rules/SKILL.md`](../skills/scikitplot-engineering-rules/SKILL.md)
   and the rule files it names for the work.
2. The active `todo/todo.md` of the area you work in.
3. That area's `MAINTAINING.md` and its skill (`skills/INDEX.md`), where present.
4. Past work on the area: `HISTORY_INDEX.md` lists the archives that name
   its packages as paths; for a complete search, grep the archives:
   `grep -rl <name> --include='*.md' maintenances docs/tasks | grep /history/`.

## Writing a record

- **Owner.** A task belongs to the area that owns most of what it changes. A
  campaign across areas (round 33: corpus, mcp, cleanprompt, `_cli` and
  native helpers) lives in its lead area; `HISTORY_INDEX.md` and the grep
  above find it from any of them.
- **Plan and results** go in the owner's `todo/todo.md` (create it if absent).
- **Done tasks** move verbatim to the owner's `_maintenance/history/`, named
  `r<round>_<slug>.md` (`r33_run2_retrieval.md`), with a link left in the
  active file. Active `todo.md` files have a 400-line budget.
- **Moves between files** are a commit of their own, proved lossless before
  any edit: `check_records.py --prove <commit>` (Rule 124).
- **Lessons** follow [`skills/scikitplot-engineering-rules/SKILL.md`](../skills/scikitplot-engineering-rules/SKILL.md).

## Checking

```sh
python -B maintenances/_repo/_maintenance/tools/check_records.py [--write-index]
```

Fails when `tasks/` holds anything but this file, a rule is malformed,
duplicated or missing, an index is stale, an archived part changed, a link
does not resolve, or an active file is over budget. CI runs it on every pull
request (`.github/workflows/pr_check_records.yml`).

## The split

`tasks/todo.md`, `tasks/lessons.md` and `tasks/removed-files.txt` (471e46c)
were split by commit `4R-1`, a pure move: its parts, listed with source lines
and sha256 in
[`records_manifest.json`](../maintenances/_repo/_maintenance/records_manifest.json),
rebuild the three files byte for byte (`check_records.py --prove <4R-1>`).
Later edits are ordinary commits.
