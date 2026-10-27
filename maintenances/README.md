# Where everything lives

The repository has two homes for everything that is not the library itself,
and each is split into groups. This file is the only map.

| Home | Holds | Groups |
|---|---|---|
| `maintenances/` | How the project is worked on: plans, results, history, procedure | one folder per area: `<area>/` mirrors `scikitplot/<area>/`; `_repo/` (CI, `libs/`, labeling, release tooling, the records tooling), `_docs/` (the documentation site), `_plugins/` (skills, agents, plugins and their release) |
| `plugins/` | What ships to Claude: skills, agents, MCP | one folder per plugin: `scikitplot/` (users), `scikitplot-mcp/` (users, opt-in MCP server), `scikitplot-dev/` (developers); `_shared/skills/` (skills every plugin carries) |

Nothing else at the root holds records, skills or agents. The retired
locations (`tasks/`, `docs/tasks/`, `docs/maintenances/`, `docs/skills/`,
root `skills/` and `agents/`, `plugins/maintenances/`) fail the checker below
if anything reappears in them. The repository-wide procedure (read order,
evidence) is [`MAINTAINING.md`](MAINTAINING.md); the plugins are mapped in
[`plugins/README.md`](../plugins/README.md) (users) and
[`_plugins/README.md`](_plugins/README.md) (maintainers).

**Instructions that name `tasks/todo.md` or `tasks/lessons.md`** (older
handoffs, standing preferences) mean, today: plans and results go to the
owner's active `todo/todo.md`; lessons go to the engineering rules, as their
[`SKILL.md`](../plugins/scikitplot-dev/skills/scikitplot-engineering-rules/SKILL.md)
describes.

| Record | Location |
|---|---|
| Active plan and results of an area | `maintenances/<area>/todo/todo.md` (e.g. [corpus](corpus/todo/todo.md)) |
| Completed tasks, verbatim | `maintenances/<area>/_maintenance/history/` |
| Repository-wide work: CI, `libs/`, labeling, release tooling | [`_repo/`](_repo/MAINTAINING.md) |
| Documentation site as a whole: Sphinx build, `conf.py`, API reference | [`_docs/`](_docs/MAINTAINING.md) |
| Skills, agents, plugins and their release channels | [`_plugins/`](_plugins/MAINTAINING.md) |
| Engineering rules, numbered, for every area | [`scikitplot-engineering-rules`](../plugins/scikitplot-dev/skills/scikitplot-engineering-rules/SKILL.md) |
| A fact only one area needs | `maintenances/<area>/todo/lessons.md` |
| Every archive and the areas it names | [`HISTORY_INDEX.md`](_repo/_maintenance/HISTORY_INDEX.md) |
| Every skill and agent, by package | [`plugins/INDEX.md`](../plugins/INDEX.md) |

`<area>` mirrors `scikitplot/<area>/` (nested where the package is, e.g.
`_externals/_sphinx_ext/_sphinx_feedback`); a leading `_` with no package
(`_repo`, `_docs`, `_plugins`) is a repository-wide area. Runtime code never
imports `maintenances/` or `plugins/`.

## Starting a session

1. This file, then the engineering rules'
   [`SKILL.md`](../plugins/scikitplot-dev/skills/scikitplot-engineering-rules/SKILL.md)
   and the rule files it names for the work.
2. The active `todo/todo.md` of the area you work in.
3. That area's `MAINTAINING.md` and its skill ([`plugins/INDEX.md`](../plugins/INDEX.md)), where present.
4. Past work on the area: `HISTORY_INDEX.md` lists the archives that name
   its packages as paths; for a complete search, grep the archives:
   `grep -rl <name> --include='*.md' maintenances | grep /history/`.

## Writing a record

- **Owner.** A task belongs to the area that owns most of what it changes; a
  campaign across areas lives in its lead area.
- **Plan and results** go in the owner's `todo/todo.md` (create it if absent).
- **Done tasks** move verbatim to the owner's `_maintenance/history/`, with a
  link left in the active file. Active `todo.md` files have a 400-line budget.
- **Moves between files** are a commit of their own, proved lossless before
  any edit: `check_records.py --prove <commit>` (Rule 124).

## Checking

```sh
python -B maintenances/_repo/_maintenance/tools/check_records.py [--write-index]
```

Fails when a retired location holds a file, a rule is malformed, duplicated
or missing, an index or generated file is stale, a shared skill's copy differs
from its source, an archived part changed, a link does not resolve, an active
file is over budget, or the marketplace and the plugins disagree. CI runs it
(`.github/workflows/pr_check_records.yml`).
