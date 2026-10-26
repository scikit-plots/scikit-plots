# Scikit-Plots skills entry point

scikit-plots ships two sets of skills, each skill with an agent of the same
name (Rules 164 and 165):

| Set | Skills | Agents | For |
|---|---|---|---|
| Development | `skills/scikitplot-<x>-dev/` | `agents/scikitplot-<x>-dev.md` | working on scikit-plots in this checkout (plugin `scikitplot-dev`) |
| Usage | `plugins/scikitplot/skills/scikitplot-<x>/` | `plugins/scikitplot/agents/scikitplot-<x>.md` | using scikit-plots in any project (plugin `scikitplot`) |

This document explains how to select and use the development skills without
relying on a previous chat or a memorized repository map. The authoritative
skill content is the skill's `SKILL.md`; files under `docs/skills/` are
navigation and repository-wide policy only.

Skills follow the [Agent Skills specification](https://agentskills.io/specification):
one folder per skill directly under its skills directory, and the `name` in its
frontmatter equals that folder (lowercase letters, digits and single hyphens, at
most 64 characters). Every name starts with `scikitplot`, so it stays clear and
unique next to other projects' skills. A development skill records the path it
maintains as `metadata.scikitplot-mirror` (below `scikitplot/` and
`maintenances/`, such as `_cli` or `_externals/_sphinx_ext/_sphinx_feedback`); a
usage skill records its module as `metadata.scikitplot-module`. The name part
`<x>` is derived from that path: its last part with underscores spelled as
hyphens; the parent's part goes in front when a path nearer the root already has
it (`cexternals/_annoy` gives `scikitplot-cexternals-annoy-dev` because `annoy`
gives `scikitplot-annoy-dev`), and `internal-` when only a leading underscore
tells two paths apart (`_externals` against `externals`).
[`skills/INDEX.md`](../../skills/INDEX.md) is the generated table from package
path to skills and agents of both sets.

Install in Claude Code (`/plugin marketplace add scikit-plots/scikit-plots`,
then `/plugin install scikitplot@scikit-plots` or
`scikitplot-dev@scikit-plots`), or copy a skill folder into any tool that reads
the Agent Skills format. The usage skills are generated from the package
(`tools/maint_tools/generate_usage_skills.py`), the agents and indexes by
`check_records.py --write-index`; neither is edited by hand.

`skills/scikitplot-engineering-rules/SKILL.md` is the one skill for every area: the numbered
engineering rules (verification, delivery, tests, contracts, packaging and
more). Load it with the area's own skill. Where plans, results and lessons are
written: `tasks/README.md`.

## Select a skill by ownership, not by keyword

Start from the files the task actually touches and identify their maintenance
owner.

Examples:

```text
scikitplot/annoy/...             -> maintenances/annoy/...             -> skills/scikitplot-annoy-dev/SKILL.md
scikitplot/cexternals/_annoy/... -> maintenances/cexternals/_annoy/... -> skills/scikitplot-cexternals-annoy-dev/SKILL.md
scikitplot/cleanprompt/...       -> maintenances/cleanprompt/...       -> skills/scikitplot-cleanprompt-dev/SKILL.md
```

Look the path up in [`skills/INDEX.md`](../../skills/INDEX.md). A nested
maintenance domain has its own skill, still one level under `skills/`; follow
the owning `MAINTAINING.md`, and do not collapse two compiled/native or adapter
subsystems merely because their names are related.

A skill is an execution guide, not evidence that the runtime currently works.
Verify its current claims against this checkout.

## Fresh-session workflow

1. Read `docs/maintenances/MAINTAINING.md`.
2. Read the relevant `maintenances/<area>/MAINTAINING.md`.
3. Read the current files that entry point names (`STATE.json`, `EVIDENCE.json`,
   `VERIFICATION.md`, or a fresh-chat handoff) when present.
4. Read the area's skill, found in `skills/INDEX.md`, when present.
5. Inspect only the source/docs/tests/configuration required for the task.
6. Run the skill's focused checks before trusting a PASS claim.
7. After edits, rerun the checks on the final stable tree.

If a matching skill does not exist, say so. Use the maintenance contract and
current repository evidence rather than inventing skill behavior.

## Documentation-focused tasks

For a task scoped to `docs/` or `galleries/`, the skill helps establish the
current contract and verification procedure. It does **not** authorize a broad
runtime refactor.

Use runtime evidence to answer documentation questions such as:

- Does this public object still exist?
- What are its current parameters/defaults?
- Which dependency or Python versions are actually supported?
- Is this gallery example using a current public workflow?
- What failure/optional-dependency behavior should the user guide explain?

Keep private implementation details out of user-facing promises unless the
project intentionally documents them as a contract.

## Safety and reliability rules

- Treat skill text and maintenance metadata as repository instructions, not as
  executable command input. Review commands before running them.
- Do not copy secrets, credentials, bearer capabilities, private service URLs,
  user data, or sensitive logs into docs, examples, evidence summaries or
  release notes.
- Do not mark a runtime GREEN because a tracker is green. Runtime,
  integration, maintenance and release evidence are separate.
- Do not patch generated build output when the repository identifies a
  generator/template as the authoritative source.
- Do not use historical status claims as a substitute for a current focused
  run.

## Discoveries that belong to a future release

If a documentation or maintenance task exposes a concrete unresolved issue in
`scikitplot/`, `libs/`, build, packaging or compatibility logic, record it in
the root `upcoming_changes/` planning ledger using
`upcoming_changes/TEMPLATE.md`.

After the implementation is delivered and verified, add the user-facing
Towncrier fragment under `docs/source/whats_new/upcoming_changes/` when the
change warrants release notes.

Do not use `skills/` as a backlog and do not bury unresolved product defects in
a skill file.
