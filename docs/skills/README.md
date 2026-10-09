# Scikit-Plots skills entry point

`skills/` contains operational instructions for maintaining specific
Scikit-Plots areas. This document explains how to select and use those skills
without relying on a previous chat or a memorized repository map.

The authoritative skill content is `skills/<area>/SKILL.md`. Files under
`docs/skills/` are navigation and repository-wide policy only; do not duplicate
module-specific instructions here.

## Select a skill by ownership, not by keyword

Start from the files the task actually touches and identify their maintenance
owner.

Examples:

```text
scikitplot/annoy/...        -> maintenances/annoy/...        -> skills/annoy/SKILL.md
scikitplot/cleanprompt/...  -> maintenances/cleanprompt/...  -> skills/cleanprompt/SKILL.md
scikitplot/mcp/...          -> maintenances/mcp/...          -> skills/mcp/SKILL.md
```

Nested maintenance domains may have nested skills. Follow the repository path
and the owning `MAINTAINING.md`; do not collapse two compiled/native or adapter
subsystems merely because their names are related.

A skill is an execution guide, not evidence that the runtime currently works.
Verify its current claims against this checkout.

## Fresh-session workflow

1. Read `docs/maintenances/MAINTAINING.md`.
2. Read the relevant `maintenances/<area>/MAINTAINING.md`.
3. Read the current files that entry point names (`STATE.json`, `EVIDENCE.json`,
   `VERIFICATION.md`, or a fresh-chat handoff) when present.
4. Read the matching `skills/<area>/SKILL.md` when present.
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
