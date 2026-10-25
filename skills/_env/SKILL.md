---
name: env-builder-maintainer
description: Maintain scikitplot._env (scikitplot env); protect plan-before-change, approvals-first, config-cannot-approve, rule-based backend choice, offline guarantees, one parse of environment.yml for all backends, state-digest idempotence and pinned, hash-checked bootstraps.
---

# `scikitplot._env` maintainer

## Read first

1. `maintenances/_env/MAINTAINING.md` (flow, module map, invariants)
2. `maintenances/_env/todo/todo.md` (active work, last results)
3. `maintenances/_env/todo/lessons.md`
4. `skills/_engineering/SKILL.md` and the rules it names (contracts, testing, logging, numpydoc)

## When adding a backend

- Subclass `_backends/_base.Backend` in its own module; return `Step`s from
  `create`/`update`, never run anything. Add one line to `BACKENDS` and, if it
  should be chosen automatically, a place in `PREFERENCE` and a planner rule
  documented in `_core/_planner.py` and `how_it_works.rst`.
- Check every flag against the tool's `--help` and say where in the module
  notes; a backend that cannot be executed in the tests is plan-tested only and
  says so in its `summary`.
- Removal is the shared `remove_tree` action; do not add a tool-specific one.

## When adding a target

- Add a `Target` (data only) to `_targets`; the planner must not branch on
  its name. Decide `backends`, `runs_here`, `pypi`, `relocate_prefix`,
  `channels`, `executable`.
- A target that cannot be built is refused with `TargetUnsupportedError`
  and a hint, never approximated.

## When changing sources or configuration

- Parse dependency files in `_sources/_readers.py` only, record every file read
  in `inputs`, refuse what no backend can apply.
- New request fields: `_core/_request.py`, `_config/_schema.py` (`_TOP`,
  `_normalise`), the CLI option, the guide's TOML example (its test loads it).
- Never accept an approval-like key in a document.

## When touching the bootstrap

- Pins live in `_bootstrap/__init__.py`; see "Moving the micromamba pin" in
  `MAINTAINING.md`. Downloads stay https-only with a size limit and a hash check
  before extraction.
