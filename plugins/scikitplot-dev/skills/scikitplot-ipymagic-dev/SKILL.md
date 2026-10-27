---
name: scikitplot-ipymagic-dev
description: Maintain scikitplot._ipymagic (the %scikitplot notebook magics); protect registry-driven routing, result objects over printing, no IPython at import, bounded and killable subprocesses, one profiler per process and a 127.0.0.1-only snakeviz.
metadata:
  scikitplot-mirror: "_ipymagic"
---

# `scikitplot._ipymagic` maintainer

## Read first

1. `maintenances/_ipymagic/MAINTAINING.md` (flow, module map, invariants)
2. `maintenances/_ipymagic/todo/todo.md` (active work, last results)
3. `maintenances/_ipymagic/todo/lessons.md`
4. `plugins/scikitplot-dev/skills/scikitplot-engineering-rules/SKILL.md` and the rules it names
   (subprocess env and decoding, one profiler, logging, numpydoc, testing)

## When changing a magic

- Keep behaviour in the engine (`_bridge`, `_profiling`); `_magics.py` parses
  the line, calls the engine and returns its result. Test the engine without
  IPython, then one shell test for the magic.
- Do not add magic options to `%scikitplot`: anything after the magic name
  belongs to the CLI. Session behaviour goes in `MagicSettings` (validate in
  `__post_init__`, document in `%scikitplot_config` and the guide).
- New magic names go in `MAGIC_NAMES` so `unload` removes them.

## When the CLI changes

- Nothing to change here for a new command, alias, flag or choice: routing,
  completion and suggestions read `scikitplot/_cli/registry.py`. Run the
  completion and routing tests.

## When touching the subprocess runner

- Argument list, no shell; `os.environ` plus the two variables of
  `_child_env`; UTF-8 with `replace`; own process group; stop the group on
  timeout, interrupt and lingering grandchildren; close every pipe.
- Patch `_child_argv` (with `autospec`) in tests to run a stand-in child.

## When touching profiling or snakeviz

- Check the profiler slot before enabling (Rule 133); never stop another
  profiler. Keep row order deterministic.
- snakeviz binds 127.0.0.1, has a startup deadline, is registered for
  `stop_all` (unload, `--stop-snakeviz`, exit). Absent snakeviz raises
  `CapabilityMissingError`; the table never needs it.

## Verify

Commands in `MAINTAINING.md` "Verification"; on IPython 8.18 (Python 3.9)
and the newest.
