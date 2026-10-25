# Lessons: `scikitplot._env`

## Active Rules

### Rule ENV-L1: Argparse - options shared through `parents` must not use `set_defaults`
- **When:** an option is accepted both before and after a subcommand
  (`parents=[common]` on the main parser and on every subparser).
- **Then:** give it `default=argparse.SUPPRESS` and fill the default after
  parsing; never `parser.set_defaults(...)` for it.
- **Verified by:** `test_plan_text_and_json` (`--format json` before and after the command).
- **Added:** 2026-10-24. Root cause: `set_defaults` changes `action.default`
  of the action objects, which `parents` shares with every subparser, so the
  subparser reset a value given before the command (`--format json plan` printed text).

### Rule ENV-L2: Idempotence - the state digest excludes how the builder may behave
- **When:** deciding whether an existing environment is up to date.
- **Then:** compare a digest of what the environment contains (packages,
  sources and their hashes, Python, target, backend, prefix, conversion);
  leave out interaction, network and bootstrap policies and the preset name.
- **Verified by:** `test_up_to_date_and_update` (an offline, never-asking re-plan is up to date).
- **Added:** 2026-10-24. Root cause: the first receipt compared the whole
  request digest; a second run with `--offline` or without `--bootstrap` saw
  a "different" request and reinstalled.

### Rule ENV-L3: Cross-builds - relocation is a property of the target, not of cross-building
- **When:** adding a target that is built for another platform.
- **Then:** set `Target.relocate_prefix` only when the runtime mounts the
  environment elsewhere (the browser mounts it at `/`); `--no-pyc` for every
  cross-build.
- **Verified by:** `test_cross_build` (no `--relocate-prefix`) and `test_browser_build`.
- **Added:** 2026-10-24. Root cause: the original Xeus code always passed
  `--relocate-prefix ""`; reused for `linux-aarch64` it would have rewritten
  every prefix path to `/`, so the copied environment could not run.

### Rule ENV-L4: Success of every step is not success of the build
- **When:** a plan's steps all exit 0.
- **Then:** check the post-condition (prefix exists, planned Python exists)
  before writing a receipt; fail with `ExecutionError` otherwise.
- **Verified by:** the executor tests; `_check_built`.
- **Added:** 2026-10-24. Root cause: a fake runner that created nothing made
  the receipt write fail with a raw `FileNotFoundError` from `tempfile`.

### Rule ENV-L5: Recognition - an environment is proven by markers only it has
- **When:** deciding whether a directory may be written into or deleted.
- **Then:** require `pyvenv.cfg`, `conda-meta` or a receipt (`ENV_MARKERS`);
  never infer it from an interpreter path (`/usr/bin/python` exists too).
- **Verified by:** `test_plain_python_install_is_not_an_environment`.
- **Added:** 2026-10-25. Root cause: the first recognition accepted `bin/python`,
  so `plan -p /usr` planned a pip install into the system Python.

### Rule ENV-L6: Authority - a value a file can set is not an approval
- **When:** a component is configured by files only (the JupyterLite add-on's
  traitlets, presets, `pyproject.toml`).
- **Then:** read approvals from the invocation (CLI, API argument, terminal,
  `SKPLT_ENV_APPROVE`); a `config=True` trait must never feed a `Decider`.
- **Verified by:** `test_approval_comes_from_the_invocation_only`.
- **Added:** 2026-10-25. Root cause: `XeusAddon.bootstrap` was a configurable
  trait turned into `approve=["bootstrap"]`, so a committed
  `jupyter_lite_config.json` could approve a download.

### Rule ENV-L7: Every lane of a policy - offline and indexes reach every installer
- **When:** a policy or index setting is added, or a new installer path (an
  action rather than a command) is added.
- **Then:** pass `pip_index_args` to it and test it under `offline`.
- **Verified by:** `test_browser_pip_layer_gets_index_and_offline`, `test_index_args_reach_pip`.
- **Added:** 2026-10-25. Root cause: the browser pip layer was an action with its
  own argv and silently ignored `--no-index` and the request's index.

## Pattern Analysis
- Pattern: behaviour copied from one context into a more general one
  (relocation from the browser lane, request digest as state digest), and a
  side path that skips a rule the main path enforces (the add-on's approval,
  the browser pip layer's index policy).
- Occurrences: 4.
- Root Cause: a value meant for one case placed where every case reads it.

## Effectiveness Metrics
- Total lessons: 7
- Repeat occurrences: 0
- Trend: Stable
