# Maintaining `scikitplot._ipymagic`

`scikitplot._ipymagic` is the notebook frontend of scikit-plots: the
`%scikitplot` / `%%scikitplot` magics run any `scikitplot` CLI command from a
notebook and return a `CommandResult`; `%scikitplot_profile` profiles a line
or a cell (table, optional snakeviz); `%scikitplot_config` holds the session
settings. Load with `%load_ext scikitplot` (root `load_ipython_extension`);
`%load_ext scikitplot._magic` is a deprecated shim. Runtime: standard library
only; IPython only in `_magics.py`; snakeviz optional (extra `ipymagic`).
User guide: `docs/source/user_guide/_ipymagic/`. Skill:
[`skills/scikitplot-ipymagic-dev/SKILL.md`](../../skills/scikitplot-ipymagic-dev/SKILL.md).

Start a session with [`tasks/README.md`](../../tasks/README.md), then the active
[`todo/todo.md`](todo/todo.md) and this area's [`todo/lessons.md`](todo/lessons.md).

## Flow

```
%scikitplot LINE [cell]            (_magics: parse nothing, expand {var}/$var by IPython)
   -> split_line (shlex, posix)    MagicUsageError on a broken quote
   -> route (_cli.registry)        unknown command: MagicUsageError + "did you mean"
   -> in-process  (native, no stdin, isolate=false)   logger state saved/restored
      subprocess  (delegated | stdin | isolate)       own process group, streamed,
                                                      bounded capture, timeout/interrupt
   -> CommandResult (exit_code, stdout, stderr, data from --format json, duration)
   -> CommandFailedError only when raise_on_error

%scikitplot_profile                (_magics -> _profiling.profile_code)
   -> refuse when a profiler is active (Rule 133)
   -> cProfile, rows (deterministic order), stats file
   -> optional _snakeviz (127.0.0.1, readiness deadline, stopped at unload/exit)
```

## Module map

| Module | Owns |
|---|---|
| `_errors.py` | `IpyMagicError` (`code`, `hint`, two-line `_render_traceback_`) and subclasses deriving from the built-in a caller expects |
| `_results.py` | `CommandResult`, `ProfileRow`, `ProfileResult` (frozen values, `to_dict`, rich reprs, escaped HTML) |
| `_settings.py` | `MagicSettings` (validated; `updated("k=v ...")` all-or-nothing) |
| `_bridge.py` | the engine: `split_line`, `route`, `wants_json`, `run_command`, in-process isolation, the subprocess runner |
| `_completion.py` | `complete(line)` from `CommandSpec` / `Param` only |
| `_profiling.py` | `profiler_busy_reason`, `rows_of`, `profile_code` |
| `_snakeviz.py` | `start_snakeviz`, `SnakevizServer`, `stop_all` |
| `_magics.py` | the only IPython import: `ScikitplotMagics`, `load`, `unload` |
| `__init__.py` | lazy public API, `load_ipython_extension`, `unload_ipython_extension` |

## Invariants

1. Importing `scikitplot` or `scikitplot._ipymagic` imports no IPython.
2. A magic always returns a result object; it raises only for a usage error,
   with `raise_on_error`, or to re-raise the profiled code's own error.
3. A mistyped command is refused with suggestions, never run.
4. Magic settings live in `%scikitplot_config`, never as magic options, so no
   magic option can shadow a CLI option.
5. In-process runs restore the `scikitplot` logger and `_cli.logging` state.
6. A subprocess runs `[sys.executable, "-m", "scikitplot", ...]` with no shell,
   from `os.environ` (Rule 77), decoded UTF-8 with `replace` (Rule 45); at
   timeout, interrupt or unload its whole process group is stopped.
7. Captured output is bounded (`MAX_CAPTURE`); `truncated` says so.
8. One profiler per process (Rule 133): checked before enabling.
9. snakeviz listens on 127.0.0.1 only; startup has a deadline.

## Verification

```
python -m pytest scikitplot/_ipymagic docs/source/user_guide/_ipymagic -q
python -m pytest scikitplot/_cli -q          # the registry the magics read
python tools/maint_tools/generate_pr_labeler.py check
python -B maintenances/_repo/_maintenance/tools/check_records.py
```

The real-kernel test needs `nbclient`, `nbformat`, `ipykernel`; the snakeviz
server test needs `snakeviz`. Both skip when absent. Run the suite on the
oldest supported IPython (8.18, Python 3.9) and the newest.

## When the CLI changes

Nothing to edit here: routing, aliases, delegated commands, flags and choices
are read from `scikitplot/_cli/registry.py`. Add a command there and the
magic, its completion and the "did you mean" hint follow. If a command must
always run in a subprocess, make it delegated in the registry.
