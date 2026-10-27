# Active Tasks

## Task: IPM-1, the notebook frontend of scikit-plots (2026-10-26, PR 870)

### Context
- Goal: `scikitplot._ipymagic` lets a notebook (Jupyter, JupyterLab, VS Code,
  Colab, terminal IPython) use every `scikitplot` command and profile code,
  with results as Python objects, errors that read like the CLI's, and no
  side effects on the kernel.
- Rationale: the package held two unrelated drafts. `_ipymagic/_ipymagic.py`
  is a verbatim copy of snakeviz's magic (server on `0.0.0.0`, an unbounded
  `readline` loop that hangs the kernel when the server fails, `sleep(3)`
  then `terminate`, a debugging `print(display)`, a `str.format` call that
  raises `KeyError` in its own error path). `_magic.py` (`%scikitplot`)
  imports matplotlib and numpy at load, creates `~/.scikitplot` on import,
  auto-executes a fuzzy-matched command (a typo can run another command),
  shells out to `scikitplot <cmd>` for every unknown word, plots from names
  that do not exist (`scikitplot.metrics`), and drives the classic-notebook
  toolbar API that Notebook 7 removed. Neither is imported anywhere.
- Affected systems: new `scikitplot/_ipymagic/`; `scikitplot/__init__.py`
  (`%load_ext scikitplot`); `scikitplot/_magic.py` (deprecation shim);
  `pyproject.toml` (`ipymagic` extra, Towncrier section); labeler; guide;
  maintenance plane; dev skill (agent generated).

### Big picture

```
%scikitplot ARGS / %%scikitplot ARGS (cell = stdin)      run_command(ARGS)  (plain Python, agents)
        \                                                  /
         +------------- _bridge: one engine ----------------+
            split -> route by the CLI registry -> run -> CommandResult
              native command, no stdin  -> in process, isolated (streams captured,
                                           `scikitplot` logger state restored)
              delegated command / stdin -> subprocess `python -m scikitplot`,
                                           streamed line by line, interruptible,
                                           optional timeout
            --format json -> parsed into result.data, shown as JSON
%scikitplot_profile / %%scikitplot_profile -> _profiling (guarded cProfile)
            -> ProfileResult (HTML / text table, .prof file) -> optional snakeviz
               server on 127.0.0.1 with a managed lifetime
%scikitplot_config key=value -> MagicSettings (validated, immutable)
completion: command names, aliases, native options from the neutral Param model
```

### Inputs, outputs, invariants
1. The magic is a projection of the CLI's neutral model (`registry`, `Param`):
   it never redeclares a command, option or exit code.
2. Running a command never changes the kernel: the `scikitplot` logger's
   level, handlers and propagation, and `_cli.logging._HANDLER`, are restored;
   `sys.stdout`/`sys.stderr` are restored; delegated commands, which own
   their parsers and may block (servers), never run in the kernel.
3. A command that is not a registered name is refused with suggestions; a
   guess is never executed.
4. Output is shown as it is produced (subprocess lines), and stdout of a
   `--format json` run is parsed, not streamed; captured text is bounded.
5. An interrupt or timeout stops the child (terminate, then kill) and is
   reported in the result; no orphan process.
6. One profiler per process (Rule 133): the profile magic refuses when
   `sys.getprofile()` is set or the `sys.monitoring` profiler slot is taken.
7. Every name shown in HTML is escaped; the snakeviz server listens on
   127.0.0.1 only, readiness has a deadline, servers are stopped on unload
   and at exit.
8. Importing `scikitplot._ipymagic` imports no IPython and no third-party
   module; IPython is needed only to load the extension.
9. Errors are `IpyMagicError` subclasses with a code and a hint, rendered by
   IPython in two lines (`_render_traceback_`), not as a traceback.

### Failure modes
- IPython missing -> `load_ipython_extension` is never reachable without it;
  `run_command` works without it.
- Child cannot start -> result with exit 127 and the OS error.
- Output not UTF-8 -> decoded with `errors="replace"` (Rule 45).
- JSON requested but stdout is not JSON (error path) -> text shown, `data`
  stays None.

### Layout
| Module | Responsibility |
|---|---|
| `__init__.py` | lazy public names; `load/unload_ipython_extension` |
| `_errors.py` | error types (code, hint, `_render_traceback_`) |
| `_results.py` | `CommandResult`, `ProfileRow`, `ProfileResult` (value objects, rendering) |
| `_settings.py` | `MagicSettings` (validated, immutable, parsed from `key=value`) |
| `_bridge.py` | split, route, in-process and subprocess runners, `run_command` |
| `_completion.py` | completions from the registry |
| `_profiling.py` | guarded cProfile, stats rows |
| `_snakeviz.py` | optional snakeviz server lifecycle |
| `_magics.py` | the IPython `Magics` class (thin) |

### Implementation steps
- [x] 1. Plan (this section).
- [x] 2. `_errors`, `_results`, `_settings`.
- [x] 3. `_bridge` (+ `run_command`), `_completion`.
- [x] 4. `_profiling`, `_snakeviz`.
- [x] 5. `_magics`, `__init__`, root `load_ipython_extension`, `_magic.py` shim.
- [x] 6. Tests: unit, adversarial, IPython shell (`globalipapp`), a real
      kernel executing a notebook (nbclient), snakeviz real server; floors.
- [x] 7. Guide, Towncrier 870, extra, labeler, maintenance plane, dev skill
      and generated agent, independent review (Rule 92).

### Acceptance criteria
- [x] `%scikitplot info --format json` returns a result whose `.data` equals
      the CLI's JSON, and the `scikitplot` logger is unchanged afterwards.
- [x] `%scikitplot env plan ...` streams, returns exit status 0; a failing
      command shows the CLI's error and (with `raise_on_error`) raises
      `CommandFailedError`, rendered in two lines.
- [x] `%scikitplot evn` is refused with "did you mean 'env'".
- [x] `%%scikitplot cleanprompt ...` reads the cell as stdin.
- [x] Interrupt and timeout leave no child process.
- [x] `%%scikitplot_profile` shows a table, refuses under another profiler,
      `--snakeviz` serves on 127.0.0.1 and stops on unload.
- [x] A real kernel runs a notebook using every magic.

### Risks and mitigations
- In-process capture with `redirect_stdout` also captures other threads'
  prints during the run -> native commands are short; documented; `isolate`
  setting runs everything in a subprocess.
- Child imports a different `scikitplot` than the kernel (sys.path edits in
  the kernel) -> the result records the child's interpreter; documented.
- IPython API drift -> only documented APIs (`Magics`, `magic_arguments`,
  `set_hook('complete_command')`, `_repr_mimebundle_`,
  `_render_traceback_`), tested on the floor (8.18, Python 3.9) and 9.17.

## Results Review - 2026-10-26

### Implementation summary
- New: `scikitplot/_ipymagic/` (9 modules, 4 test modules), guide
  `docs/source/user_guide/_ipymagic/` (4 pages + sync test), Towncrier
  `870.feature.rst` / `870.fix.rst`, extra `ipymagic`, labeler entry,
  `maintenances/_ipymagic/`, skill `scikitplot-ipymagic-dev` + generated agent.
- Changed: `scikitplot/__init__.py` (`load_ipython_extension`,
  `unload_ipython_extension`), `scikitplot/_magic.py` (shim, `FutureWarning`),
  `pyproject.toml`, `docs/source/user_guide/index.rst`.
- Removed: `scikitplot/_ipymagic/_ipymagic.py` (copy of snakeviz's magic).

### Verification evidence
- Source tree, Python 3.13 / IPython 9.17.1: 91 passed (+1 expected
  `captured_logger` error with `--noconftest`), bridge tests 3x stable.
- Installed mode with the repo `pytest.ini` (warnings are errors) and
  conftest: `scikitplot._ipymagic` + `scikitplot._cli` 270 passed; guide 3 passed.
- Python 3.9 / IPython 8.18.1 (the floor): 85 passed, 3 skipped.
- Real Jupyter kernel (nbclient) runs every magic; real snakeviz server
  answers 200 on 127.0.0.1.
- ruff, black, codespell clean; numpydoc lint (GL06 GL07 PR01-04 PR10 RT01
  YD01) clean; Sphinx `-W -n` clean; labeler check PASS; towncrier check
  PASS; `check_records` ok rules/links/budgets/skills.

### Independent review (Rule 92)
10 findings, all fixed with regression tests: lingering/escaped processes
hung the cell and a finished command was reported timed out (L6); combined
or abbreviated root flags ran delegated commands in the kernel (L7); Ctrl-C
on an in-process command was swallowed (L8); `interrupted`/`EXIT_STOPPED`
documented but never produced (removed); display cut despite "never cut"
(in-process shown in full, a note after cut output); `--snakeviz` ran the
code before checking snakeviz and lost the profile on failure; quoted `-D`
kept its quotes; names of the extension did not share load/unload; the
deprecation warning was invisible (L9); labeler regenerated, POSIX-only
tests marked, loopback test bypasses proxies. A stats file that cannot be
written now keeps the profile (`stats_file` error) instead of replacing it.

### Deviations from plan
- `_ipymagic` is not added to the skinny distribution (follow-up).

### Technical debt
- Windows: lingering processes of an exited child cannot be found
  (`taskkill /T` needs the live parent); documented in `_stop`.
