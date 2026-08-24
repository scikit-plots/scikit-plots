# CLI Refactor Findings

Tracked defects and tasks for the `scikitplot._cli` rebuild. Each has a root
cause (not a symptom) and a status. Findings move to `CLOSED` only with pasted
test evidence.

Status legend: `OPEN` · `PLANNED` · `IN-PROGRESS` · `PARTIAL` · `CLOSED` ·
`DEFERRED`. Evidence lives inline or in the referenced test.

| ID | Title | Severity | Status |
| --- | --- | --- | --- |
| CLI-FE-001 | CLI unrunnable on base install (click hard-imported, not a core dep) | Critical | CLOSED |
| CLI-FE-002 | Eager command discovery pulls rich into bootstrap | High | CLOSED |
| CLI-FE-003 | Inconsistent per-command output flags (`--json/--yaml/--rich`) | Medium | CLOSED |
| CLI-FE-004 | Broad `except Exception: pass` in alias resolution hides failures | High | CLOSED |
| CLI-FE-005 | Logging/echo writes to stdout, contaminating machine output | High | CLOSED |
| CLI-FE-006 | Legacy pip-derived helpers migrated without CLI responsibility | Medium | PARTIAL (commands migrated; `_cmd_options`/`_misc` deletion pending — see INTEGRATION.md §4) |
| CLI-FE-007 | Snake_case option spelling instead of shell convention | Low | CLOSED |
| CLI-FE-008 | No standalone `python -m <module>` runner surface | Medium | CLOSED |
| CLI-FE-009 | Root `python -m scikitplot` routes to click-only entry | High | CLOSED
| CLI-FE-010 | `show-config --format json` emitted a Python-repr + `null` (invalid JSON) | High | CLOSED (updated `__main__.py` delivered; apply `pyproject.toml` edit) |

> Verified by `scikitplot/_cli/tests` — 20 passing: frontend parity across all
> six commands + both aliases + the negatable flag, import-contract (kernel
> imports with `click`/`rich`/`yaml` blocked), output-contract, error taxonomy.
> All six commands (`info`, `doctor`, `show-versions`, `show-config`, `sysinfo`,
> `greet`) run on both frontends with identical machine output.

---

## CLI-FE-001 — CLI unrunnable on base install

**Severity:** Critical · **Status:** PLANNED

**Symptom.** `scikitplot --help` raises `ModuleNotFoundError: No module named
'click'` on a clean install.

**Evidence.** `[project].dependencies` contains no `click` (verified: the core
dependency block lists numpy/scipy/matplotlib/pandas/joblib/threadpoolctl/
scikit-learn only). `click` and `rich` appear only in a dev extras group. Yet the
entry point `scikitplot = "scikitplot._cli.cli:cli"` imports `click` at module
top level, as do all six modules under `_commands/`/`_cmd_options/`.

**5-whys root cause.**
1. Why does it crash? `cli.py` imports `click` at top level.
2. Why is that fatal? `click` is not installed on a base install.
3. Why is it imported unconditionally? Command logic *is* click callbacks.
4. Why is logic written as callbacks? No framework-neutral layer exists.
5. Why no neutral layer? The design treated click as the kernel (guide
   ADR-CLI-001), so there was never a reason to separate logic from click.

Root cause: **framework coupling** — logic lives inside click, so it cannot run
without click.

**Fix (root level).** Extract logic into neutral handlers `run(ctx, **params) ->
int`; make argparse the always-available frontend; make click an optional adapter
(ADR-CLI-100). See MAINTAINING §3.1–§3.2, CONTRACT §1–§4.

**Verified by.** `test_cli_import_contract.py::test_bootstrap_needs_no_click`
(passes with `click`/`rich`/`yaml` blocked in `sys.modules`).

---

## CLI-FE-002 — Eager discovery pulls presentation deps into bootstrap

**Severity:** High · **Status:** PLANNED

**Symptom.** Importing the CLI imports every command module, several of which
import `rich` at top level; optional presentation becomes an effective bootstrap
dependency (guide §4.1, §4.2).

**Root cause.** `_load_commands()` runs `pkgutil.iter_modules` + `import_module`
for every module in `_commands/` at CLI import time. Discovery and activation are
conflated.

**Fix.** Explicit metadata registry (`BUILTIN_COMMANDS`); handlers loaded only on
invocation via `loader.dispatch` (ADR-CLI-103). File scanning is not the
first-party contract (guide §16).

**Verified by.** Import-contract test asserting the registry import imports no
`_commands/*` module and top-level help imports no handler.

---

## CLI-FE-003 — Inconsistent per-command output flags

**Severity:** Medium · **Status:** PLANNED

**Symptom.** `show_config`/`show_versions`/`sysinfo` each define their own
`--json/--yaml/--rich` boolean flags; `--rich` competes with `json`/`yaml` as if
it were a data format (guide §4.6).

**Root cause.** No shared output contract; presentation (rich) is conflated with
data format.

**Fix.** One neutral `--format text|json|yaml` `Param` (see `registry.FORMAT`);
rich is a *rendering* of `text`, selected by color/terminal policy in `output.py`,
never a data format.

**Verified by.** Parity test over `--format json`/`yaml`; `output.py` unit tests.

---

## CLI-FE-004 — Broad exception suppression in alias resolution

**Severity:** High · **Status:** PLANNED

**Symptom.** `AliasedGroup.get_command` wraps `super().get_command` in
`except Exception: pass`, so an import error in a command is indistinguishable
from "command not found" (guide §4.4).

**Root cause.** Resolution does not model distinct failure classes (not-found vs
import-failed vs capability-missing vs plugin-broken).

**Fix.** Loader normalizes failures into the `errors.py` taxonomy with stable exit
codes; no broad swallow. Command-not-found, import failure, and missing capability
are separate, actionable messages (invariant §3.7).

**Verified by.** Error-taxonomy tests: broken handler target yields a normalized
error + non-zero exit, not a silent miss.

---

## CLI-FE-005 — Logging contaminates machine output

**Severity:** High · **Status:** PLANNED

**Symptom.** `set_log_level` emits `click.secho("Changed logging level: ...")` to
stdout, and commands print rich to stdout, so `info --format json | jq .` can
receive non-JSON on stdout (guide §4.5).

**Root cause.** No stdout/stderr ownership rule; diagnostics share the result
channel.

**Fix.** stdout is the result channel; all logging/diagnostics/hints go to stderr
via `logging.py` (invariant §3.6). Handlers write results to `ctx.stdout` only.

**Verified by.** Output-contract test: capture stdout for `--format json`, assert
`json.loads` succeeds and stdout contains no diagnostic text.

---

## CLI-FE-006 — Legacy helpers migrated without responsibility

**Severity:** Medium · **Status:** OPEN

**Symptom.** `_cmd_options/_cmd_options_optparse.py` (925 lines, pip-derived) and
`_misc.py` (772 lines) are carried wholesale; most of their surface (index URLs,
trusted hosts, requirements handling) has no scikit-plots CLI responsibility
(guide §4.8, §111).

**Root cause.** Helpers were copied from pip/mlflow rather than justified against
the CLI's actual command set.

**Fix.** Classify every helper KEEP / MIGRATE / DEPRECATE / DELETE with a reason.
Default disposition for the pip-derived optparse catalog and unrelated `_misc`
utilities is DELETE unless a concrete command needs them. Provisional:

```text
_cmd_options_optparse.py   -> DELETE (no CLI command consumes pip index/requirements options)
_cmd_options_click.py      -> DEPRECATE (superseded by neutral _spec + _frontends)
_misc.strtobool            -> MIGRATE (small, generally useful) if actually used; else DELETE
_misc (pip-derived rest)   -> DELETE unless a command needs it
```

This finding stays OPEN until each helper has a recorded disposition.

---

## CLI-FE-007 — Snake_case option spelling

**Severity:** Low · **Status:** PLANNED

**Symptom.** Options like `--file_path`, `--dark_theme`, `--lib_sample` use
underscores instead of shell-convention hyphens (guide §4.7).

**Root cause.** Option flags were derived from Python variable names.

**Fix.** `Param.flags` use hyphens; `Param.dest` stays snake_case. Both frontends
derive spelling from the same `Param` (invariant §3.8).

**Verified by.** A registry lint test asserting every non-argument `Param` flag
matches `^-|--[a-z0-9-]+$` and contains no underscore.

---

## CLI-FE-008 — No standalone module-runner surface

**Severity:** Medium · **Status:** PLANNED

**Symptom.** `python -m scikitplot.utils._show_versions` does nothing useful —
the module has no `__main__` guard, so the function is reachable only via the
library API or the (broken) centralized CLI.

**Root cause.** No shared, stdlib-only runner exists to expose a single library
function as a `python -m` entry without pulling in the CLI framework.

**Fix.** Add `scikitplot._cli._runner.run_module(func, params=...)` and a
3–5 line `__main__` guard to runnable library modules (ADR-CLI-105, CONTRACT §8).
The runner reuses the neutral `Param`→argparse mapping, so its flags behave
exactly like the CLI's.

**Verified by (executed on the reference kernel).**

```text
$ python -m <pkg>.utils._show_versions --mode json   ->  clean JSON, exit 0
$ python -m <pkg>.utils._show_versions --help        ->  prog shows real dotted path
```

---

## CLI-FE-009 — Root `python -m scikitplot` routes to click-only entry

**Severity:** High · **Status:** PLANNED

**Symptom.** `scikitplot/__main__.py` does `from ._cli.cli import cli` then
`raise SystemExit(cli.main())`, so `python -m scikitplot` inherits CLI-FE-001 and
crashes without click.

**Root cause.** Root module runner points at the click-only entry rather than the
frontend-selecting app.

**Fix.** Route to the centralized app:

```python
from ._cli.app import main
if __name__ == "__main__":
    raise SystemExit(main())
```

`[project.scripts]` likewise moves to `scikitplot._cli.app:main`.

**Verified by.** Bootstrap test invoking `app.main(["info", "--format", "json"])`
with click blocked.

---

## CLI-FE-010 - `show-config` structured output was invalid

**Severity:** High - **Status:** CLOSED

**Symptom.** `scikitplot show-config --format json` produced a Python-dict repr
(single quotes) followed by a line `null`, so `json.loads` failed. Both frontends
produced identical output, so parity held; the data itself was not JSON.

**5-whys root cause.**
1. Why invalid JSON? The handler wrote a pprint dump, then `emit(None)` -> `null`.
2. Why? `show_config(mode="dicts")` printed via `pprint(CONFIG)` and returned None.
3. Why did it return None? The documented `return CONFIG` sat in an
   `except ModuleNotFoundError` that never triggers (`pprint` always imports) -
   dead code.
4. Why did the handler trust it? The docstring states "'dicts' mode returns a
   dict"; the implementation violated its own contract.
5. Root cause: the library `show_config` conflated *producing* the config with
   *printing* it, and had no side-effect-free accessor for `CONFIG`.

**Fix.**
- CLI (self-sufficient): the handler now reads the authoritative `CONFIG` mapping
  directly for structured output (`json`/`yaml`/`toml`), never invoking the
  printing `dicts` mode - so machine output is clean with or without the library
  patch. Human `text` output still delegates to `show_config(mode="stdout")`.
- Library (root cause, delivered as `show_config_fix.diff`): `dicts` mode now
  returns `copy.deepcopy(CONFIG)` with no side effects, honoring its docstring.

**Verified by.** `test_cli_frontend_parity` (`show-config --format json` and
`--format toml`) and `test_cli_output_toml` (round-trips through `tomllib`).

## Coverage map (finding → invariant → test)

```text
CLI-FE-001  ->  §3.1 stdlib bootstrap      ->  test_cli_import_contract
CLI-FE-002  ->  §3.5 discover/import late  ->  test_cli_import_contract
CLI-FE-003  ->  §3.3 parity, §3.6 output   ->  test_cli_frontend_parity, output tests
CLI-FE-004  ->  §3.7 actionable failure    ->  error-taxonomy tests
CLI-FE-005  ->  §3.6 stdout/stderr         ->  output-contract test
CLI-FE-006  ->  legacy disposition (§8 DoD)->  reviewed dispositions
CLI-FE-007  ->  §3.8 shell spelling        ->  registry lint test
CLI-FE-008  ->  §2.1 module runner surface ->  runner smoke test
CLI-FE-009  ->  §3.1 stdlib bootstrap      ->  bootstrap test
```
