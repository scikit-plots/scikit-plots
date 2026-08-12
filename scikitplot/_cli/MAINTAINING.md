# Maintaining `scikitplot._cli`

> **New here / no chat history?** Read [`_maintenance/RULESET.md`](./_maintenance/RULESET.md) first — it is the continuation contract (rules + how to proceed).

> **Status:** Foundation / pre-implementation contract
> **Audience:** Maintainers of the scikit-plots CLI runtime
> **Primary design rule (unchanged):** *Discover early, import late, initialize only when required.*
> **Primary design rule (added by this document):** *Command logic is framework-neutral. `argparse` is always available; `click` is an optional adapter that renders the **same** logic.*

This file is the single entry point a maintainer reads first. It states the big
picture, the invariants that every change must preserve, the module contract,
the dependency policy, and the acceptance gate. The detailed rationale, technical
contract, and tracked work live in [`_maintenance/`](./_maintenance/).

It **amends** the uploaded `_maintenance/CLI_SUBMODULE_DESIGN_GUIDE.md`. Where the two differ,
this file and the records under `_maintenance/` win, and the superseded guide
sections are named explicitly in [`_maintenance/DECISIONS.md`](./_maintenance/DECISIONS.md).

---

## 1. Why this refactor exists (root cause, not symptom)

The shipped console entry point is:

```toml
[project.scripts]
scikitplot = "scikitplot._cli.cli:cli"
```

`scikitplot._cli.cli` imports `click` at module top level. So does every module
under `_commands/` and `_cmd_options/` (several also import `rich` at top level).

But `click` and `rich` are **not** declared in `[project].dependencies`. They
appear only inside a development extras group. Therefore, on a clean install of
the library, the very first thing the console script does is:

```text
$ scikitplot --help
ModuleNotFoundError: No module named 'click'
```

The public CLI is unrunnable for any user who does not also install the dev
extras. This is not a styling bug or a missing feature — it is a **framework
coupling** defect: the command *logic* is written *as* `click` callbacks, so
there is no way to run a command without `click`.

The fix at root level is not "add `click` to core dependencies." That would make
every base install pay for a CLI framework and a terminal renderer, and it
contradicts the guide's own dependency discipline. The fix is to **stop writing
logic inside `click`** and instead:

1. put command logic in framework-neutral handlers that depend only on the
   standard library;
2. make `argparse` (standard library, always present) the default frontend;
3. make `click` an optional frontend that projects the **same** neutral command
   specifications when it is installed.

See [`_maintenance/FINDINGS.md`](./_maintenance/FINDINGS.md) for the full,
tracked list of defects this refactor closes.

---

## 2. Big picture

```text
                         shell argv
                            │
                            ▼
                 ┌────────────────────┐
                 │  FRONTEND SELECT   │   argparse by default;
                 │  app.main(argv)    │   click only when opted in
                 └─────────┬──────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                      ▼
┌────────────────┐                    ┌────────────────┐
│ argparse       │  same specs        │ click          │  optional
│ frontend       │◄──────┐    ┌──────►│ frontend       │  (Tier 2)
│ (stdlib only)  │       │    │       │ (if installed) │
└───────┬────────┘       │    │       └───────┬────────┘
        │                │    │               │
        │        ┌───────┴────┴───────┐       │
        │        │  COMMAND REGISTRY  │       │   metadata only,
        │        │  CommandSpec/Param │       │   no handler imports
        │        └─────────┬──────────┘       │
        │                  │                  │
        └────────► build neutral Context ◄────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  LAZY HANDLER LOAD │   import "module:attr"
                 │  loader.dispatch   │   only on invocation
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  NEUTRAL HANDLER   │   run(ctx, **params) -> int
                 │  stdlib-only logic │   calls library/service API
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  RENDERER (ctx)    │   text / json / yaml
                 └─────────┬──────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
              stdout              stderr
             (result)           (diagnostics)
```

The one-sentence description that must remain true as the CLI grows:

> **The registry knows what commands exist; a frontend (argparse by default,
> click when opted in) renders them; the runtime loads only the handler,
> dependency, and resources the user actually activates — and the handler never
> depends on which frontend called it.**

### 2.1 Three invocation surfaces, one logic path

The same capability is reachable three ways, and there is exactly one place the
logic lives — the **library function**. Everything else is a thin adapter over it
(ADR-CLI-105):

```text
library call    scikitplot.show_versions(mode=...)              <- source of truth
module runner   python -m scikitplot.utils._show_versions       <- stdlib argparse,
                                                                   the function's native
                                                                   options, no click
centralized CLI scikitplot show-versions --format json          <- registry + neutral
                                                                   handler; argparse
                                                                   default, click opt-in
```

- The **module runner** is "module based, without CLI interface": a 3–5 line
  `__main__` guard delegating to `scikitplot._cli._runner.run_module`. Stdlib
  only. No registry, no subcommands, no click.
- The **centralized CLI** normalizes options to the public contract (e.g.
  `--format text|json|yaml`) and is what `[project.scripts]` and `python -m
  scikitplot` invoke.
- Module runner and CLI command MAY differ in option *spelling* (`--mode` vs
  `--format`); they are semantically equivalent because both call the same
  library function. This is invariant §3.3 applied across surfaces.

The exact `run_module` helper and the root `__main__.py` routing are specified in
[`_maintenance/CONTRACT.md`](./_maintenance/CONTRACT.md) §8.

---

## 3. Core invariants (never violate)

Each invariant below is objective and testable. The acceptance gate in §8 and the
tests named in [`_maintenance/CONTRACT.md`](./_maintenance/CONTRACT.md) exist to
enforce them.

1. **Stdlib-only bootstrap.** `import scikitplot._cli` and
   `scikitplot._cli.app.main(["--help"])` MUST succeed using only the Python
   standard library. No `click`, `rich`, `yaml`, `matplotlib`, `mlflow`, or any
   feature dependency may be imported to reach `--help` or `--version`.

2. **Handlers are framework-neutral.** A command handler MUST have the signature
   `run(ctx: Context, /, **params) -> int` and MUST NOT import `click`,
   reference a `click.Context`, or call `click.echo`/`rich` directly. It receives
   already-parsed native values and a neutral `Context`.

3. **Frontends differ in presentation, never in semantics.** For any argv, the
   argparse and click frontends MUST resolve the same command, the same parameter
   values, and the same exit code, and MUST produce byte-identical machine output
   (`--format json`, `--format yaml`). Colors, help layout, and text prose MAY
   differ.

4. **argparse is the deterministic default.** With no explicit opt-in, behavior
   MUST NOT change based on whether `click` happens to be installed. The frontend
   is selected by an explicit, documented switch (see §5).

5. **Discover early, import late.** The registry holds metadata only and MUST NOT
   import handler modules. Rendering top-level help MUST NOT import any handler.
   A handler is imported only when its command is actually invoked.

6. **stdout is the result channel; stderr is for diagnostics.** Logging,
   progress, and human hints go to stderr. `scikitplot info --format json | jq .`
   MUST receive only JSON on stdout.

7. **Optional capability failures are actionable, not tracebacks.** A missing
   optional dependency or unavailable platform capability MUST produce a
   normalized CLI error and a stable non-zero exit code, not a raw import
   traceback (unless debug mode is requested).

8. **Shell-convention spelling.** CLI option and command names use hyphens
   (`--mask-envs`, `show-config`); Python `dest`/variable names stay snake_case
   (`mask_envs`). Both frontends derive spellings from the same `Param`.

---

## 4. Module contract

Target layout for the rebuilt submodule. Create a module only when a real
responsibility exists for it.

```text
scikitplot/_cli/
├── __init__.py            # exports app.main; no heavy imports
├── __main__.py            # `python -m scikitplot._cli` -> raise SystemExit(main())
├── app.py                 # frontend selection + main(argv) -> int
│
├── _spec.py               # Param, CommandSpec  (frozen dataclasses, stdlib only)
├── _runner.py             # run_module(func, params): standalone `python -m` runner
├── registry.py            # BUILTIN_COMMANDS, alias/collision indexes (metadata only)
├── loader.py              # "module:attr" import + dispatch (lazy)
├── context.py             # Context: io streams, format, color, verbosity
├── errors.py              # CLI error taxonomy
├── exit_codes.py          # centralized semantic exit codes
├── output.py              # text/json/yaml rendering; safe fallback
├── logging.py             # stderr-routed logging policy
│
├── _frontends/
│   ├── __init__.py        # is_click_available(); no top-level click import
│   ├── _argparse.py       # build ArgumentParser from specs; run(argv) -> int
│   └── _click.py          # build click app from specs; run(argv) -> int  (opt import)
│
├── _commands/             # neutral handlers: run(ctx, **params) -> int
│   ├── __init__.py
│   ├── info.py
│   └── doctor.py
│
├── MAINTAINING.md         # this file
└── _maintenance/          # decisions, contract, findings, workflow
```

Responsibilities:

- **`_spec.py`** — the neutral intermediate representation. Owns `Param` and
  `CommandSpec`. Depends only on the standard library. This is the shared source
  of truth both frontends consume. Frozen and validated in `__post_init__`.
- **`registry.py`** — explicit `BUILTIN_COMMANDS` tuple plus immutable
  name/alias indexes. Metadata only. MUST NOT import handler modules. Collisions
  are validated at import/test time.
- **`loader.py`** — resolves a `handler` target of the form `"package.module:attr"`,
  imports it lazily, validates it is callable, and dispatches
  `handler(ctx, **params)`. Normalizes import/attribute errors into the taxonomy.
- **`context.py`** — the neutral runtime `Context` (stdout, stderr, format,
  color, verbosity). Cheap to construct; no I/O, no network, no file writes.
- **`_frontends/_argparse.py`** — projects specs onto `argparse`. Always
  importable. Owns the argv→params mapping and the neutral `Context` construction
  for the argparse path.
- **`_frontends/_click.py`** — projects the same specs onto `click`, including a
  lazy metadata command so top-level help does not import handlers. Imported only
  when the click frontend is selected.
- **`_commands/*`** — thin, framework-neutral adapters. Validate CLI-specific
  input, call a library/service function, return an exit code, render through
  `output`. No business or scientific logic lives here.

The precise dataclass fields, both frontend builders, the parity rules, and a
runnable reference kernel are specified in
[`_maintenance/CONTRACT.md`](./_maintenance/CONTRACT.md).

---

## 5. Frontend-selection policy

`app.main` chooses a frontend once, before parsing, via one explicit switch:

```text
SCIKITPLOT_CLI_FRONTEND = "argparse"  -> force argparse (default when unset)
SCIKITPLOT_CLI_FRONTEND = "click"     -> use click if installed, else argparse + stderr note
(unset)                               -> argparse
```

**Recommended default: argparse**, even when `click` is installed. Rationale
(recorded as ADR-CLI-101 in `_maintenance/DECISIONS.md`): determinism across
desktop, Docker, CI, and notebooks is worth more than automatically prettier
output. A CLI whose parsing and machine output depend on whether an optional
package is present is an environment-coupled contract, which invariant §3.4
forbids. Users who want the enhanced click experience opt in explicitly.

The switch is a single function (`app._select_frontend`). If the project later
decides to auto-enable click on interactive terminals, that is a one-line policy
change in that function plus an ADR update — no change to handlers, specs, or the
registry.

---

## 6. Dependency policy (amended)

The guide's Tier 1 previously listed `click`. This document moves `click` out of
the kernel.

```text
Tier 0  standard library        argparse, importlib, json, logging, ...
        -> the CLI kernel. Always available. Bootstrap runs here.

Tier 1  (empty for the kernel)  no third-party package is required to run the CLI.

Tier 2  optional presentation   click, rich, yaml serializer
        -> imported only when the relevant frontend/renderer is selected.

Tier 3  feature dependencies    matplotlib, mlflow, streamlit, gradio, native ANN
        -> imported only on the specific command path that needs them.

Tier 4  plugins                 third-party entry-point commands
        -> discovered by metadata; imported only when activated.
```

Rules:

- The CLI MUST run its bootstrap and built-in metadata commands with Tier 0 only.
- `click`/`rich`/`yaml` MUST be imported behind a guarded, lazy path and MUST fail
  with an actionable message if requested but absent.
- Follow the project dependency-versioning policy: no `==` pinning; declare
  ranges; document any upper bound. If `click`/`rich` are later added as an
  optional extra (e.g. `scikit-plots[cli]`), use ranges and test against minimum
  and latest.

---

## 7. Maintenance workflow

For any change to this submodule:

1. **Read** this file and [`_maintenance/DECISIONS.md`](./_maintenance/DECISIONS.md)
   before touching code.
2. **Plan** in `_maintenance/FINDINGS.md` (or `tasks/todo.md` per the project
   review rules) if the change is ≥ 3 steps, ≥ 2 files, or architectural.
3. **Preserve the invariants** in §3. If a change would break one, stop and open
   a decision record instead.
4. **Keep logic in handlers.** New commands add a `CommandSpec` to the registry
   and a neutral handler under `_commands/`. Do not add `click`-only logic.
5. **Prove parity.** Any new command MUST pass the frontend-parity test
   (§8, item 3) for its machine output.
6. **Verify** with the acceptance gate (§8) and paste evidence (test output) into
   the finding/task before marking it done.

---

## 8. Acceptance gate (definition of done)

The rebuild — and every later command — is complete only when all of the
following are demonstrably true (with evidence, not assertion):

**Bootstrap**

- [ ] `python -c "import scikitplot._cli"` succeeds with `click`, `rich`, and
      `yaml` uninstalled.
- [ ] `scikitplot --help` and `scikitplot --version` succeed with no Tier 2/3
      imports (verified by an import-contract test that stubs `sys.modules`).
- [ ] Bootstrap performs no network access and no filesystem writes.

**Lazy loading**

- [ ] Registry import does not import any module under `_commands/`.
- [ ] Rendering top-level help does not import any handler.
- [ ] A broken optional command does not prevent `--help` from working.

**Frontend parity (invariant §3.3)**

- [ ] A parametrized parity test runs a matrix of argv through both frontends and
      asserts identical exit codes and identical `--format json`/`yaml` bytes.
- [ ] argparse-only environment (click stubbed absent) passes the full suite.

**Automation & output**

- [ ] `scikitplot info --format json | jq .` yields clean JSON; stderr carries
      any diagnostics.
- [ ] Exit codes come from `exit_codes.py`, not scattered literals.

**Reliability & security**

- [ ] Optional/feature failures produce normalized errors, not raw tracebacks.
- [ ] No broad `except Exception: pass` in command/alias resolution.
- [ ] Diagnostics redact sensitive environment values (e.g. `--mask-envs`).

**Invocation surfaces**

- [ ] `python -m scikitplot.utils._show_versions` runs the library function via
      `run_module` with click uninstalled; `--help` shows the real dotted prog
      name; `--mode json` yields clean JSON.
- [ ] `python -m scikitplot` routes to `scikitplot._cli.app:main` (not a
      click-only entry) and works with click uninstalled.

**Legacy disposition**

- [ ] Every retained helper from `_misc.py` / `_cmd_options_optparse.py` is
      classified KEEP / MIGRATE / DEPRECATE / DELETE with a reason
      (see `_maintenance/FINDINGS.md`, CLI-FE-006).

---

## 9. Cross-references

- [`_maintenance/EXTENDING.md`](./_maintenance/EXTENDING.md) — how to add any submodule to the CLI (delegated pass-through, e.g. `mcp`, or native commands).

- [`_maintenance/README.md`](./_maintenance/README.md) — index and workflow.
- [`_maintenance/DECISIONS.md`](./_maintenance/DECISIONS.md) — ADRs, including the
  argparse-first inversion and the guide sections it supersedes.
- [`_maintenance/CONTRACT.md`](./_maintenance/CONTRACT.md) — the neutral IR, both
  frontend builders, the parity rules, and a runnable reference kernel.
- [`_maintenance/FINDINGS.md`](./_maintenance/FINDINGS.md) — tracked defects and
  their root causes.
- `_maintenance/CLI_SUBMODULE_DESIGN_GUIDE.md` — the prior, click-centric guide this document
  amends.
