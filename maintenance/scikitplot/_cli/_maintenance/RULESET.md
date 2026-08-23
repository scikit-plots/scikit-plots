# CLI RULESET — continuation contract

> **Why this file exists.** So that a new session, developer, or agent with **no
> chat history** can continue `scikitplot._cli` with the *same* architecture,
> invariants, and discipline. If you are picking this up cold: read this file
> first, then the three documents in §1, then proceed. Everything here is binding
> unless a new ADR in `_maintenance/DECISIONS.md` supersedes it.

---

## 1. Read-first order (bootstrap)

1. `RULESET.md` — this file (rules + how to continue).
2. `MAINTAINING.md` — big picture, invariants, module map, acceptance gate.
3. `_maintenance/DECISIONS.md` — the ADRs (why things are the way they are).
4. `EXTENDING.md` — how to add any submodule (delegated vs native).
5. `_maintenance/FINDINGS.md` — tracked defects/history; `_maintenance/CONTRACT.md`
   — the precise IR/frontend contract with a reference kernel.

Do not start coding until the change is located against these. **Big picture
before code.**

---

## 2. Mission & architecture (one screen)

- **Framework-neutral core.** Command *logic* lives in stdlib-only handlers
  (`run(ctx, **params) -> int`) or in submodules. Frontends only render.
- **argparse is always available and the deterministic default.** **click is an
  optional adapter** selected by `SCIKITPLOT_CLI_FRONTEND=click`. Behavior must
  not change based on whether click happens to be installed (ADR-CLI-100/101).
- **Registry-driven.** `registry.BUILTIN_COMMANDS` holds metadata only; handlers
  load lazily by `"module:attr"` (ADR-CLI-103).
- **Two command kinds:** *native* (typed params + neutral handler) and *delegated*
  (`delegate="module:attr"`, all argv forwarded to a submodule's `main(argv)->int`;
  ADR-CLI-106). `mcp` is the delegated example.
- **Three invocation surfaces, one logic path:** library call `scikitplot.foo()`,
  module runner `python -m scikitplot.<mod>`, centralized `scikitplot <cmd>`.

Module map: `_spec.py` (Param/CommandSpec IR), `registry.py`, `loader.py`
(lazy dispatch + `run_delegate`), `context.py`, `output.py` (text/json/yaml/toml),
`logging.py` (stderr + verbosity), `errors.py`, `exit_codes.py`, `app.py`
(frontend select), `_frontends/{_argparse,_click}.py`, `_runner.py`,
`_commands/*` (neutral handlers).

---

## 3. Hard invariants (never violate — enforced by tests)

- **INV-1 stdlib bootstrap.** `import scikitplot._cli` and `--help`/`--version`
  succeed with `click`/`rich`/`yaml`/feature deps absent.
- **INV-2 neutral handlers.** `run(ctx, /, **params) -> int`; no `click`/`rich`
  import; no format enumeration in handlers (let `output.emit` own formats).
- **INV-3 frontend parity.** For any argv, argparse and click resolve the same
  command, params, exit code, and byte-identical machine output (json/yaml/toml).
- **INV-4 deterministic default.** argparse unless explicitly opted into click.
- **INV-5 discover early, import late.** Registry imports no handler; help imports
  no handler; a handler/submodule is imported only when invoked.
- **INV-6 stdout = results, stderr = diagnostics.** `... --format json | jq .`
  stays clean. Logging/verbosity go to stderr.
- **INV-7 actionable failures.** Missing optional dep/capability →
  `CapabilityMissingError` (exit 69) + install hint, never a raw traceback.
- **INV-8 shell spelling.** Long flags hyphenated (`--mask-envs`); `dest`
  snake_case (`mask_envs`). Both frontends derive from the same `Param`.

---

## 4. Engineering rules (project constitution)

- **ER-1 Zero hallucination.** Every claim traceable to the latest uploaded
  source. Always work from the newest uploaded files; never cached/old copies.
  When unsure of a name/structure, open the file and check.
- **ER-2 Minimal-impact root-cause fixes.** Find the true cause (5 whys); no
  band-aids, no speculative refactors, no commenting-out of failing tests.
- **ER-3 Always-green gate.** The full suite must pass before packaging. Fix code
  or correct a wrong test's *contract* — never delete/disable a test to go green.
- **ER-4 Per-turn evidence.** Every claim is backed by a named guard test and a
  reproducible run result (paste the output). "It works" is insufficient.
- **ER-5 Conventional Commits; NumPyDoc docstrings** (Parameters, Returns, Raises,
  See Also, Notes, References, Examples order).
- **ER-6 No `==` version pins.** Use `>=`/`~=`/documented upper bounds; support
  Python 3.8 → 3.15+; feature-detect (e.g. `BooleanOptionalAction`, `tomllib`).
- **ER-7 Deliver drop-in.** Ship a folder-structured zip mirroring the package
  path + a short README; note any manual steps (deletes/patches the zip can't do).
- **ER-8 Respect the layers.** Handlers/submodules own logic; frontends render;
  registry holds metadata. Don't leak click/rich into the core.

---

## 5. Adding a command (summary; full guide in `EXTENDING.md`)

Decide by one question — *does the submodule already have its own CLI?*

- **Yes → delegated.** Submodule exposes `main(argv=None) -> int`. Add one line:
  `CommandSpec(name=..., summary=..., delegate="pkg.mod.__main__:main",
  capabilities=(...,), install_hint="pip install scikit-plots[extra]")`. All argv
  (incl. `--help`) forwarded; lazy import; actionable error if absent.
- **No → native.** Write `_commands/<name>.py::run(ctx, *, fmt="text", ...) -> int`
  and register `CommandSpec(name=..., handler="scikitplot._cli._commands.<name>:run",
  params=(MODE?, FORMAT, ...))`. Reuse the shared `FORMAT` param.

Then add tests (delegated → `tests/test_cli_delegation.py` pattern; native → parity
+ format-coverage matrices) and document any new extra in `pyproject.toml`.

---

## 6. Output, logging, exit codes

- Output formats: `--format text|json|yaml|toml`. `text`/`json` stdlib-only;
  `yaml`/`toml` are Tier-2 (lazy import; `CapabilityMissingError` if absent). TOML
  drops `None` (no null type); json/yaml preserve it. Handlers call `output.emit`.
- Verbosity: `-v`/`-vv`/`-vvv` up, `-q`/`-qq` down, accepted at root **and** every
  subcommand, summed; net level → `logging.configure` (stderr) and `ctx.verbosity`.
  `logging.resolve(v,q)` is the shared resolver (both frontends must agree).
- Exit codes from `exit_codes.py` (OK=0, USAGE=2, UNAVAILABLE=69, SOFTWARE=70,
  INTERRUPTED=130). Delegated commands propagate the submodule's code.

---

## 7. Testing & acceptance gate

Run: `pytest scikitplot/_cli/tests -q` (must be fully green in a built env).

Suites that must stay green and be extended with each change:
`test_cli_import_contract` (INV-1/5), `test_cli_frontend_parity` (INV-3),
`test_cli_output_contract`/`test_cli_output_toml` (INV-6), `test_cli_errors`
(INV-7), `test_cli_format_coverage` (every command × every format),
`test_cli_verbosity`, `test_cli_mode`, `test_cli_doctor`, `test_module_runners`,
`test_cli_delegation`.

A change is **done** only when: the gate is green with evidence pasted; new
behavior has a named test; no invariant is broken; docs (this file / EXTENDING /
FINDINGS / DECISIONS) are updated if the rules or decisions changed.

> **Sandbox note.** In an unbuilt source checkout, `show-config`/config-module and
> `mcp` tests may error only because optional/native modules (`scikitplot._testing`,
> `config._config`, `pydantic`) aren't importable there. That is an environment
> gap, not a code defect — verify in a built env or a faithful harness.

---

## 8. When continuing with no chat history — do this

1. Read §1 documents. Identify which invariant/ADR the task touches.
2. Reproduce the current behavior; capture the exact failure/output first.
3. Find the root cause (ER-2). Write the smallest fix that upholds §3 and §4.
4. Add/adjust a named test proving it; run the full gate (§7); paste evidence.
5. Update the relevant docs and add a `FINDINGS.md` entry if it's a fix/feature.
6. Package a drop-in zip mirroring `scikitplot/...`, with a README noting manual
   steps. State assumptions inline.

## 9. Never do

- Never make the default path import `click`/`rich`, or change behavior based on
  their presence. Never enumerate output formats inside a handler. Never write
  diagnostics to stdout. Never pin with `==`. Never delete/disable a test to pass.
  Never re-implement a submodule's own CLI as native params (use delegation).
  Never claim success without a reproducible run.
