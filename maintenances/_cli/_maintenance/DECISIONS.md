# CLI Architecture Decision Records

These records amend `CLI_SUBMODULE_DESIGN_GUIDE.md`. Where a guide ADR and a
record here conflict, the record here is authoritative. Superseded guide sections
are named explicitly.

Normative language (from the guide): **MUST**, **MUST NOT**, **SHOULD**,
**SHOULD NOT**, **MAY**.

---

## ADR-CLI-100 — The CLI kernel is framework-neutral; argparse is the base

**Supersedes:** guide ADR-CLI-001 ("Use Click as the CLI kernel"); guide §88
Tier 1 (which listed `click`).

**Context.** The console entry point `scikitplot._cli.cli:cli` hard-imports
`click`, and every command/option module imports `click` (and often `rich`) at
module top level. Neither `click` nor `rich` is in `[project].dependencies`; they
are dev-only extras. On a clean install, `scikitplot --help` raises
`ModuleNotFoundError: click`. The public CLI is therefore unrunnable for base
installs. (See FINDINGS CLI-FE-001.)

**Decision.**

1. Command *logic* MUST live in framework-neutral handlers with signature
   `run(ctx: Context, /, **params) -> int`, depending only on the standard
   library plus the project's own APIs.
2. `argparse` (standard library) is the default frontend and MUST always be
   available.
3. `click`, when installed and selected, is an *adapter* that projects the same
   `CommandSpec`s. It MUST NOT be required to run the CLI.
4. The CLI bootstrap and built-in metadata commands MUST run on Tier 0 (stdlib)
   alone.

**Consequences.**

- The "kernel" is now the neutral IR (`_spec.py`), the registry, the loader, and
  the argparse frontend — all stdlib-only.
- `click`/`rich`/`yaml` become Tier 2 optional presentation, imported lazily.
- No change to core dependencies is required; the shipped CLI works on a base
  install.

**Rejected alternatives.**

- *Add `click` to core dependencies.* Rejected: forces every base install to
  carry a CLI framework and terminal renderer, and still leaves logic coupled to
  one framework.
- *Keep click as kernel, guard the import with a friendly error.* Rejected: the
  CLI would still be non-functional without an optional package; it treats the
  symptom (bad error message) rather than the coupling.

---

## ADR-CLI-101 — argparse is the deterministic default even when click is present

**Supersedes:** none (new policy). Refines guide §6.5 (human vs machine
contracts) and guide §35 (structured output stability).

**Context.** With two available frontends, "which one runs by default" is a
public-contract decision. If the default were "click when installed," then argv
parsing, help, and machine output could change based on whether an optional
package is present in the environment — a hidden, environment-coupled contract.

**Decision.** Frontend selection is governed by one explicit switch,
`SCIKITPLOT_CLI_FRONTEND`:

```text
"argparse"  -> force argparse
"click"     -> use click if installed; else fall back to argparse + a stderr note
unset       -> argparse (deterministic default)
```

The selection is isolated in `app._select_frontend(argv, env)`.

**Consequences.**

- Behavior is reproducible across desktop, Docker, CI, and notebooks regardless
  of optional packages.
- Users opt into the enhanced click experience explicitly.
- Auto-enabling click later (e.g. on interactive TTYs) is a one-line change to
  `_select_frontend` plus an ADR update; handlers, specs, and registry are
  unaffected.

**Trade-off (acknowledged).** Users with `click` installed do not get colored,
grouped help by default. This is accepted in exchange for a deterministic
contract; it is discoverable via `--help` text that names the env switch.

---

## ADR-CLI-102 — One neutral intermediate representation, consumed by both frontends

**Supersedes:** none (new). Complements guide §14 (`CommandSpec`).

**Context.** "Same logic in argparse and click" is only achievable without
duplication if commands are described once in a form both frontends can render.

**Decision.** `_spec.py` defines two frozen dataclasses — `Param` and
`CommandSpec` — as the single source of truth. Each frontend has a pure builder
that maps specs onto its own primitives. The IR expresses only the *intersection*
of what both frontends can render faithfully. Anything one frontend cannot
express is presentation, not semantics, and lives in the renderer — never in the
IR.

**Consequences.**

- Adding a command is: append a `CommandSpec`, write a neutral handler. Both
  frontends pick it up automatically.
- A parity test (CONTRACT §5) mechanically enforces that both frontends agree on
  command resolution, parsed values, exit codes, and machine output.

**Constraint.** Frontend-specific niceties (rich help, colors, click's
`--foo/--no-foo` sugar) MUST NOT leak into `Param`. Negatable booleans are
expressed once in the IR (`kind="flag"`, `negatable=True`) and each builder
realizes them in its own way (CONTRACT §4.3).

---

## ADR-CLI-103 — Registry holds metadata only; handlers load lazily by target string

**Reaffirms:** guide ADR-CLI-002, ADR-CLI-003, and §§14–18. Restated here because
it is essential to argparse-first parity.

**Decision.** `registry.BUILTIN_COMMANDS` is an explicit tuple of `CommandSpec`.
Each spec's `handler` is a `"package.module:attribute"` string. The registry MUST
NOT import handler modules. The loader imports the target only on invocation and
validates it is callable. Both frontends share the same loader and dispatch path.

**Consequences.** Top-level help (in either frontend) renders from metadata alone.
The click frontend uses a lazy metadata command so click's help formatter does
not force handler imports (guide §9, §96).

---

## ADR-CLI-105 — Two invocation surfaces over one logic path

**Supersedes:** none (new). Complements ADR-CLI-100 and ADR-CLI-102.

**Context.** The same capability (e.g. version/environment reporting) must be
reachable both as a small standalone module runner and through the centralized
CLI, without duplicating logic:

```text
python -m scikitplot.utils._show_versions      # standalone module runner
scikitplot info | scikitplot show-versions     # centralized CLI command
```

**Decision.** There is exactly one logic path — the **library function**
(e.g. `scikitplot.utils._show_versions.show_versions`). Everything else is a thin
adapter over it:

1. **Library call.** `scikitplot.show_versions(...)` — the source of truth.
2. **Standalone module runner** — a 3–5 line `if __name__ == "__main__"` guard
   that calls `scikitplot._cli._runner.run_module(func, params=...)`. This is a
   minimal, stdlib-only argparse over the *function's own native parameters*
   (e.g. `--mode`). It has **no** centralized registry, **no** click, and **no**
   subcommand machinery — "module based, without CLI interface."
3. **Centralized CLI command** — a `CommandSpec` in the registry whose neutral
   handler wraps the same library function and *normalizes* its options to the
   public CLI contract (e.g. `--format text|json|yaml`). Rendered by argparse by
   default, click when opted in (ADR-CLI-100/101).

**Consequences.**

- The module runner and the CLI command MAY expose different option *spelling*
  (`--mode` vs `--format`) because they serve different audiences (developer/debug
  vs public UX). They are semantically equivalent because both terminate in the
  same library function. This is the presentation-vs-semantics split of
  invariant §3.3 applied across surfaces.
- `run_module` is stdlib-only and reuses the neutral `Param`→argparse mapping from
  the argparse frontend, so a module runner's flags behave exactly like the CLI's
  (types, choices, count, negatable) with zero extra code.
- The root `python -m scikitplot` entry (`scikitplot/__main__.py`) MUST route to
  `scikitplot._cli.app:main`, not to a click-only `cli` (FINDINGS CLI-FE-009).

**Layering note.** A runnable library module importing `scikitplot._cli._runner`
is acceptable: `_runner` pulls only the standard library and the neutral IR — no
click, no rich, no feature dependencies. A module that must not depend on `_cli`
at all MAY inline its own 5-line argparse instead; the library function remains
the single source of truth either way.

---

## ADR-CLI-104 — Retained ADRs from the guide

These guide decisions are carried forward unchanged and apply to both frontends:

- **ADR-CLI-004** — three lazy layers (command import / dependency import /
  resource initialization) remain separate boundaries.
- **ADR-CLI-005** — CLI is an adapter; scientific/application logic stays out of
  callbacks. Under this rebuild, that logic stays out of *both* argparse and click
  callbacks — it lives in neutral handlers.
- **ADR-CLI-006** — stdout is the result channel; diagnostics/logging to stderr.
- **ADR-CLI-007** — machine output is a stable, documented contract.
- **ADR-CLI-008** — plugins use package entry points; discovery does not imply
  import or trust.

---

## Superseded-guide-section index

| Guide reference | Status | Replacement |
| --- | --- | --- |
| ADR-CLI-001 (Click as kernel) | Superseded | ADR-CLI-100 |
| §88 Tier 1 lists `click` | Amended | `MAINTAINING.md` §6 (click -> Tier 2) |
| §11 responsibilities (Click-centric) | Extended | `MAINTAINING.md` §4 + `_frontends/` |
| §7 big picture (single Click app) | Extended | `MAINTAINING.md` §2 (frontend select) |
| §95 loader returns `click.Command` | Amended | CONTRACT §3 (loader returns handler; frontends wrap) |

---

## ADR-CLI-106 — Submodule integration via delegated (pass-through) commands

**Context.** Submodules (e.g. `mcp`) are frequently developed independently and
ship their own complete CLI (`argparse`, subcommands, `--help`, validation).
Re-describing every such option as native `CommandSpec` params would duplicate the
submodule's parser, couple the CLI to the submodule's internals, and drift.

**Decision.** Introduce a second command kind — **delegated** — alongside native
commands. A `CommandSpec` sets `delegate="module:attr"` (and no `handler`/`params`).
The frontends forward all trailing arguments (including `--help`) verbatim to the
submodule's `main(argv) -> int`, imported lazily by `loader.run_delegate`. A bare
`"module"` target is also supported and executed like `python -m module`.

**Consequences.**

- Adding a self-contained submodule to the CLI is a single registry line
  (see `EXTENDING.md`).
- The submodule owns its parsing; the CLI never mangles its options.
- Lazy import preserves the stdlib-only bootstrap invariant; a missing submodule
  or dependency surfaces as `CapabilityMissingError` (exit 69) with an
  `install_hint`, not a traceback.
- Global options (`-v/-q/-V/-h`) before the command are honored; everything after
  the command name is forwarded.
- Both frontends forward byte-identical argv (parity test).

**Implementation note.** argparse intercepts delegated commands *before* parsing
(`_split_delegated`) rather than using `argparse.REMAINDER`, which drops a leading
`--help`. click uses `add_help_option=False` + `ignore_unknown_options` with a
`nargs=-1, UNPROCESSED` argument.

**Rejected alternative.** *Re-implement each submodule's options as native params.*
Rejected: duplication, coupling, and drift; loses the submodule's own validation
and help.
