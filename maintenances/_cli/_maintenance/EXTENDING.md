# Adding a submodule to the scikit-plots CLI

> **Audience:** anyone who built a scikit-plots submodule (often in its own chat
> or PR) and wants it reachable as `scikitplot <name> ...`.
> **Read first:** [`MAINTAINING.md`](./MAINTAINING.md) for the CLI's invariants.

This guide is **submodule-independent**. `mcp` is used only as the worked
example; the same steps apply to any submodule.

There are two ways to expose a submodule. Pick by answering one question:

> **Does the submodule already have its own command-line parsing (its own
> `argparse`, subcommands, and `--help`)?**

- **Yes → Delegated command** (§1). The CLI forwards everything to the submodule.
  One registry line. This is how `mcp` is integrated.
- **No → Native command** (§2). You describe the arguments once and write a small
  neutral handler; you get `--format`, `-v/-q`, and argparse↔click parity for free.

---

## 1. Delegated command (pass-through) — the default for self-contained submodules

### 1.1 The contract your submodule must satisfy

Expose a callable in the submodule that accepts an argument vector and returns an
exit code:

```python
# scikitplot/<yourmod>/__main__.py
def main(argv: list[str] | None = None) -> int:
    ...
    return 0
```

`mcp` already does exactly this (`scikitplot/mcp/__main__.py::main`). Notes:

- Returning an `int` is the exit code. Raising `SystemExit` (which `argparse`
  does for `--help` and validation errors) is fine — the CLI converts it to an
  exit code.
- Write results to **stdout**, diagnostics/logging to **stderr** (CLI invariant
  §3.6). `mcp` does this: JSON config to stdout, logs to stderr.
- Do all argument parsing yourself. The CLI does **not** parse your options.

### 1.2 Register one line

Add a `CommandSpec` with `delegate` (and no `handler`/`params`) to
[`registry.py`](./registry.py):

```python
CommandSpec(
    name="mcp",
    summary="Run or probe the scikit-plots documentation MCP server.",
    delegate="scikitplot.mcp.__main__:main",      # "module:attr" -> attr(argv) -> int
    capabilities=("mcp",),                          # advisory; shown by `doctor`
    install_hint="Install the MCP extra: pip install scikit-plots[mcp]",
)
```

Target forms accepted by `delegate`:

| Form | Meaning | Use when |
| --- | --- | --- |
| `"module:attr"` | Import `module` lazily, call `attr(argv) -> int` | The submodule exposes `main(argv)` (**recommended**) |
| `"module"` | Execute like `python -m module` via `runpy` | The submodule only has a bare `__main__` guard |

That is the entire integration. Nothing else changes.

### 1.3 What you get automatically

- **`scikitplot mcp ...`** forwards every trailing argument verbatim, including
  `--help` — so `scikitplot mcp --help` prints the submodule's own help.
- **Lazy import.** The submodule is imported only when its command runs, so the
  CLI stays stdlib-only at startup (invariant §3.1) and unrelated commands work
  even if the submodule's dependencies are absent.
- **Actionable failure.** If the submodule or a dependency is missing, the user
  sees a `CapabilityMissingError` (exit code 69) with your `install_hint`, not a
  traceback. (Verified: with `pydantic` absent, `scikitplot mcp --help` reports
  the missing capability instead of crashing.)
- **Exit-code propagation.** The submodule's `int` return (or `SystemExit` code)
  becomes the process exit code.
- **Global options before the command still work.** `scikitplot -v mcp serve`
  applies `-v` at the CLI, then forwards `serve` to the submodule.
- **argparse and click parity.** Both frontends forward byte-identical argv.
- **Listed in help.** The command appears in `scikitplot --help`.

### 1.4 How it works (so you can trust it)

The CLI cannot let its own `argparse` touch your options (a stray `--help` or an
unknown flag would be rejected). So delegated commands are intercepted **before**
parsing:

- **argparse frontend** (`_frontends/_argparse.py`): `_split_delegated(argv)`
  scans for the first registered command; if it is delegated, the tokens before
  it are parsed as global options (`-v/-q/-V/-h`) and everything after it is
  handed to `loader.run_delegate` untouched. (This avoids `argparse.REMAINDER`,
  which drops a leading `--help`.)
- **click frontend** (`_frontends/_click.py`): a command with
  `add_help_option=False` + `ignore_unknown_options` and a
  `nargs=-1, type=UNPROCESSED` argument captures everything and forwards it.
- **Dispatch** (`loader.run_delegate`): lazy import → call `main(argv)` →
  normalize `SystemExit`/`ImportError` → return an exit code.

---

## 2. Native command — for first-party commands that should feel built in

Use this when the submodule has **no** CLI of its own and you want the command to
share the CLI's output and verbosity contracts (e.g. `info`, `doctor`,
`show-versions`).

### 2.1 Write a neutral handler

```python
# scikitplot/_cli/_commands/<name>.py
from ..context import Context
from ..output import emit

def run(ctx: Context, *, fmt: str = "text") -> int:
    """No click, no rich; return an exit code; write results to ctx.stdout."""
    data = {...}                     # gather from your library API (lazy import)
    if fmt == "text":
        ...                          # human rendering to ctx.stdout
        return 0
    emit(ctx, data)                  # json / yaml / toml handled centrally
    return 0
```

Rules (all enforced by tests): signature `run(ctx, *, **params) -> int`; no
`click`/`rich` import; results to `ctx.stdout`, diagnostics to `ctx.stderr`; never
enumerate output formats (let `output.emit` own json/yaml/toml — see
`FINDINGS` CLI-FE-011).

### 2.2 Register with typed params

```python
CommandSpec(
    name="<name>",
    summary="One line shown in help.",
    handler="scikitplot._cli._commands.<name>:run",
    params=(FORMAT,),                # reuse the shared FORMAT param
)
```

Each `Param` is rendered by **both** frontends from the same spec, so argparse and
click stay in sync automatically. Long flags use hyphens (`--mask-envs`); the
handler receives snake_case (`mask_envs`).

---

## 3. Security & robustness checklist (both styles)

Before merging a new command, confirm:

- [ ] **Lazy import.** Nothing heavy is imported at CLI startup; the submodule is
      imported only when its command runs. (`import scikitplot._cli` must stay
      stdlib-only — there is an import-contract test.)
- [ ] **stdout = results, stderr = diagnostics.** `scikitplot <cmd> --format json |
      jq .` (native) or the submodule's machine output must be clean on stdout.
- [ ] **Exit codes.** Return `0` on success, non-zero on failure; use the codes in
      `exit_codes.py` for native commands.
- [ ] **Actionable optional-dependency handling.** Missing extras produce a clear
      message + install hint, never a raw traceback.
- [ ] **No secrets in help or output.** Redact tokens/credentials (see `doctor
      --mask-envs`). If your submodule binds a network port, default to localhost
      and require an explicit opt-in for remote binds (as `mcp` does with
      `--docker` / `--allow-unauthenticated-remote`).
- [ ] **No argument mangling for delegated commands.** The submodule owns parsing;
      the CLI must forward argv unchanged (parity test covers this).
- [ ] **Deterministic.** Behavior does not depend on whether `click` happens to be
      installed (argparse is the default frontend; ADR-CLI-101).
- [ ] **Tests added.** Delegated: assert routing + forwarded argv + missing-dep
      error (see `tests/test_cli_delegation.py`). Native: add to the parity and
      format-coverage matrices.

---

## 4. Recommended workflow ("develop in its own chat, then wire in")

1. **Build the submodule independently**, exposing `main(argv) -> int` in its
   `__main__.py`. Keep its CLI, tests, and docs inside the submodule
   (`scikitplot/<mod>/`), exactly as `mcp` does.
2. **Decide the style** with the question at the top (self-contained CLI →
   delegated; no CLI → native).
3. **Add one `CommandSpec`** to `registry.py` (delegate or handler).
4. **Add a test** (delegated: `tests/test_cli_delegation.py` pattern; native:
   parity + format coverage).
5. **Run** `pytest scikitplot/_cli/tests -q` and verify
   `scikitplot <name> --help` works with the frontend both present and absent
   (`SCIKITPLOT_CLI_FRONTEND=click` / default).
6. **Document** any new optional extra in `pyproject.toml`
   (`[project.optional-dependencies]`, version ranges, no `==`).

---

## 5. Worked example: `mcp`

- Submodule: `scikitplot/mcp/` with `__main__.py::main(argv) -> int` (its own
  argparse: `--transport`, `--docker`, `--self-test`, `--probe`,
  `--print-effective-config`, …).
- Registry: the delegated `CommandSpec` in §1.2.
- Result:

  ```console
  $ scikitplot mcp --print-effective-config     # mcp's JSON config on stdout
  $ scikitplot mcp --help                        # mcp's own help
  $ scikitplot mcp --self-test                    # mcp's read-only self-test
  $ scikitplot -v mcp --docker                    # -v at CLI, rest to mcp
  $ scikitplot mcp                                # (mcp deps absent) -> actionable
                                                  #   "capability 'mcp' not available;
                                                  #    pip install scikit-plots[mcp]"
  ```

No CLI code changed to add `mcp` beyond the single registry entry.
