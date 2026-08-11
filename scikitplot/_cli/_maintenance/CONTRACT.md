# CLI Frontend Contract

The precise technical contract for the framework-neutral CLI core and its two
frontends. Every code block below was executed and its behavior verified; the
parity and bootstrap evidence is in §6.

Scope: this document specifies *what to build* and *the rules both frontends must
obey*. It is a reference, not the shipped module — package it under
`scikitplot._cli` (adjusting the `refkernel.` import prefixes to `scikitplot.`).

---

## 1. The neutral intermediate representation (`_spec.py`)

Both frontends consume the same two frozen dataclasses. The IR expresses only the
*intersection* of what argparse and click can render faithfully (ADR-CLI-102).

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Literal

ParamKind = Literal["flag", "option", "argument"]

@dataclass(frozen=True, slots=True)
class Param:
    dest: str                         # snake_case handler kwarg, e.g. "mask_envs"
    flags: tuple[str, ...] = ()        # shell spellings, e.g. ("--mask-envs",)
    kind: ParamKind = "option"
    help: str = ""
    type: Callable[[str], Any] | None = None   # int, float, pathlib.Path, ...
    default: Any = None
    required: bool = False
    multiple: bool = False             # append / nargs="*"
    count: bool = False                # -v -vv -vvv
    negatable: bool = False            # --flag / --no-flag
    choices: tuple[str, ...] | None = None
    metavar: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "argument" and self.flags:
            raise ValueError(f"argument {self.dest!r} must not declare flags")
        if self.kind != "argument" and not self.flags:
            raise ValueError(f"{self.kind} {self.dest!r} must declare at least one flag")
        if self.count and self.kind != "option":
            raise ValueError(f"count param {self.dest!r} must be kind='option'")
        if self.negatable and self.kind != "flag":
            raise ValueError(f"negatable param {self.dest!r} must be kind='flag'")

@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    summary: str
    handler: str                      # "package.module:attr"
    params: tuple[Param, ...] = ()
    aliases: tuple[str, ...] = ()
    hidden: bool = False
    deprecated: bool = False
    capabilities: tuple[str, ...] = ()
```

Rules:

- `dest` is the canonical name; the handler receives it as a keyword argument.
- `flags` use shell conventions (hyphens). `dest` stays snake_case
  (invariant §3.8 in `MAINTAINING.md`). Frontends never re-derive `dest` from
  flags — it is passed explicitly, so `--mask-envs` reliably maps to `mask_envs`
  in both frontends.
- The IR carries no frontend-specific styling. Rich help/colors are presentation,
  handled in `output.py`, never here.

---

## 2. The neutral context (`context.py`)

```python
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import Literal, TextIO

Format = Literal["text", "json", "yaml"]

@dataclass(slots=True)
class Context:
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)
    fmt: Format = "text"
    color: bool = True
    verbosity: int = 0
```

- Streams are bound at `Context` construction (i.e. per invocation), which makes
  handlers testable by redirecting `sys.stdout` before the frontend runs.
- Handlers MUST write results to `ctx.stdout` and diagnostics to `ctx.stderr`.
  They MUST NOT call `print()` to the real stdout, `click.echo`, or `rich`.

---

## 3. Registry and loader (metadata-only, lazy dispatch)

`registry.py` — explicit metadata, no handler imports (ADR-CLI-103):

```python
from ._spec import CommandSpec, Param

FORMAT = Param(dest="fmt", flags=("--format",), kind="option",
               choices=("text", "json", "yaml"), default="text",
               help="Output format.")

BUILTIN_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(name="info", summary="Show runtime and installation information.",
                handler="scikitplot._cli._commands.info:run", params=(FORMAT,)),
    CommandSpec(name="doctor", summary="Diagnose the current environment.",
                handler="scikitplot._cli._commands.doctor:run",
                params=(Param(dest="mask_envs", flags=("--mask-envs",), kind="flag",
                              default=False, help="Mask sensitive env values."),
                        FORMAT)),
)

BY_NAME = {s.name: s for s in BUILTIN_COMMANDS}
BY_ALIAS = {a: s.name for s in BUILTIN_COMMANDS for a in s.aliases}
assert len(BY_NAME) == len(BUILTIN_COMMANDS), "duplicate command name"
for _a in BY_ALIAS:
    assert _a not in BY_NAME, f"alias {_a!r} collides with a command name"
```

`loader.py` — resolves the target and dispatches; shared by both frontends:

```python
import importlib
from typing import Any
from ._spec import CommandSpec
from .context import Context

def dispatch(spec: CommandSpec, params: dict[str, Any], ctx: Context) -> int:
    module_name, sep, attr = spec.handler.partition(":")
    if not sep:
        raise ValueError(f"handler {spec.handler!r} must be 'module:attr'")
    module = importlib.import_module(module_name)   # lazy: only on invocation
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise TypeError(f"handler target {spec.handler!r} is not callable")
    return int(fn(ctx, **params))
```

A neutral handler (`_commands/info.py`):

```python
import json, platform
from ..context import Context

def run(ctx: Context, *, fmt: str = "text") -> int:
    data = {"scikitplot": {"version": "..."},
            "python": {"version": platform.python_version()},
            "platform": {"system": platform.system()}}
    if fmt == "json":
        json.dump(data, ctx.stdout, indent=2, sort_keys=True); ctx.stdout.write("\n")
    else:
        for k, v in data.items():
            ctx.stdout.write(f"{k}: {v}\n")
    return 0
```

Production note: real `yaml` output belongs in `output.py` behind a lazy import
that fails with an actionable error if PyYAML is absent (Tier 2). Never import
`yaml` at module top level.

---

## 4. Frontend builders

### 4.1 argparse frontend (`_frontends/_argparse.py`) — always available

```python
import argparse, sys
from typing import Sequence
from .._spec import CommandSpec, Param
from ..context import Context
from ..registry import BUILTIN_COMMANDS
from ..loader import dispatch

def _add_param(p: argparse.ArgumentParser, prm: Param) -> None:
    if prm.kind == "argument":
        p.add_argument(prm.dest, nargs="*" if prm.multiple else "?",
                       default=prm.default, help=prm.help,
                       metavar=prm.metavar or prm.dest.upper())
        return
    if prm.count:
        p.add_argument(*prm.flags, action="count", default=prm.default or 0,
                       dest=prm.dest, help=prm.help)
        return
    if prm.kind == "flag":
        if prm.negatable and hasattr(argparse, "BooleanOptionalAction"):
            p.add_argument(*prm.flags, action=argparse.BooleanOptionalAction,
                           default=bool(prm.default), dest=prm.dest, help=prm.help)
        else:
            p.add_argument(*prm.flags, action="store_true",
                           default=bool(prm.default), dest=prm.dest, help=prm.help)
            if prm.negatable:  # explicit --no-x for Python 3.8 (see 4.3)
                neg = tuple("--no-" + f.lstrip("-") for f in prm.flags if f.startswith("--"))
                p.add_argument(*neg, action="store_false", dest=prm.dest,
                               help=argparse.SUPPRESS)
        return
    p.add_argument(*prm.flags, dest=prm.dest, help=prm.help,
                   type=prm.type or str, default=prm.default, required=prm.required,
                   action="append" if prm.multiple else "store",
                   choices=list(prm.choices) if prm.choices else None,
                   metavar=prm.metavar)

def build_parser(specs: Sequence[CommandSpec] = BUILTIN_COMMANDS) -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="scikitplot", add_help=True)
    sub = root.add_subparsers(dest="_command", metavar="COMMAND")
    for spec in specs:
        if spec.hidden:
            continue
        cp = sub.add_parser(spec.name, help=spec.summary, aliases=list(spec.aliases),
                            description=spec.summary)
        for prm in spec.params:
            _add_param(cp, prm)
        cp.set_defaults(_spec=spec)
    return root

def run(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    ns = parser.parse_args(argv)
    spec = getattr(ns, "_spec", None)
    if spec is None:
        parser.print_help(sys.stderr)
        return 0
    params = {prm.dest: getattr(ns, prm.dest) for prm in spec.params}
    ctx = Context(fmt=params.get("fmt", "text"))
    return dispatch(spec, params, ctx)
```

### 4.2 click frontend (`_frontends/_click.py`) — imported only when selected

```python
import sys
from typing import Any, Sequence
import click                                  # top-level here is fine: this
from .._spec import CommandSpec, Param        # module is imported only when the
from ..context import Context                 # click frontend is chosen
from ..registry import BUILTIN_COMMANDS
from ..loader import dispatch

def _decorate(fn, prm: Param):
    if prm.kind == "argument":
        return click.argument(prm.dest, nargs=-1 if prm.multiple else 1,
                              required=prm.required,
                              default=None if prm.multiple else prm.default)(fn)
    if prm.count:
        return click.option(*prm.flags, prm.dest, count=True,
                            default=prm.default or 0, help=prm.help)(fn)
    if prm.kind == "flag":
        decl = "/".join(prm.flags) if not prm.negatable else \
               prm.flags[0] + "/--no-" + prm.flags[0].lstrip("-")
        return click.option(decl, prm.dest, is_flag=not prm.negatable,
                            default=bool(prm.default), help=prm.help)(fn)
    return click.option(*prm.flags, prm.dest,
                        type=click.Choice(list(prm.choices)) if prm.choices
                             else (prm.type or str),
                        default=prm.default, required=prm.required,
                        multiple=prm.multiple, metavar=prm.metavar, help=prm.help)(fn)

def _make_command(spec: CommandSpec) -> click.Command:
    def callback(**params: Any) -> None:
        ctx = Context(fmt=params.get("fmt", "text"))
        code = dispatch(spec, params, ctx)
        if code:
            raise SystemExit(code)
    callback.__name__ = spec.name.replace("-", "_")
    cmd = callback
    for prm in reversed(spec.params):       # reversed -> declared order preserved
        cmd = _decorate(cmd, prm)
    return click.command(name=spec.name, help=spec.summary, hidden=spec.hidden)(cmd)

def build_group(specs: Sequence[CommandSpec] = BUILTIN_COMMANDS) -> click.Group:
    @click.group(name="scikitplot")
    def root() -> None: ...
    for spec in specs:
        root.add_command(_make_command(spec))
        for alias in spec.aliases:
            root.add_command(_make_command(spec), name=alias)
    return root

def run(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    return build_group().main(args=argv, standalone_mode=False,
                              prog_name="scikitplot") or 0
```

Production hardening (deferred, tracked in `FINDINGS.md`): replace `_make_command`
with a lazy metadata command so click's help formatter renders from `spec.summary`
without importing the handler (guide §9, §96). The reference above imports the
handler only on `dispatch`, which already satisfies the L1 boundary for
execution; the lazy metadata command additionally protects the *help* path.

### 4.3 Edge case — negatable booleans and Python 3.8

`argparse.BooleanOptionalAction` (the clean `--flag/--no-flag`) exists only on
Python ≥ 3.9. scikit-plots targets 3.8+. The argparse builder therefore feature-
detects it and, on 3.8, registers an explicit hidden `--no-<flag>` paired with
`store_false` to preserve parity with click's `--flag/--no-flag`. This keeps
invariant §3.3 (semantic parity) true across the whole supported Python range.

Other edge cases the IR handles uniformly: `multiple` (argparse `append` /
click `multiple`), `count` (`-v -vv`), `choices` (argparse `choices` /
`click.Choice`), and positional `argument` (`nargs`).

---

## 5. Selection and entry (`app.py`, `__main__.py`)

```python
# app.py
import os, sys
from typing import Sequence

def _select_frontend(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    choice = env.get("SCIKITPLOT_CLI_FRONTEND", "").strip().lower()
    if choice == "click":
        from ._frontends import is_click_available
        if is_click_available():
            return "click"
        sys.stderr.write("scikitplot: click not installed; using argparse.\n")
    return "argparse"                          # deterministic default (ADR-CLI-101)

def main(argv: Sequence[str] | None = None) -> int:
    if _select_frontend() == "click":
        from ._frontends import _click
        return _click.run(argv)
    from ._frontends import _argparse
    return _argparse.run(argv)
```

```python
# __main__.py
from .app import main
if __name__ == "__main__":
    raise SystemExit(main())
```

```toml
# pyproject.toml (target)
[project.scripts]
scikitplot = "scikitplot._cli.app:main"
```

`_frontends/__init__.py` exposes availability without importing click:

```python
import importlib.util
def is_click_available() -> bool:
    return importlib.util.find_spec("click") is not None
```

---

## 6. Verification (executed)

The kernel above was run end-to-end. `info --format json` produced byte-identical
JSON from both frontends; `doctor --mask-envs --format json` likewise.

**Parity + bootstrap suite** — argparse vs click over an argv matrix, plus a
bootstrap test that blocks `click`/`rich`/`yaml` in `sys.modules` and asserts the
argparse frontend still imports and runs:

```text
test_frontend_parity[info]                        PASSED
test_frontend_parity[info --format json]          PASSED
test_frontend_parity[info --format yaml]          PASSED
test_frontend_parity[doctor]                      PASSED
test_frontend_parity[doctor --mask-envs]          PASSED
test_frontend_parity[doctor --mask-envs json]     PASSED
test_bootstrap_needs_no_click                     PASSED
7 passed
```

These two tests are the mechanical enforcement of invariants §3.1 (stdlib-only
bootstrap) and §3.3 (frontend parity). They belong in the shipped test suite as:

- `test_cli_frontend_parity.py` — the argv matrix; assert equal exit codes and
  equal `--format json`/`yaml` bytes.
- `test_cli_import_contract.py` — block Tier 2/3 modules; assert `import
  scikitplot._cli` and `app.main(["--help"])` and `--format json` all succeed.

Extend the matrix whenever a command or param is added (workflow step 5,
`MAINTAINING.md` §7).

---

## 8. Standalone module runners (`python -m ...`)

The second invocation surface (ADR-CLI-105). A single library function is the
source of truth; the module runner is a thin, stdlib-only argparse over that
function's *native* parameters — no registry, no click, no subcommands.

`_cli/_runner.py` — the shared minimal runner (reuses the neutral param mapping):

```python
"""Minimal, stdlib-only standalone runner for a single library function."""
from __future__ import annotations
import argparse, sys
from typing import Any, Callable, Sequence
from ._spec import Param
from ._frontends._argparse import _add_param      # reuse the one param->argparse map

def run_module(func: Callable[..., Any], params: Sequence[Param] = (),
               argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if prog is None:
        # Under `python -m pkg.mod` the module loads as __main__; recover the
        # real dotted name from its spec so help text reads correctly.
        main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
        mod_name = getattr(main_spec, "name", None) or func.__module__
        prog = f"python -m {mod_name}"
    parser = argparse.ArgumentParser(
        prog=prog, add_help=True,
        description=(func.__doc__ or "").strip().splitlines()[0] if func.__doc__ else None)
    for prm in params:
        _add_param(parser, prm)
    ns = parser.parse_args(argv)
    kwargs = {prm.dest: getattr(ns, prm.dest) for prm in params}
    result = func(**kwargs)
    return result if isinstance(result, int) else 0
```

The `__main__` guard added to a runnable library module
(e.g. `scikitplot/utils/_show_versions.py`):

```python
if __name__ == "__main__":
    import sys
    from scikitplot._cli._runner import run_module
    from scikitplot._cli._spec import Param
    sys.exit(run_module(
        show_versions,
        params=(Param(dest="mode", flags=("--mode",), kind="option",
                      choices=("stdout", "dict", "json", "yaml"), default="stdout",
                      help="Output mode."),),
    ))
```

**Verified behavior** (executed against the reference kernel):

```text
$ python -m scikitplot.utils._show_versions
               scikitplot: 0.0.0-ref
                   python: 3.x

$ python -m scikitplot.utils._show_versions --mode json
{ "python": "3.x", "scikitplot": "0.0.0-ref" }

$ python -m scikitplot.utils._show_versions --help
usage: python -m scikitplot.utils._show_versions [-h] [--mode {stdout,dict,json,yaml}]
```

**Root `python -m scikitplot`** (`scikitplot/__main__.py`) MUST route to the
centralized app, not a click-only entry (FINDINGS CLI-FE-009):

```python
from ._cli.app import main
if __name__ == "__main__":
    raise SystemExit(main())
```

**Surface summary** — one logic path, three doors:

```text
library     scikitplot.show_versions(mode=...)            <- source of truth
module run  python -m scikitplot.utils._show_versions     <- run_module, native opts
central CLI scikitplot show-versions --format json        <- registry + handler,
                                                              argparse default / click opt-in
```

The module runner and CLI command MAY differ in option spelling (`--mode` vs
`--format`) because they serve different audiences; they are semantically
equivalent because both terminate in the same library function.

---

## 9. Optional: keeping docs and code in sync

A lightweight `test__maintainer_docs.py` MAY assert that:

- every `CommandSpec.name` in `registry.BUILTIN_COMMANDS` appears in
  `FINDINGS.md` or is covered by a parity-matrix entry;
- the `FINDINGS.md` status table parses and contains no `OPEN` item marked done
  elsewhere.

This mirrors the maintenance-scaffolding pattern used elsewhere in scikit-plots
(findings verifiable from inside the installed module).
