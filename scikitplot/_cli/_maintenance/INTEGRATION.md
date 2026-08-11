# Integrating the CLI slice

The framework-neutral CLI is complete: six built-in commands (`info`, `doctor`,
`show-versions`, `show-config`, `sysinfo`, `greet`), both frontends, the module
runner, and 20 passing tests. This note lists the remaining edits to files that
live *outside* `scikitplot/_cli/`, plus the old files to delete.

## 1. Console entry point (`pyproject.toml`)

```toml
[project.scripts]
# before: scikitplot = "scikitplot._cli.cli:cli"
scikitplot = "scikitplot._cli.app:main"
```

`click`/`rich` stay OUT of `[project].dependencies`. Optional enhanced frontend:

```toml
[project.optional-dependencies]
cli = ["click>=8.1,<9"]        # optional; argparse works without it
```

## 2. Root `python -m scikitplot` (`scikitplot/__main__.py`)

Delivered in this drop as `scikitplot/__main__.py`:

```python
from ._cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
```

(The old body did `from ._cli.cli import cli` + `cli.main()`, which inherited the
base-install crash. `scikitplot._cli.cli:cli` still resolves via the shim in §4.)

## 3. Standalone runner guard (`scikitplot/utils/_show_versions.py`)

Append ONLY this guard to the **existing** module (do not replace it; it already
defines `show_versions`). Enables `python -m scikitplot.utils._show_versions`:

```python
if __name__ == "__main__":
    import sys

    from .._cli._runner import run_module
    from .._cli._spec import Param

    sys.exit(
        run_module(
            show_versions,
            params=(
                Param(
                    dest="mode", flags=("--mode",), kind="option",
                    choices=("stdout", "dict", "yaml", "rich"),
                    default="stdout", help="Output mode.",
                ),
            ),
        )
    )
```

## 4. Old files: DELETE these (superseded)

The migrated neutral handlers replace the old click-based ones. Nothing else in
the repo imports these (verified). Delete:

```text
scikitplot/_cli/_cmd_options/                       -> DELETE (whole package; pip-derived, unused)
scikitplot/_cli/_misc.py                            -> DELETE (only _cmd_options_optparse used it)
scikitplot/_cli/_commands/greet.py        (old)     -> REPLACED by neutral handler in this drop
scikitplot/_cli/_commands/show_config.py  (old)     -> REPLACED
scikitplot/_cli/_commands/show_versions.py(old)     -> REPLACED
scikitplot/_cli/_commands/sysinfo.py      (old)     -> REPLACED
```

`scikitplot/_cli/cli.py` is KEPT as a thin deprecation shim (`cli = main`) so the
historical target `scikitplot._cli.cli:cli` keeps resolving. Remove it once no
external caller imports that path.

## 5. Verify

```bash
pytest scikitplot/_cli/tests -q          # 20 passed
python -m scikitplot info --format json | jq .
python -m scikitplot greet --no-emoji Allen
python -m scikitplot.utils._show_versions --mode json
SCIKITPLOT_CLI_FRONTEND=click python -m scikitplot info --format json   # if click installed
```
