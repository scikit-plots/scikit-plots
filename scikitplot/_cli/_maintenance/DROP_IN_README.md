# scikit-plots CLI — drop-in replacement

Unzip at the **repository root** (the folder containing `scikitplot/`):

```bash
unzip scikitplot-cli-dropin.zip -d /path/to/scikit-plots
```

## Contents

```text
scikitplot/__main__.py                 # python -m scikitplot -> neutral app
scikitplot/_cli/                        # complete framework-neutral CLI runtime
├── app.py registry.py loader.py output.py …   # stdlib-only kernel
├── _frontends/{_argparse,_click}.py    # argparse always; click optional
├── _commands/{info,doctor,show_versions,show_config,sysinfo,greet}.py
├── _runner.py                          # python -m <module> standalone runner
├── tests/                              # 24 passing tests
└── _maintenance/                       # MAINTAINING.md, ADRs, contract, findings
show_config_fix.diff                    # root-cause library fix (see below)
```

## Output formats

`--format text|json|yaml|toml`. `text`/`json` are stdlib-only; `yaml` needs
PyYAML; `toml` needs a TOML writer (`tomli-w` or `toml`). Missing optional
writers produce an actionable error (exit 69), never a traceback. TOML drops
`null` values (TOML has no null type); json/yaml preserve them.

## Manual steps the zip cannot do

1. **Delete superseded files** (unzip cannot remove files):

   ```bash
   rm -rf scikitplot/_cli/_cmd_options
   rm -f  scikitplot/_cli/_misc.py
   ```

2. **Point the console script at the neutral entry** (`pyproject.toml`, optional —
   the old `cli:cli` target still resolves via the shim):

   ```toml
   [project.scripts]
   scikitplot = "scikitplot._cli.app:main"
   ```

3. **Apply the library fix** `show_config_fix.diff` (root cause of the
   `show-config --format json` bug):

   ```bash
   git apply show_config_fix.diff        # patches scikitplot/config/__config__.py
   ```

   The CLI already works without this patch (it reads the config mapping
   directly), but the diff fixes `scikitplot.show_config(mode="dicts")` for all
   callers so it returns a dict per its docstring instead of printing and
   returning None.

## Optional: `python -m scikitplot.utils._show_versions`

Append the guard from `scikitplot/_cli/_maintenance/INTEGRATION.md` §3 to the
existing `scikitplot/utils/_show_versions.py`.

## Standalone module runners (apply the two diffs)

Two library modules gain a self-contained argparse `main()` so they run directly:

```bash
git apply add_show_versions_runner.diff   # scikitplot/utils/_show_versions.py
git apply add_config_runner.diff          # scikitplot/config/__config__.py
```

Then:

```bash
python -m scikitplot.utils._show_versions -m dict     # or: stdout | yaml | rich
python -m scikitplot.config.__config__ --mode dicts   # or: stdout
```

Both expose `-m/--mode` (default `stdout`), matching the CLI `--mode`. Note: a
benign `RuntimeWarning` from `runpy` may appear on **stderr** because each parent
package eagerly imports the submodule; stdout stays clean. For warning-free use,
prefer `scikitplot show-versions` / `scikitplot show-config`.

## `--mode` on show-config / show-versions

Both commands now accept `--mode` (default `stdout`) exposing the library's native
render modes, alongside `--format`:

```bash
scikitplot show-versions --mode rich          # library's rich rendering
scikitplot show-versions --mode dict --format json
scikitplot show-config   --mode dicts --format toml
```

Precedence: an explicit structured `--format` (json/yaml/toml) wins and emits the
data; otherwise `--mode` drives. Defaults (`mode=stdout`, `format=text`) preserve
today's human output.

## Verify

```bash
pytest scikitplot/_cli/tests -q                       # 79 passed
python -m scikitplot show-config --format json | jq .
python -m scikitplot info --format toml
SCIKITPLOT_CLI_FRONTEND=click python -m scikitplot info --format json
```

`click`/`rich` are NOT required — the default argparse path never imports them.
