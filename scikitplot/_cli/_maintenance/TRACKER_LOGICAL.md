# Logical Tracker — `scikitplot._cli`

What the code **promises**. Not re-derivable from the tree.

## 1. Contracts

| Contract | Where | Invariant that must not break |
|---|---|---|
| delegation registry | `registry.py`, `_spec.py` | Targets are **strings**, resolved at runtime. `_cli` must never import a delegated submodule — that would make every optional dependency mandatory for `--help`. |
| target resolution | `loader.py` | `module:attr`, via `importlib` / `runpy`. A malformed target raises with the expected shape named. |
| output contract | `output.py`, `exit_codes.py` | `(stdout, stderr, exit code)` is the API. **Exit codes are consumed by scripts**, so they are a compatibility surface, not a detail. |
| frontend parity | `_frontends/` | `argparse` and `click` expose the same commands with the same behaviour. Divergence gives two user populations different software. |
| error channel | `errors.py` | A failure the user can act on: what went wrong, and what to install or change. |
| import contract | `tests/test_cli_import_contract.py` | Importing `_cli` must not import the submodules it can delegate to. |

## 2. Cross-cutting invariants

| Invariant | Enforced by |
|---|---|
| Every delegation target resolves | `check_trackers.py` |
| `_cli` imports no delegated submodule | `tests/test_cli_import_contract.py` |
| Both frontends stay at parity | `tests/test_cli_frontend_parity.py` |
| Exit codes match outcomes | `tests/test_cli_errors.py` |

**This submodule is the best-instrumented of the five.** It already has an
import-contract test and a frontend-parity test — the exact kind of check the
other campaigns had to *add*. The gate contributes the one thing missing:
verifying that delegation strings point at something real.

## 3. Known logical debt

| Item | Consequence |
|---|---|
| A dangling target fails at use time | The command appears in `--help` and then raises `ImportError` |
| The only current target is `scikitplot.mcp.__main__:main` | It cannot be validated end-to-end until MCP's suite collects — MCP finding `MCP-M00-01` |
| No documented exit-code table | Scripts depend on these; the mapping should be published, not inferred |
