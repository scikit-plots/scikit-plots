# Verification — `scikitplot._cli`

## Commands

```console
$ python scikitplot/_cli/_maintenance/check_trackers.py
$ python -m pytest scikitplot/_cli -q -p no:cacheprovider
$ python -m scikitplot._cli --help          # must work with nothing else installed
```

The third is load-bearing. **`--help` must not require any delegated
submodule** — that is the whole point of string delegation.

## What the gate checks

| Check | Fails when |
|---|---|
| DRIFT | inventory differs from the tree by more than 10% |
| DELEGATION | a target's module is not importable, or its attribute is missing |
| TRIPWIRE | `__pycache__` in the tree; test:source below 0.08 |

## What is NOT verified today

| Claim | Status |
|---|---|
| Every exit code is documented | **no table exists** |
| `--help` works with no optional dependencies | tested by `test_cli_import_contract.py`; not asserted end-to-end |
| The `mcp` delegation works end-to-end | **blocked** — MCP's suite does not collect without `[mcp]` (`MCP-M00-01`) |

## Evidence standard

- **"I tested it" is insufficient.** Paste the command and its output.
- A finding is resolved **with evidence**, never deleted.
- A test is never weakened to make a change pass.
