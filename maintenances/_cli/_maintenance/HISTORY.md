# History — `scikitplot._cli`

Read only when the *why* behind a current rule is unclear.

## This submodule

`_backup/` preserves the previous CLI: `optparse` and `click` option modules, a
`_misc.py`, its own `cli.py` and command set. The current implementation
replaced it with a registry-and-loader design — string delegation, two
frontends, an explicit output contract — and kept the old tree in place.

The existing `_maintenance/` folder already held a design guide, a contract, a
decisions log, findings, an extending guide and a ruleset. That is more
maintenance material than any other submodule started with, and this set
supersedes only the tracker-shaped parts of it.

## What the other campaigns established

`_cli` is the fifth and last submodule to get a maintenance pair. The four before
it each solved the same problem — keep an optional dependency optional — by a
different mechanism:

| Module | Mechanism |
|---|---|
| `corpus` | ~288 deferred (call-time) imports |
| `mcp` | the `DocsRetriever` Protocol; backends injected |
| annoy family | relative paths inside Cython `cdef extern` |
| `_cli` | module-path strings in a registry |

All four share the property that the dependency is invisible to a static reader,
and all four fail at use time rather than import time. Each campaign's gate
checks its own mechanism; this one checks the strings.

Observation **O-6** — `__pycache__` shipping in the release archive — was
recorded during the Corpus campaign and routed here. It is still open.
