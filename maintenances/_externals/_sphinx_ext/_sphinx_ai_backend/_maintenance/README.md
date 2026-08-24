# `_maintenance/` — `_sphinx_ai_backend` (LIVE v1)

> **PROPOSED submodule** — contents live in `_sphinx_ai_assistant/` today.

Self-contained: a fresh session needs no chat history.

Mirrors `corpus`, `mcp` and the annoy family. Supersedes the
`..._maintenance_bootstrap_20260820.zip` skeleton.

## Read order

| # | File | Answers |
|---|---|---|
| 1 | `../MAINTAINING.md` | What is this, what state is it in |
| 2 | `DEPENDENCY_MAP.md` | **The family graph** — three different edge kinds |
| 3 | `MAINTENANCE_MODEL.md` | why / when / where / which / how many / how much |
| 4 | `TRACKER_LOGICAL.md` | What this submodule promises |
| 5 | `TRACKER_PHYSICAL.md` | What is on disk; tripwires |
| 6 | `SUBMODULE_STRUCTURE.md` | Where things go; debt disposition |
| 7 | `VERIFICATION.md` | **What is NOT verified** — read this one |

## First command

```console
$ python scikitplot/_externals/_sphinx_ext/_sphinx_ai_backend/_maintenance/check_trackers.py
$ sphinx-build -b html docs/source docs/_build    # must work with NO backend
```

## The one rule

> The vendored NVIDIA tree is never edited — not even renamed. Everything we add
> lives **beside** it.

And the one thing not to forget: **the MCP edge is claimed, not proven.** See
`VERIFICATION.md`.
