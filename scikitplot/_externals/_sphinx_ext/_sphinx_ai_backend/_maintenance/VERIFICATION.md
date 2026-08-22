# Verification — `_sphinx_ai_backend`

## Commands

```console
$ python scikitplot/_externals/_sphinx_ext/_sphinx_ai_backend/_maintenance/check_trackers.py
$ sphinx-build -b html docs/source docs/_build     # must succeed with NO backend
```

The second is the load-bearing one: **documentation must build without a live
backend**. If it does not, every contributor needs a deployed proxy to build
docs.

## What is NOT verified today

| Claim | Status |
|---|---|
| The assistant reaches MCP for verified sources | **UNVERIFIED** — wiring in 6 files; behaviour unestablished |
| Returned sources are *verified* rather than displayed | **UNVERIFIED** |
| The MCP path honours `RetrievalResponse.DEGRADED` | **UNVERIFIED** — also MCP's M04 question |
| The proxy is safe against hostile input | **UNTESTED** — zero tests |
| No secrets are committed in service source | **UNCHECKED** |

Recording these as unverified is the point. A claimed capability with no probe
is the defect class three prior campaigns were spent removing.

## Evidence standard

- **"I tested it" is insufficient.** Paste the command and its output.
- A finding is resolved **with evidence**, never deleted.
- A test is never weakened to make a change pass.
