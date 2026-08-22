# Maintaining `_sphinx_ai_backend`

> **PROPOSED submodule** — contents live in `_sphinx_ai_assistant/` today.

Entry point. **Self-contained**: a fresh session needs no chat history.

```text
archive: scikit-plots.zip
sha256:  59bfa61efc838a2e1daa17335a7f861f9d5232fd69930140455533a461385950
```

## One of three submodules

```text
_sphinx_llm  --build artifacts-->  _sphinx_ai_assistant  --HTTP-->  _sphinx_ai_backend
```

Three *different kinds* of edge. Read `_maintenance/DEPENDENCY_MAP.md` before
any change that crosses one.

## Read order

1. `_maintenance/DEPENDENCY_MAP.md`
2. `_maintenance/MAINTENANCE_MODEL.md`
3. `_maintenance/TRACKER_LOGICAL.md`
4. `_maintenance/TRACKER_PHYSICAL.md`
5. `_maintenance/SUBMODULE_STRUCTURE.md`
6. `_maintenance/VERIFICATION.md` — **what is not verified**

Machine-readable: `_maintenance/STATE.json`, `TRACKER.json`.

## Current state

```text
source  15 files /   9896 LOC
tests    0 files /      0 LOC
backup   0 files /      0 LOC
open findings: 4   (S00 input — revalidate, do not accept)
```

**Do not begin implementation.** Establish the big picture first, as Corpus, MCP
and the annoy family did.

## Two corrections to the bootstrap

1. **Decline the `sphinx_llm/` → `upstream/` rename.** Renaming the vendored
   directory is itself a modification of it.
2. **Export the deployable services** as `_sphinx_ai_backend` — 9 896 LOC with
   zero tests, currently invisible inside an extension.

## The unfinished thing

The assistant is **claimed** to reach MCP for verified sources. Wiring exists in
six files. Whether sources are *verified*, and what happens on a `DEGRADED`
retrieval, is unestablished — and cannot be settled from this side alone. It
needs MCP's run **M04**. Recorded as checkpoint **S05**.
