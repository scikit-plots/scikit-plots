# Maintaining `_sphinx_ai_backend`

> **PROPOSED submodule** — contents live in `_sphinx_ai_assistant/` today.

Entry point. **Self-contained**: a fresh session needs no chat history.

```text
archive: scikit-plots.zip
sha256:  3f0d6862c27c41811cc19a91e34bd9789a653328a9247dbf34492a1cccbd726d
```

## One of two live submodules

```text
_sphinx_ai_assistant  --HTTP-->  _sphinx_ai_backend
```

`_sphinx_llm` is frozen and has no live edge. The one that remains is **runtime
HTTP**: the backend is imported by nothing, deployed separately, and versioned
independently. Read `_maintenance/DEPENDENCY_MAP.md` before any change that
crosses it.

## Read order

1. `_maintenance/DEPENDENCY_MAP.md`
2. `_maintenance/MAINTENANCE_MODEL.md`
3. `_maintenance/TRACKER_LOGICAL.md`
4. `_maintenance/TRACKER_PHYSICAL.md`
5. `_maintenance/SUBMODULE_STRUCTURE.md`
6. `_maintenance/VERIFICATION.md` — **what is not verified**

Machine-readable: `_maintenance/STATE.json`, `TRACKER.json`.

Do not create new parallel files named `FINAL`, `REVISED`, `EXPANDED`,
date-suffixed, or chat-specific variants inside the source tree.

## Current state

```text
source  15 files /   9942 LOC all files /  8384 LOC of .py and .js
tests    0 files /      0 LOC
backup   0 files /      0 LOC
open findings: 5   (S00 input — revalidate, do not accept)
```

Both LOC figures are recomputed and gated by `_maintenance/check_trackers.py`.
The single figure of 9 896 previously recorded here was not reproducible under
either counting rule.

**Do not begin implementation.** Establish the big picture first, as Corpus, MCP
and the annoy family did.

## The correction that still stands

**Export the deployable services** as `_sphinx_ai_backend` — 8 384 LOC of
internet-facing `.py`/`.js` with zero tests, currently invisible inside an
extension whose suite covers the extension.

The earlier vendoring correction is moot: `_sphinx_llm` is frozen.

## The unfinished thing

The assistant is **claimed** to reach MCP for verified sources. Wiring exists in
six files. Whether sources are *verified*, and what happens on a `DEGRADED`
retrieval, is unestablished — and cannot be settled from this side alone. It
needs MCP's run **M04**. Recorded as checkpoint **S05**.
