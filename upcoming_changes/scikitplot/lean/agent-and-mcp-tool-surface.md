---
title: "lean: no tool surface for agents and MCP clients"
status: open
kind: "other"
area: "scikitplot/lean"
discovered_during: "Design review of agent and MCP access to scikitplot.lean, gap probed on main ca0870b (2026-11-27)"
release_note: "required"
towncrier_section: "scikitplot.lean"
towncrier_type: "feature"
towncrier_fragment: ""
---

# lean: no tool surface for agents and MCP clients

## Summary

`LeanIndex` answers what a reviewer of a formal proof asks (summary, dossier, discovery, near-duplicates, refresh), but only as a Python API and a command line. An agent built on `scikitplot.agent` or a client of the Model Context Protocol (Claude Code, Cursor, VS Code and others) cannot call it as tools, and the MCP server in the package serves documentation search only.

## Why it matters

Proof review is a long, branching investigation: load the index, read the summary, open dossiers, follow red flags, compare near-duplicates, cite file and line. That is the work an agent does well and a person does slowly. Without tools, every client re-implements calls, size limits and safety rules, or shells out to the command line and parses Markdown.

## Current evidence

- The public surface is the class and the command line: `scikitplot.lean.__all__` is `AUDIT_FLAGS`, `LEAN_FLAGS`, `LeanIndex` (and `defined_commands` with PR 1025); `python -m scikitplot.lean` prints Markdown or `--json`.
- Every report is already JSON-ready, bounded by `max_items`/`top`, and names `file:line` for each item; a saved index loads in seconds (mathlib4: 1.4 s, 57 MB; 85 MB with `keep_shapes=True`, which lets near-duplicate search run without the files).
- `scikitplot.agent` provides `Tool` (a JSON Schema for its arguments, an effect and scopes) and `Toolset`, and gates every call through validation, permissions, scope, budget and approval, with a hash-chained audit trail.
- `scikitplot.mcp` is built on the official MCP SDK v2 and exposes one read-only tool, `search_docs`, and one resource; its README lists an agent loop among its non-goals.
- The plugins hold a generated `scikitplot-lean` agent (`plugins/scikitplot/agents/scikitplot-lean.md`) that helps write code using the module, not review a proof; `plugins/scikitplot-mcp/.mcp.json` configures only the `scikitplot-docs` server.
- `grep -rEil 'lean_toolset|lean_dossier|lean_summary' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.
- Related code exists: `scikitplot/agent/_tools.py`.
- Related code exists: `scikitplot/mcp/_integrations/_agno/docs_toolkit.py`, `scikitplot/mcp/_server.py`.

## Root cause / current understanding

Not implemented: the index was built as a library and a command line first.

## Expected behavior

One tool catalog, defined once and free of any SDK, that describes each Lean operation (name, JSON Schema of its arguments, effect, output bounds) and runs it against a loaded index; thin adapters that expose that catalog to `scikitplot.agent` and to MCP clients; and a skill that tells an agent how to run a review with them.

## Affected paths and ownership

- `scikitplot/lean/` (catalog, adapters)
- `scikitplot/agent/` (consumer, unchanged API)
- `scikitplot/mcp/` (route B only)
- `plugins/scikitplot-mcp/`, `plugins/_shared/skills/` (configuration, review skill)
- Maintenance owner: `maintenances/corpus/` (lean is maintained with the corpus reader)

## Constraints and non-goals

Optional extras imported lazily, no new base dependency; Python 3.8 floor for everything SDK-free (the MCP SDK v2 needs Python >= 3.10); read-only by default; deterministic output for the same index and arguments; bounded responses; no network access. The index never reads outside the roots it was loaded with (as `LeanIndex.load(root=)` does now); tool output quotes source text, which is untrusted content (a comment can say anything) and must be marked as data, never instructions. Do not widen the docs server's declared scope by accident (route B is an explicit decision).

## Edge cases to cover

- No index loaded, or a snapshot from an earlier development version: a tool error that says what to load, never a traceback
- Index loaded without `root`: statements are not quoted; tools say so per item instead of failing
- Target names no declaration, or several: the dossier's `matches` list, returned as a result, not an error
- A huge project (175,187 files): every response capped in items and bytes, with `count` and `cut` kept
- Near-duplicates at a low threshold: a time budget, refused or capped before it runs for minutes (0.5 takes about 4 minutes on mathlib4)
- Concurrent calls on one index: reports are read-only, but `refresh` mutates; it needs a lock or a separate effect class
- Layered indexes: every tool takes `layer` where the method does; an unknown layer is a validation error
- Source text containing instructions or control characters: delivered as data, sanitized of terminal escapes

## Proposed direction

**Phase 1, a tool catalog** (`scikitplot/lean/_tools.py`, standard library only). One entry per operation, each with a closed JSON Schema for its arguments, an effect, a response byte cap and a handler that calls the existing method:

| tool | method | effect |
|---|---|---|
| `lean_summary(layer, max_items)` | `summary` | read |
| `lean_dossier(target, max_items)` | `dossier` | read |
| `lean_find(name)` | `find` | read |
| `lean_discover(layer, top, min_instances)` | `discover` | read |
| `lean_near_duplicates(threshold, layer, against, min_pairs, top)` | `near_duplicates` | read, with a time budget |
| `lean_refresh(layer)` | `refresh` | refresh (off unless enabled) |

One function validates arguments against the schema. Responses keep each report's `count` and `cut`, and source text is wrapped as untrusted data.

**Phase 2, the agent adapter.** `lean_toolset(index, *, allow_refresh=False) -> scikitplot.agent.Toolset`, with scopes `lean:read` and `lean:refresh`, so the agent's five gates and audit trail apply unchanged.

**Phase 3, MCP.** Three routes:

- Route A (recommended): a separate server, `python -m scikitplot.lean.mcp --index FILE [--root DIR ...] [--allow-refresh]`, serving the catalog's tools, resources `lean://summary`, `lean://dossier/{name}` and `lean://layer/{name}/summary`, and an optional `lean-review` prompt. The docs server stays as it is. Its hardening (argument validation, result envelope, health endpoint) is reused, not copied.
- Route B: register the catalog in `scikitplot.mcp` behind an opt-in flag. One server to configure, but its declared scope widens and its README must change.
- Route C: no server. Agents call `python -m scikitplot.lean ... --json`. The least code, but every client parses output and sets its own limits.

**Phase 4, the plugin.** A `scikitplot-lean` entry in `plugins/scikitplot-mcp/.mcp.json` (route A), and a proof-review skill in `plugins/_shared/skills/` with its agent, kept apart from the generated `scikitplot-lean` coding agent (which `check_records.py --write-index` owns). The skill's process:

1. load the snapshot once (`root` for quotes);
2. `lean_summary`: flags, missing modules, generators;
3. `lean_discover`: headline results and unproven claims;
4. `lean_dossier` for each headline and each flagged result, following the red-flag paths;
5. `lean_near_duplicates(layer=<project>, against='')`: what the project restates;
6. a report in which every claim cites `file:line` and names the kernel check that settles it (`lake build`, `#print axioms`).

## Verification / acceptance criteria

- The catalog's schemas validate the arguments each method accepts and refuse the rest; one test per tool calls it on a small project and checks the result equals the method's
- Responses stay under the byte cap on mathlib4 with `count` and `cut` correct
- `lean_toolset` runs a scripted planner through `scikitplot.agent` with the audit trail verifying
- Route A: the server lists its tools and resources over stdio with the SDK's client, answers a dossier, and refuses a path outside its roots
- Python 3.8: the catalog and the agent adapter import and pass their tests without the MCP SDK

## Documentation impact

Lean user guide (a section on agents and MCP), API reference, the lean usage skill, the MCP README (route B) or a new server README (route A), a gallery example driving the agent adapter with a rule-based planner.

## Release-note promotion

Required: `scikitplot.lean` / `feature`, one bullet, when the change lands.
