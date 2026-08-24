# B08 — Static Representation Consumer

Status: **NOT_STARTED**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Make `_sphinx_llm` canonical static Markdown the assistant’s primary page-context source.

## Prerequisites

- `B07`
- `_sphinx_llm:A11`

## Scope

- manifest/alternate lookup
- fidelity selection
- untrusted reference packaging

## Non-goals

- delete fallback immediately

## Required evidence before editing

- Record the exact source snapshot/commit being reviewed.
- Re-run the maintenance tracker before changing production code.
- Name the logical contract(s) touched by this checkpoint.
- Record current behavior with a test, build artifact, source anchor, or explicit `UNVERIFIED` status.
- If external/upstream behavior matters, pin the upstream revision used as evidence.

## Execution record schema

Fill these fields in this file when the checkpoint becomes active:

```yaml
checkpoint: B08
status: NOT_STARTED | IN_PROGRESS | BLOCKED | COMPLETE | DEFERRED | SUPERSEDED
started_at: null
completed_at: null
source_anchor: null
upstream_anchor: null
production_code_modified: false
contracts_touched: []
files_read: []
files_changed: []
findings_opened: []
findings_closed: []
risks: []
rollback: null
```

## Verification gates

- [ ] static canonical selected when available
- [ ] no system-role page injection

## Closure rule

A checkpoint is not `COMPLETE` until:

1. its evidence is reproducible from the repository;
2. every changed contract has a regression gate;
3. `REGISTRY.md`, `STATE.json`, and relevant tracker files agree;
4. any remaining limitation is explicitly `DEFERRED` or `BLOCKED`, never hidden;
5. the next bounded checkpoint is `B09`.

---

## Revised intent (pivot)

The assistant is no longer a *consumer* of someone else's artifacts. It is the
**producer**: `generate_markdown_files()` and `generate_llms_txt()` run on
`build-finished`, after the normal Sphinx build, over the final HTML.

That ordering is what makes the whole extension stack a non-problem. Every
directive — `sphinx_design`, `sphinx_tabs`, Sphinx-Gallery, IPython, Matplotlib,
JupyterLite, the PyData theme directives — has already been resolved to HTML
before conversion begins. There is no custom node to teach a Markdown visitor
about.

## The contract this checkpoint owns

```text
CANONICAL     static page.md          build-time, machine-fetchable
CONVENIENCE   browser Turndown        runtime, clipboard only

VIEW      -> canonical      (static .md URL)
ASK AI    -> canonical      (static .md URL; a blob: URL is unfetchable externally)
COPY      -> convenience by default, canonical when the toggle selects `static`
```

## Exit criteria

```text
[ ] page.md exists for every built page that is not excluded
[ ] llms.txt lists them, in the structured v2 layout
[ ] COPY in `static` mode and ASK AI resolve to byte-identical content
[ ] the assistant produces all three surfaces with _sphinx_llm absent
```
