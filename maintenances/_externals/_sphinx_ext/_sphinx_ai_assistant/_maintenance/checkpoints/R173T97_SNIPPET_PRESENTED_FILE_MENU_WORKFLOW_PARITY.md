# R173T97 — Snippet / Presented-file menu workflow parity

Status: COMPLETE

Base: R173T96 speak-toggle real-device paint stability.

## Reported UX issue

Two overflow menus used the same popup shell but taught different workflows.
Anonymous answer snippets exposed only `Save as a tracked file…` and
`Download as…`, while Presented/tracked files exposed the clearer four-step
workflow `Open in a sheet`, `Save as…`, `Download patch`, `Continue editing`.
The visual component was shared, but the action grammar was not.

## Root cause

`_buildOverflowMenu()` already centralized popup mechanics, placement, icons,
keyboard focus, Escape and outside-click handling. Its callers still supplied
two unrelated capability models:

- snippets had a frozen two-row array built inside `_appendArtifactCards`;
- tracked files had a richer four-row dynamic list inside `_buildFileOverflow`.

That meant the same `⋮` affordance had different first actions, different Save
wording and no continuation path on snippets. It also made promotion a one-way
UI transition: the original snippet trigger did not graduate when the snippet
became a tracked file.

## Repair

- Extract canonical tracked-file actions into `_fileOverflowItems(key)`.
- Keep `_buildFileOverflow()` as a thin wrapper over that shared action list.
- Add `_buildSnippetOverflow()` with the same workflow shape:
  1. `Open in a sheet` — full preview with line numbers;
  2. `Save as…` — download under a chosen local name;
  3. `Track as file…` — establish repository-relative identity, revisions,
     diffs and patch export;
  4. `Continue editing` — track if necessary, then stage the exact tracked
     revision through the canonical continuation pipeline.
- Do **not** show `Download patch` before a snippet has stable tracked identity.
- Make `_promoteSnippetToFile()` return the existing/new ledger entry so the
  caller can immediately continue the exact revision.
- Once promotion succeeds, the original snippet trigger resolves
  `_fileOverflowItems(entry.key)` and therefore becomes the exact Presented-file
  menu, including `Download patch` and live `Continue` / `Stop continuing` state.
- Normalize the T93 snippet-scroll harness summary to the canonical harness
  reporting format discovered by the full registered Node gate.

## Verification

- T97 menu-workflow parity: **16/16**;
- activity/latest-file neighbor: **203/203**;
- working-file binding neighbor: **144/144**;
- complete registered Node/UI harness plane: **169/169**;
- mutation catalogue metadata + unique anchors: **272/272**;
- bounded parallel mutation execution: **269/269 mutants caught**;
- targeted T97 mutation checks: **12/12**;
- JavaScript syntax: **GREEN**.

## Prevention

Sharing menu chrome is not enough. When two surfaces use the same disclosure
pattern, also centralize the user's workflow grammar: inspection first, save /
export next, lifecycle capability third, continuation last. Capability-specific
rows may differ, but equivalent actions should keep the same wording/order and
the same artifact should graduate to the richer capability list when its state
changes. Never expose patch semantics before stable path/revision identity.

## Maintenance closure

- maintenance core: **35/35**;
- `_sphinx_ext` family gate: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**.
