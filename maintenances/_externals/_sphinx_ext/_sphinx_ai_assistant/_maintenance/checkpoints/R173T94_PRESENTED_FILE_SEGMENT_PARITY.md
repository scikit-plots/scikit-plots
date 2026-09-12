# R173T94 — Presented-file segmented-control parity

Status: COMPLETE

Base: R173T93 inline snippet scroll handoff.

## Reported UX issue

A Presented-file row could render with the divider and artifact content feeling
out of order or visually detached, even though a normal generated-artifact card
looked correct. The intended shape is one segmented artifact control:

```text
[ icon + filename/meta | Download ] [ ⋮ ]
```

not a collection of independently styled file actions.

## Root cause

The JavaScript had already moved Presented files onto `_buildArtifactSegmentGroup`,
but the stylesheet still carried several older generations of the same component:

- an original `preview | download` row;
- later `preview | download | patch` / four-control grids;
- a `preview | download | save-as` grid;
- a `preview | download | overflow` grid;
- the final `[segmented preview/download group] | overflow` grid.

Those selectors shared the same component names and could win at different points
in the cascade. The old regression suite even asserted both the obsolete
three-column and final two-column layouts, preserving the contradiction.

Presented-file Download also had only its behavior-specific class instead of the
normal artifact's base `ai-md-artifact-download-label` class, so the two visually
identical controls still depended on parallel CSS.

## Repair

- Keep exactly one Presented-file primary-row authority:
  `grid-template-columns: minmax(0, 1fr) auto` for `[artifact group] [overflow]`.
- Remove dead CSS for direct patch/continue/save-as/secondary/more controls that
  no longer exist in the runtime DOM.
- Give Presented-file Download both classes:
  `ai-md-artifact-download-label ai-assistant-panel-changed-file-download`.
  The base class owns visual geometry; the changed-file class remains a
  behavior/state hook.
- Strengthen `.ai-md-artifact-group` itself so its primary card explicitly owns
  remaining width with `flex: 1 1 auto; min-width: 0; width: auto`, while the
  Download segment is `flex: 0 0 auto`.
- Preserve DOM order in the shared builder as Preview → separator → Download.
- Keep diff stats, type badge, latest-revision state, Download, overflow menu,
  patch/save-as/open/continue capabilities, and accessible names unchanged.

## Verification

- T94 segmented parity contract: **18/18**;
- activity/latest-file preview neighbor: **202/202**;
- diff-stat neighbor: **35/35**;
- working-file binding neighbor: **143/143**;
- raw-body dedup neighbor: **21/21**;
- registered Node/UI harnesses: **164/164**;
- mutation catalogue metadata + unique anchors: **259/259**;
- deliberate mutation execution: **256/256 mutants caught**;
- five new T94 mutation controls: **10/10** architecture assertions/executions;
- JavaScript syntax: **GREEN**.

## Prevention

When two surfaces claim to share a component, share both its DOM primitive and
its layout authority. A compatibility/behavior class may add state, but it must
not recreate geometry already owned by the base component. Remove superseded
selectors instead of relying on a later override, and never keep tests that
simultaneously require contradictory layout generations.
