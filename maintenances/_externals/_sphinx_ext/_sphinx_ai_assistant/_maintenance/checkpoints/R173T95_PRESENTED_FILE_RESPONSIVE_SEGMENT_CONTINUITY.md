# R173T95 — Presented-file responsive segment continuity

Status: COMPLETE

Base: R173T94 Presented-file segmented-control parity.

## Reported UX issue

At responsive widths around 560×640 and small-phone / iPhone-SE-class panels,
the Presented-file row could still look reordered even though R173T94 made the
DOM and desktop geometry correct. The intended invariant remains:

```text
[ icon + filename/meta | Download ] [ ⋮ ]
```

On a tighter surface the Download label may compact to its icon, but the
Download segment must never claim the whole segmented group.

## Root cause

R173T94 removed the competing base-layout generations, but one old viewport
breakpoint survived from the period when Download was a standalone row:

```css
@media (max-width: 560px) {
    .ai-assistant-panel-changed-file-download { width: 100%; }
}
```

After Download moved inside `.ai-md-artifact-group`, that declaration changed
meaning completely. The trailing Download segment became 100% of the group;
because Preview is the flexible segment, it was forced toward zero width. The
result could visually resemble a separator at the left edge with squeezed file
content, especially at exactly the 560px breakpoint and narrower phones.

The Presented list and primary grid also lacked explicit `min-width: 0`, leaving
an additional min-content path for long filenames to push a small panel wider.

## Repair

- Remove all responsive `width:100%` authority from Presented-file Download.
- Keep the shared artifact group geometry identical at every width:
  flexible Preview, one separator, fixed trailing Download.
- Add `min-width:0` to the Presented-files list, card, and primary grid so long
  filenames may shrink/ellipsis instead of expanding the panel.
- Move Presented-files heading stacking from viewport width to the existing
  `ai-artifact-surface` container at `35rem`.
- Preserve T90's two-stage label policy: per-file Download text compacts at
  `26rem`; bulk footer labels retain text until `22rem`.
- Keep one DOM for desktop, tablet, and mobile; no mobile-only alternate row.

## Verification

- T95 responsive contract: **21/21**;
- T94 segmented parity neighbor: **18/18**;
- activity/latest-file preview neighbor: **202/202**;
- diff-stat neighbor: **35/35**;
- working-file binding neighbor: **143/143**;
- registered Node/UI harnesses: **165/165**;
- mutation catalogue metadata + unique anchors: **260/260**;
- deliberate mutation execution: **260/260 mutants caught**;
- four new T95 mutation controls: **4/4 caught**;
- maintenance core: **35/35**;
- family maintenance: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**;
- JavaScript syntax: **GREEN**.

## Prevention

Responsive CSS must be reviewed whenever a component changes structural
ownership. A rule that was correct when an action occupied its own row can
become destructive after that action moves inside a segmented control. For
resizable panel components, prefer component/container width over viewport
width, and keep `min-width:0` through every grid/flex ancestor that contains
ellipsisable filenames.
