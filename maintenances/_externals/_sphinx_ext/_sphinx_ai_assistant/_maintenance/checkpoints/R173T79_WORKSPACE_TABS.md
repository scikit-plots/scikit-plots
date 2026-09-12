# R173T79 — Workspace tabs are not format tabs

Status: COMPLETE

Base: R173T78 chevrons share one treatment, same package.

## The stale blue edge

The feedback tablist borrows `.ai-assistant-conv-share-format-switcher` for its
visual tokens — and inherited its **layout** along with them.

R173T41 turned that switcher into an even `auto-fit` grid. That is right for
five interchangeable format choices and wrong here: two or three *named
sections* were stretched to equal fractions of the row, so a short label sat in
a wide cell with its active edge drawn across the whole of it. The indicator
looked detached from the thing it indicated.

This is R173T67's fault in a new place: one class serving two surfaces whose
needs diverged, and a change made for one silently reaching the other.

## Overridden, not abandoned

The shared class stays. The tokens are worth sharing; the layout is not.

The tablist switches the grid off and reads as a row: sections start where a
tablist starts, size to their labels (`flex: 0 1 auto`), and wrap rather than
clip a section name. The active edge then spans the label rather than an empty
fraction of the row.

The indicator is one inset edge drawn from `--ai-artifact-accent`, so it cannot
drift from the accent used everywhere else in the panel, and it is keyed on
`[aria-selected="true"]` rather than a parallel class — the glyph-and-state rule
from R173T51, applied to a tab.

## Verification

- browser wrapper gate: **153/153** (one new harness, 11 assertions);
- architecture gates: **642/642**, two new CSS mutants, both caught:
  `workspace-tabs-inherit-the-format-grid` and
  `workspace-tab-indicator-drifts-from-the-accent`.

The second mutant's anchor was first written with the indentation of the
forced-colours block rather than the top-level rule, so it matched nothing. The
uniqueness check reports zero matches the same way it reports several, which is
what caught it.
