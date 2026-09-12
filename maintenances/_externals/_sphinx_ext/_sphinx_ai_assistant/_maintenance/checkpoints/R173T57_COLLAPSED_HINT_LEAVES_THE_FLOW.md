# R173T57 — Collapsed, the hint stops costing a row

Status: COMPLETE

Base: R173T56 collapsed hint trailing edge, same package.

## The remaining waste

Collapsed the hint was one icon, and it still held a full-width flex line with
empty space beside it. Collapsing bought a narrower control, not a taller
transcript.

The pill now leaves the flow: absolutely positioned at the bottom-trailing
corner, with the transcript taking the width and the row's margins gone.

```
| …the answer continues to the end of this line.        [ 🎤 › ] |
```

## Anchored to the footer, not the panel

The pill sits at `bottom: calc(100% + 0.35rem)` of the **footer**. Anchored to
the panel instead, a multi-line draft growing the composer upward would slide
under it. Anchored to the footer's top edge, it rises with the composer and
never overlaps the input or the footnote line.

`inset-inline-end`, so a right-to-left interface mirrors from the same rule.

## The trade is paid, not hidden

A control floating over the transcript will sit on the last line of an answer.
So the transcript keeps `padding-bottom: 2.25rem` while the hint is collapsed —
smaller than the row it replaces, so the transcript gains both width and
height, but not free, and pretending it were would mean a pill that sometimes
covers the final line of an answer.

The pill also gains a surface and a shadow, because it now sits over text
rather than over panel background — with its own dark-theme ground and a
`ButtonText` border where shadows are discarded.

## Verification

- browser wrapper gate: **152/152** (30 assertions in the owning harness);
- architecture gates: **569/569**, two new CSS mutants, both caught:
  `collapsed-hint-still-holds-a-row` and `floating-hint-covers-the-last-line`.

## A first-match assertion, again

Two assertions failed against correct CSS because the selector
`[data-collapsed="true"]` now has **two** rules — R173T56's alignment and this
checkpoint's placement — and the regex matched the first. They now select the
rule by content (`position: absolute`) rather than by position in the file.

Sixth instance this run of the same shape. The lesson stands as filed: a match
across combined text finds whichever occurrence came first, not the one the
assertion is about.
