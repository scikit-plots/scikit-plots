# R173T34 — The file menu stays inside the panel body

Status: COMPLETE

Base: R173T33 share redaction parity, same package.

## The problem

`.ai-assistant-panel-changed-file-menu` was placed by CSS alone —
`inset-inline-end: 0; top: calc(100% + .25rem)` — which is correct only when
there happens to be room below and to the inline-end. A file row near the
bottom of the panel body, or a narrow panel, pushed the menu past the edge
where it was clipped or scrolled out of reach, and `Save as…`, `Download patch`
and `Continue editing` became unreachable for that row.

## Reused, not reimplemented

The panel already solves this. `_positionAnchoredPopupWithinPanelBody` places
the bubble action menu and the feedback popup: it tries each side, picks one
that fits, falls back to the side with the best proportional room, clamps both
axes, and corrects for a transformed or scaled panel.

The file menu is now its **third caller**, not a third implementation. A fresh
placement routine would have looked right in a normal panel and been wrong in a
scaled one — the correction that is easy to omit and hard to notice.

Two details the reuse required:

- the menu marks itself `data-open="true"`, which is what the routine reads;
- placement runs **after** insertion. The routine measures the rendered box,
  so positioning first would size it from nothing and clamp everything to the
  corner — a broken-looking menu rather than a mis-measured one.

## Following the row

A menu anchored to a row inside a scrolling body detaches from its trigger the
moment the reader scrolls — which is exactly when a long file list is being
read. It now repositions on panel-body scroll and window resize, and both
listeners are removed on close alongside the existing click and keydown ones.

The menu is also height-clamped (`min(60vh, 22rem)`) and scrolls internally,
with `overscroll-behavior: contain` so scrolling it does not scroll the
transcript behind it.

## Verification

- browser wrapper gate: **148/148** (90 assertions in the owning harness);
- architecture gates: **489/489**, two new mutants, both caught:
  `file-menu-placed-before-it-is-measured` and
  `file-menu-detaches-when-the-body-scrolls`.

## An ordering assertion that matched the wrong call

`placement runs after insertion` compared `indexOf` of the placement call
against `indexOf` of `appendChild`. There are **two** placement call sites —
one on open, one inside the reflow handler — and the reflow one is also after
`appendChild`, so deleting the insertion-time call left the assertion passing.

It now counts both call sites as well as ordering them. An `indexOf` comparison
on a string that occurs more than once compares whichever occurrence came
first, not the one the assertion is about.
