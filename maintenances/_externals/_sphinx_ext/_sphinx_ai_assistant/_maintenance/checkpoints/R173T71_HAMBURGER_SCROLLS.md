# R173T71 — The hamburger menu scrolls

Status: COMPLETE

Base: R173T70 bounds that yield, same package.

## No bound at all

`.ai-assistant-panel-hamburger` had no `max-height` and no `overflow`. On a
small panel its lower entries simply ran past the bottom edge — nothing to
scroll, and no indication the entries existed. Both anchors (`data-anchor="left"`
and `"right"`) and the nested `data-more-open` submenu shared the fault.

## `100%` is the right unit here, and only here

Every other menu bound in this run is measured in `vh`, because those are
placed against the viewport and a panel-relative unit would let a maximized
panel produce a menu taller than the display.

This one is different, and the difference is worth stating: it is **absolutely
positioned inside the panel**, so a percentage resolves against the panel's own
box. It therefore tracks a resize with no JavaScript at all — the bound is
correct at every panel size by construction rather than by recomputation.

`min(calc(100% - 3.3rem - 0.75rem), 80vh)` keeps the `vh` term for the case the
percentage cannot see: a panel taller than the window, where the screen clips
what the panel does not.

`overscroll-behavior: contain` stops a flick in the menu scrolling the
transcript behind it, and a coarse pointer gets `85vh` for the same reason the
shared menus do.

## A duplicate rule, merged — third time

The bound was first appended as a second `.ai-assistant-panel-hamburger` block.
That is the same mistake as R173T46 and R173T61, and the same tell each time:
the harness assertion for "defined once" fails, or a regex matches the wrong
block. Merged into the base rule, with the count asserted.

Three occurrences is enough to name the habit: appending a rule is easier than
finding the existing one, and it is wrong every time in a stylesheet this size.

## Verification

- browser wrapper gate: **152/152** (62 assertions in the owning harness);
- architecture gates: **613/613**, one new CSS mutant caught:
  `hamburger-menu-runs-past-the-panel`.
