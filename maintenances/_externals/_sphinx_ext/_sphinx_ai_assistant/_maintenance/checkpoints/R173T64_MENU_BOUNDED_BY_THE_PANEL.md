# R173T64 — Menus stay inside the panel

Status: COMPLETE

Base: R173T63 containing block correction, same package.

## The remaining edge

R173T63 clamped menus to the **viewport**. A narrow panel inside a wide window
has plenty of viewport beside it, so a menu could still spill onto the page — a
list belonging to a control in the panel, drawn over the documentation behind
it, reading as part of neither.

The bound is now the **intersection** of the panel's rect and the viewport's.
Both halves earn their place:

- the **panel** is what the menu should stay inside, so it looks attached to
  the control that opened it;
- the **viewport** still matters, because a panel taller than the window
  extends past the screen, and the screen is what clips.

Asserted in both directions: a narrow right-hand panel keeps the menu inside
its own edges, and a panel whose box runs from `-200` to `1400` still produces
a menu inside the 911px screen.

## One guard worth naming

`if (pr.width > 0 && pr.height > 0)`. A `display: none` or zero-height ancestor
reports an empty rect, and intersecting with it collapses the bounds to nothing
— every menu clamped into a single point, which is worse than not clamping at
all. Its mutant removes the check.

## Verification

- browser wrapper gate: **152/152** (109 assertions in the owning harness);
- architecture gates: **589/589**, two new mutants caught
  (`menu-spills-outside-the-panel`, `hidden-panel-clamps-the-menu-to-a-point`)
  and two retargeted, whose anchors moved when the clamps stopped reading the
  raw viewport.

The placement routine is now driven by five separate geometry cases across the
harness — shifted containing block, narrow panel, oversized panel, and the
two-direction offset check — rather than inspected. Three positioning bugs in
this area (T57, T62, T63) were all cases where reading the code agreed with the
author and the arithmetic did not.
