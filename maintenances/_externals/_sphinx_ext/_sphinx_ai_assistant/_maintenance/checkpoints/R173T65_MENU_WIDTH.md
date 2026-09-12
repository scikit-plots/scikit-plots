# R173T65 — A menu as wide as its longest row

Status: COMPLETE

Base: R173T64 menu bounded by the panel, same package.

## `width: max-content`, with the ceiling it needs

A fixed width padded short menus and truncated the long labels in the model
list — the one place where the label *is* the information. `max-content` sizes
the menu to its longest row instead.

On its own that trades one fault for another: a single long label makes the
menu wider than the panel it belongs to. The ceiling comes from the placement
routine, computed from the **same panel-and-viewport bounds that already limit
the height**, rather than from a constant. A CSS-only cap would have to guess a
panel width, and this panel is resizable.

## Two ordering details

**Constrained before measured.** The rect read in the placement routine drives
every later calculation; measuring an unconstrained `max-content` menu means
placing a box the menu will never have. The `max-width` is written first, and
its mutant moves it after the measurement.

**Reset before recomputed.** `maxWidth` is cleared at the top alongside
`maxHeight`, so a menu reopened after the panel is widened is not still bounded
by the old panel.

## Two rows that would otherwise decide the width

With `max-content` the widest row wins, so:

- rows are `width: 100%` and fill the menu rather than sizing it;
- the secondary **hint wraps**. Left non-wrapping it would make the menu as
  wide as a sentence of explanatory text rather than as wide as the labels a
  reader is choosing between — a menu sized by its own footnotes.

## Verification

- browser wrapper gate: **152/152** (117 assertions in the owning harness);
- architecture gates: **593/593**, two new mutants, both caught:
  `menu-width-measured-before-it-is-constrained` and
  `menu-hint-decides-the-menu-width`;
- a 900px-wide menu in a 380px panel is asserted to stay pinned inside it.
