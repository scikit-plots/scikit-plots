# R173T61 — A twelve-item menu fits the screen

Status: COMPLETE

Base: R173T60 quick model swap chevron, same package.

## Two causes, both in shared code

**No anchor.** `_positionFileMenuWithinPanelBody` clamps a menu against the
panel, but its `wrapperSelector` listed only the two artifact-row surfaces it
was written for. The model picker lives in the **footer**, so a menu opened
there matched nothing, was never clamped, and a twelve-model list ran straight
off the bottom of the screen.

The selector now covers every surface that opens one of these menus — artifact
rows, snippet rows, the picker wrapper and the preview title bar. Adding a
surface without adding it here is what produced the bug, so its mutant removes
the picker entry specifically.

**No bound.** Placement flips a menu above its trigger when there is no room
below, but a twelve-item list is taller than the space either way. The menu is
now bounded by the **viewport**, not the panel: `min(50vh, 20rem)` with
scrolling.

`vh` is the point. A panel-relative unit would let a maximized panel produce a
menu taller than the display that clips it — the screen is what cuts the list
off, so the screen is what has to bound it. `20rem` caps it on a tall desktop,
where half a viewport is a menu long enough to lose the trigger it belongs to,
and a coarse pointer gets `min(60vh, 24rem)`: a thumb scrolls a long list more
readily than it reaches a menu that has been flipped and clamped.

## A duplicate rule, merged

R173T34 had already given this class a `max-height`. The new bound was written
as a second block, so the cascade had to be read to know what it was — and the
harness's `.find` picked whichever came first. Merged into one rule, with an
assertion that the selector is defined once.

## Three test corrections

- The rule match was unanchored, so it also matched the indented coarse-pointer
  override inside its media query and reported two rules where there was one.
  Anchored to line start.
- `the bound is measured against the screen` rejected any `100%` in the rule,
  which caught an unrelated `max-width`. Narrowed to the height declaration.
- The mutant removed the whole declaration block including the closing brace,
  producing CSS the assertion could not read. Rewritten to neutralise the two
  declarations instead.

## Verification

- browser wrapper gate: **152/152** (99 assertions in the owning harness);
- architecture gates: **583/583**, two new mutants, both caught:
  `long-menu-runs-off-the-screen` and `footer-menu-has-no-anchor`.
