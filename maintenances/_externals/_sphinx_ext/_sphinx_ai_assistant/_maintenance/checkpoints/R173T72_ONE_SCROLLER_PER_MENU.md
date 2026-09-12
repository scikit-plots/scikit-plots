# R173T72 — Two scrollers, nested, and no model reachable

Status: COMPLETE

Base: R173T71 hamburger scrolls, same package.

## Root cause

`.ai-assistant-panel-bubble-model-list` carried its own
`max-height: min(50vh, 18rem)` and `overflow-y: auto` (R173T35). That is right
for a list standing alone and wrong for one nested in a menu that is itself
bounded.

On a short panel the outer menu was clamped to roughly 180px while this list
claimed up to 288px. Two scroll regions, one inside the other:

- the **inner** list took the wheel, so the outer menu could not be scrolled to
  reveal it;
- the **outer** menu clipped the inner list, so the list could not scroll
  itself into view.

Neither could move, and no model was reachable at all — which is why it looked
like the list had no scroll rather than too much of it.

## One scroller wins

The list defers: `max-height: none`, `overflow-y: visible`,
`overscroll-behavior: auto`. The outer menu is the single scroller, and it is
also the element the placement routine measures and bounds — so the thing that
scrolls and the thing that is bounded are now the same element.

This is R173T46's rule arriving in a second place. There it was a gutter beside
a block that already scrolled; here a list inside a menu that already scrolled.
Worth stating generally: **when a bounded container gains a bounded child, one
of the two bounds is wrong.**

## The bound also has to be recomputed

Placement ran when the menu opened, with the list collapsed, and wrote a
`max-height` for that content. Expanding the list made the menu taller than the
bound it had been given, so the new rows sat below a clamp computed before they
existed.

The disclosure now re-places its owning menu. A control that changes a bounded
element's height has to tell whatever computed the bound.

## Verification

- browser wrapper gate: **152/152** (65 assertions in the owning harness);
- architecture gates: **617/617**, two new mutants, both caught:
  `model-list-keeps-its-own-scroller` and `expanded-list-outgrows-a-stale-bound`.

The duplicate `.ai-assistant-panel-bubble-model-list` rule was merged — fourth
occurrence of that habit, and the assertion for "defined once" is now part of
this harness too.
