# R173T66 — The composer is a floor, not empty space

Status: COMPLETE

Base: R173T65 menu width, same package.

## Borrowed from a menu that already got it right

`.ai-assistant-panel-bubble-action-more-menu` has always been bounded this way:
it clamps within `.ai-assistant-panel-body`, which ends where the footer
begins. Nothing about that was written down as a principle, but it is why that
menu reads better than the others.

The trigger-anchored menus — file rows, snippet rows, the preview title bar —
clamped to the panel, which includes the footer. So one opened from low in the
transcript could be drawn over the composer: a menu that looks like it belongs
to the input, covering the draft about to be sent.

The composer is now a floor for them too.

## Only for triggers above it

`if (fr.height > 0 && t.bottom <= fr.top)`. The model picker's own trigger
lives **inside** the footer, and a bound above its own button would leave that
menu nowhere to open. Its mutant applies the floor unconditionally.

Both cases are driven: a menu opened from the transcript is asserted to clear
the composer entirely, and one opened from the picker is asserted still to be
placed and still inside the panel.

## A guard the fixtures found

`panel.querySelector` was called without checking it exists. The harness
fixtures model a panel with no footer, and that is not an artificial case —
it is what a panel looks like before the footer is built. Guarded rather than
worked around in the test.

## Verification

- browser wrapper gate: **152/152** (123 assertions in the owning harness);
- architecture gates: **597/597**, two new mutants, both caught:
  `menu-covers-the-composer` and `footer-trigger-bounded-above-its-own-button`;
- one mutant retargeted: the trigger measurement moved earlier in the routine
  so the footer test could use it, and its anchor moved with it.
