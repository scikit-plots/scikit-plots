# R173T62 — Menus sit against the button that opened them

Status: COMPLETE

Base: R173T61 menu fits the screen, same package.

## Why it landed in the middle

R173T61 added the picker to the placement routine's anchor list, and the menu
still appeared centred in the panel.

The routine is `_positionAnchoredPopupWithinPanelBody`: it measures against
`.ai-assistant-panel-body`. A trigger in the **footer** sits outside that box,
so no side "fits", and the fallback clamped the menu into the body's own
rectangle — the middle of the panel, unattached to the control that opened it.
Adding an anchor to a routine that measures the wrong region could not fix it.

## Placed against the trigger, fixed to the viewport

`_positionMenuNearTrigger(menu, btn)` reads the trigger's rect, opens below or
above by whichever side has room, aligns to the trigger's trailing edge, and
clamps to the viewport with an 8px margin. When neither side fits the list, it
takes the larger and bounds itself to it, so a long menu scrolls instead of
overflowing.

**`position: fixed` is the load-bearing choice.** These triggers live in four
different subtrees — artifact rows, the composer footer, the preview dialog —
with different overflow and transform ancestors. R173T58 is what assuming a
containing block costs: a pill positioned against an ancestor it did not have,
pushed clean off the panel. Fixed coordinates have one containing block, the
viewport, which is also the thing that clips the menu.

`data-placement` records the chosen side, so styling can follow the geometry
rather than guessing it.

## Four test corrections

- Two assertions were **duplicated**: the block I replaced existed twice, so
  the old copies kept asserting the removed function. Deleted rather than left
  passing against nothing.
- The call-count assertion still named the old function.
- R173T34's mutant anchored on it too, and was retargeted to the new call.
- `a long menu scrolls` matched `min(60vh` — the coarse-pointer override — so a
  mutation inflating the base bound to `500vh` left a menu taller than any
  screen while the assertion still passed. It now pins the exact bound.

## Verification

- browser wrapper gate: **152/152** (97 assertions in the owning harness);
- architecture gates: **585/585**, with `menu-lands-in-the-middle-of-the-screen`
  and `menu-placed-against-an-assumed-containing-block` both caught.
