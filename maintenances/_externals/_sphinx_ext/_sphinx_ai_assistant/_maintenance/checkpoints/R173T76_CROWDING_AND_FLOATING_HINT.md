# R173T76 — The chevron stops crowding the mic; the hint floats in both states

Status: COMPLETE

Base: R173T75 speak row matched ends, same package.

## 1. Two thumb targets, 0.1rem apart

R173T74 gave the model chevron a 44px target on coarse pointers. The footer
actions row has `gap: 0.1rem` and did not wrap, so the enlarged control was
pushed into the microphone beside it: two adjacent thumb-sized buttons with
almost nothing between them, and the reader aiming at one hits the other.

The gap now grows with the targets, the row may wrap, and the mic keeps a 44px
target of its own with `flex: 0 0 auto` so it is never shrunk to make room.
Wrapping is the right failure here — two rows of reachable controls beat one
row of overlapping ones.

## 2. The expanded hint sat on an opaque row

Collapsed, the row was lifted out of the flow (R173T58) and the space beside
the pill showed the conversation through it. Expanded it was still an in-flow
row, so that same space was panel background: opaque, with nothing behind it.

Both states are now zero-height with their contents lifted. The difference
between them is what is drawn, not whether the row occupies a line — and the
transcript reserves the room in both, since a floating control that can cover
the last line of an answer costs the same either way.

Two duplicate rule pairs were merged in the process (`speak-row`, and the
`:has()` body reservation), which would have been the sixth instance of that
habit.

## The gate that was not guarding anything

`expanded-toggle-has-no-ground` reported as uncaught, and the CSS was correct.
Line-surgery on the harness three checkpoints earlier had deleted the entire
R173T75 assertion block — the mutant survived because **nothing was testing
it**, not because the test was weak.

Restored, and bounded to the rule body: an unbounded lazy match runs past the
closing brace into a later rule setting the same token and would pass with the
toggle's own declaration removed.

The lesson is about the repair, not the rule: deleting harness lines by pattern
removes whatever else happened to match. Three assertions vanished silently and
the only reason it surfaced was a mutant that stopped dying.

## Verification

- browser wrapper gate: **152/152** (36 assertions in the owning harness);
- architecture gates: **631/631**, two new CSS mutants:
  `enlarged-chevron-crowds-the-mic` and `expanded-hint-sits-on-an-opaque-row`;
  one superseded mutant removed and three assertions rescoped to the rules
  their mutants actually change.
