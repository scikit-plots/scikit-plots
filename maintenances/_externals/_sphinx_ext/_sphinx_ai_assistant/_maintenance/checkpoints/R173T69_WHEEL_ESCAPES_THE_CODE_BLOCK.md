# R173T69 — The wheel escapes the code block

Status: COMPLETE

Base: R173T68 trigger closes its menu, same package.

## Root cause: an overflow axis promoted behind our back

```css
overflow-y: visible;
overflow-x: auto;     /* long lines */
```

CSS promotes a `visible` axis to `auto` when the other axis is not `visible`.
So the code block was a scroll container on **both** axes — silently, and
**only for files whose lines happen to be long**, which is why it looked like a
long-line bug rather than a stylesheet one.

A scroll container that cannot scroll vertically still consumes the wheel, and
`overscroll-behavior: contain` on the sheet (R173T46) stopped the gesture
chaining out. With the pointer over a wide code block, the page would not
scroll at all.

Two rules, one symptom: neither is wrong alone, and the pairing was never
written down.

## The fix

`overflow-y: hidden` on the block. It clips nothing — the block has no height
limit, so its box already fits its content — it only stops the promotion. The
sheet, or the preview body, remains the vertical scroller as R173T46 intended.

And the preview overlay's sheet releases the wheel: `overscroll-behavior: auto`
there, because that sheet does not scroll — the overlay body does. Containment
belongs to the sheets that own a scrollbar, and the harness asserts both halves
so one cannot be changed without the other being looked at.

## Verification

- browser wrapper gate: **152/152** (185 assertions in the owning harness);
- architecture gates: **607/607**, two new CSS mutants, both caught:
  `code-block-swallows-the-wheel` and `preview-sheet-traps-the-wheel`.

## Two mutant repairs

`preview-sheet-traps-the-wheel` first *inserted* a containing declaration while
the later `auto` still won, so the rule behaved correctly and the mutant
survived. It now replaces the winning declaration.

`numbered-code-scrolls-itself` broke on its anchor: the explanatory comment was
inserted between the two lines it matched, so the find returned zero. Anchored
past the comment. The comment that documents a fix can break the mutant that
guards it — worth remembering alongside the rule about comments defeating
assertions.
