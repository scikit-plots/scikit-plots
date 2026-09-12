# R173T56 — The collapsed hint sits where the thumb is

Status: COMPLETE

Base: R173T55 speak hint collapse, same package.

## The two states want opposite alignment

Expanded, the row is **text to read**, and text starts where the reader's eye
does — leading edge, unchanged.

Collapsed, it stops being text and becomes a **single small target whose only
purpose is to be pressed**. On a phone the trailing edge is where the thumb
already rests; the leading edge is the far corner for a right-handed
one-handed grip, and the collapsed form is exactly the state a reader on a
small screen will be in.

```
expanded    [ 🎤  Speak with your assistant   Space   ‹ ]
collapsed   [                                      🎤 › ]
```

## Two details worth stating

`justify-content: flex-end`, not `right`. A right-to-left interface gets the
mirrored placement from the same rule rather than needing a second one — the
thumb argument is about the reading direction's trailing edge, not about the
physical right.

**Only the alignment moves.** The banner still precedes the toggle in the DOM
in both states, so tab order and screen-reader reading order are identical
whether the hint is open or closed. A collapse that reorders the document to
achieve a visual effect makes the two states behave differently for anyone not
using a pointer, and the harness asserts the DOM order to keep that from
drifting.

## Verification

- browser wrapper gate: **152/152** (21 assertions in the owning harness);
- architecture gates: **565/565**, one new CSS mutant caught:
  `collapsed-hint-parks-in-the-far-corner`.
