# R173T75 — Both ends of the speak row read the same

Status: COMPLETE

Base: R173T74 chevron on touch, same package.

## The inconsistency was in the expanded state, not the collapsed one

Collapsed, R173T57 gave both the pill and the toggle a surface, and that pair
reads correctly: two controls on a transparent row.

Expanded, the banner was a filled pill and the toggle was bare and transparent.
One end of the row looked like a control and the other like a glyph resting
beside it — the same two elements presenting as two different kinds of thing
depending on width.

The toggle now carries the banner's own `--pst-color-on-background` at every
width, and stretches to the banner's height when expanded so the pair sits on
one baseline rather than a pill beside a smaller circle.

## What did not change

The **row** still draws nothing. The space around the two controls is
transparent in both states, which is what made the collapsed form work and is
asserted here so a future surface on the row cannot creep in: the controls are
drawn, the row is not.

The collapsed override still wins where it applies — it sets its own surface
and shadow because there it sits over transcript text rather than over panel
background (R173T57).

## Verification

- browser wrapper gate: **152/152** (43 assertions in the owning harness);
- architecture gates: **629/629**, one new CSS mutant caught:
  `expanded-toggle-has-no-ground`.
