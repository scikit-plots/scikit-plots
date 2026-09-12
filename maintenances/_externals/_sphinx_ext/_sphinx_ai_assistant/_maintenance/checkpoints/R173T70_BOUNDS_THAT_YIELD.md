# R173T70 — Bounds that yield rather than clip

Status: COMPLETE

Base: R173T69 wheel escapes the code block, same package.

Two menus, over-constrained in two different ways, reported from one live DOM:

```
max-height: 181.906px
```

## 1. The bubble menu was bounded by the transcript

`_positionAnchoredPopupWithinPanelBody` hardcoded its boundary to
`#ai-assistant-panel-body`. That menu carries the **model list** — the longest
in the panel — so on a short body it was clamped to whatever height the
transcript happened to have and its last entries fell off the bottom.

The boundary is now a caller option, defaulting to the transcript as before.
The bubble menu passes `.ai-assistant-panel`, so it may use the panel's full
height. A menu whose items cannot be reached is worse than one that overlaps
the composer.

## 2. R173T66's composer floor could squeeze

That checkpoint made the composer a floor for trigger-anchored menus, and it
reads better — when there is room. On a short panel it left too little height
and cut rows off, which is the same fault as (1) arriving from a change made
for readability.

The floor now applies only while it leaves a usable menu (200px). Below that it
is dropped and the menu may cover the composer.

The ordering is deliberate: **keeping clear of the composer is a preference,
showing every row is a requirement.** When they conflict the preference yields.
Overlapping an input the reader is not currently using beats hiding choices
they are trying to make.

## Driven, not read

The harness places a menu in a tall panel and a short one and asserts the
opposite outcomes: with room the menu still stops above the composer; without
it, the floor is dropped rather than rows being cut. Two mutants make each half
fail — an unconditional floor, and a bubble menu without its widened boundary.

## Verification

- browser wrapper gate: **152/152** (143 assertions in the owning harness);
- architecture gates: **611/611**, two new mutants and one retargeted:
  `composer-floor-squeezes-the-menu`,
  `bubble-menu-clamped-to-a-short-transcript`, and
  `footer-trigger-bounded-above-its-own-button` whose anchor grew the new
  condition.
