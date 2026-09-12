# R173T74 — The model chevron on a phone

Status: COMPLETE

Base: R173T73 footer three regimes, same package.

## Two failures that made each other worse

`.ai-assistant-panel-inline-picker-more` was 1.6rem with a transparent
background: a fine cursor target and a poor thumb one — and on touch there is
no hover to reveal that it is a control at all.

Hard to hit and hard to notice compound, because **a control you cannot see is
one you will not aim at**. Fixing only the size would have left a 44px target
nobody looks for.

## Size and resting visibility together

On `pointer: coarse`:

- **2.75rem (44px)** minimum in both axes — the smallest target the platform
  guidelines call comfortable, and the size a thumb hits without aiming;
- a **resting ground**, because hover cannot supply one on touch;
- a larger glyph, since 0.8rem is legible on a laptop at arm's length and not
  on a phone;
- the divider is dropped: at this size it is decoration that only narrows the
  target;
- the picker beside it grows to match, or the joined pair looks broken.

A `:active` state applies everywhere: on touch it is the only feedback there
is, since there is no hover state to precede the press.

## Browser-specific corrections

- `appearance: none` / `-webkit-appearance: none` — Safari and Firefox each
  apply their own button chrome, which renders the chevron at a different
  height from the picker it is joined to.
- `touch-action: manipulation` — removes the ~300ms tap delay on iOS Safari and
  Android Chrome, which reads as a control that did not respond.
- `-webkit-tap-highlight-color: transparent` — removes the grey flash that
  reads the same way.
- `forced-colors` gives it a `ButtonText` border, since the resting ground is
  discarded there.

Dark theme states its own resting colour and ground rather than relying on a
`--pst-color-surface` the host may never define — the R173T49 correction,
applied to a new control rather than found in it later.

## Verification

- browser wrapper gate: **152/152** (77 assertions in the owning harness);
- architecture gates: **627/627**, three new CSS mutants, all caught:
  `chevron-too-small-for-a-thumb`, `chevron-invisible-until-hovered`,
  `chevron-keeps-browser-button-chrome`.

The duplicate rule was merged before it shipped — fifth time this habit has
appeared, and the first time it was caught by the "defined once" assertion in
the same turn rather than a checkpoint later.
