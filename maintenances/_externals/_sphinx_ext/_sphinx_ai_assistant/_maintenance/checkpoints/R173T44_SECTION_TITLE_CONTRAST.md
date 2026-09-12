# R173T44 — Section titles read as headings

Status: COMPLETE

Base: R173T43 line numbers everywhere, same package.

## The gap

`summary.ai-md-section-summary` set weight and size but **no colour**, so it
inherited the bubble's own `--pst-color-text-base`. A section title therefore
differed from the paragraph beneath it by `font-weight: 700` and 2% of size and
nothing else — not enough separation against a tinted bubble for a heading a
reader is meant to scan by.

## "Darker" is a direction, and it reverses

The request was for a darker title. That is the **light-theme** direction only.
On a dark bubble, darker text moves toward the background rather than away from
it, and the heading becomes the least readable text in the block — the opposite
of what was asked for, arrived at by taking the wording literally.

What the two themes share is the *intent* — more contrast than body text — not
the direction:

```css
:root            { --ai-section-title-color: color-mix(… text-base 86%, #000 14%); }
[data-bs-theme="dark"] { --ai-section-title-color: color-mix(… text-base 86%, #fff 14%); }
```

Written as one token per theme so the reversal is explicit rather than hidden
inside a colour literal, and mixed **from the theme's own text colour** rather
than set to a fixed value, so a site that retints its text keeps its titles in
the same family.

The mutant `section-title-darkens-on-a-dark-bubble` applies the literal reading
and is caught.

## Two details

The chevron stays at the muted colour: it is an affordance, not part of the
heading, and matching the title's weight made the row read as two emphasised
things. It brightens to the title's colour on hover, where it is the thing
being acted on.

`color-mix` results are discarded in forced-colours modes, so both fall back to
`CanvasText` rather than to whatever the mix last computed.

## Verification

- browser wrapper gate: **148/148** (45 assertions in the owning harness);
- architecture gates: **521/521**, two new CSS mutants, both caught:
  `section-title-darkens-on-a-dark-bubble` and
  `section-title-indistinguishable-from-body`.
