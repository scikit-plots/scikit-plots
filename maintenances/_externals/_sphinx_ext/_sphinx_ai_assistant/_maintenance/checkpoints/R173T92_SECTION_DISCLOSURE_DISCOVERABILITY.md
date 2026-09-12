# R173T92 — Answer-section disclosure discoverability

Status: COMPLETE

Base: R173T91 speak-toggle sticky-hover contrast.

## Reported UX issue

Expandable answer headings such as `Mechanism` were technically clickable but
looked too much like ordinary headings. Bold text and a small chevron did not
explain the interaction, especially to first-time users and touch users.

## Root cause

The disclosure relied on three weak signals:

1. sections default open, so their surrounding content makes the summary read
   like a normal heading;
2. the only persistent control cue was a muted chevron;
3. the stronger surface appeared only on `:hover`, which touch devices cannot
   rely on.

The chevron state was also non-standard: the source icon pointed down while
closed and rotated sideways while open. Conventional disclosure language is
right when closed, down when open.

## Repair

- Keep native `<details>/<summary>` as the state/accessibility authority.
- Give the summary a persistent, subtle theme-aware disclosure surface with a
  border, touch-readable height, hover/active progression and focus ring.
- Give the chevron its own small persistent badge and use conventional
  right-closed / down-open geometry.
- Add a visual-only action hint: `Hide section` while open, `Show section` while
  closed. The hint is `aria-hidden` because native details already announces
  expanded/collapsed state; CSS derives the visible copy directly from `[open]`
  so there is no second JavaScript state machine.
- Give light and dark themes separate surface mixes. Light mode uses a slightly
  darker neutral than the surrounding answer; dark mode uses a slightly lighter
  neutral for equivalent perceptual separation.
- Preserve forced-colors and reduced-motion behavior.

## Verification

- T92 disclosure discoverability contract: **22/22**;
- existing section/custom DOM neighbor: **45/45**;
- registered Node/UI harnesses: **162/162**, plus **2/2** discovery/target guards;
- mutation catalogue metadata + unique anchors: **250/250**;
- deliberate mutation execution: **247/247 mutants caught**;
- JavaScript syntax: **GREEN**.

## Prevention

An expandable section must advertise interaction before the user already knows
it is expandable. Do not make discoverability depend on hover, font weight, or
an ambiguous glyph. Use native disclosure semantics plus persistent visual
surface, conventional state geometry, and concise action-oriented copy.

## Maintenance closure

- maintenance core: **35/35**;
- `_sphinx_ext` family gate: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**.
