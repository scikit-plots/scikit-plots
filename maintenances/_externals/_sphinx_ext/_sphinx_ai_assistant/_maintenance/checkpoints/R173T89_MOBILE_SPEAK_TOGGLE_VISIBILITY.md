# R173T89 — Mobile speak-toggle resting visibility

Status: COMPLETE

Base: R173T88 mobile model action menu trigger anchoring.

## Reported UI issue

The small chevron in `.ai-assistant-panel-speak-toggle` was visible on desktop
but could become effectively invisible on mobile/touch, especially on host
surfaces whose muted text token is close to the control background.

## Root cause

The control had no touch-specific resting-visibility contract. Desktop pointer
interaction could recover contrast through `:hover` / `:focus-visible`, while a
touch device has no dependable hover state.

There was also a direct cascade contradiction in the base rule: it declared the
same `background-color` as the adjacent speak banner, then later used
`background: transparent`, which reset that color. The historical regression
only checked that `background-color` text existed and therefore missed the
shorthand override.

## Repair

- Remove the self-cancelling `background: transparent` shorthand so the
  intended resting surface actually paints.
- Reset native mobile button appearance with `appearance: none` and
  `-webkit-appearance: none`.
- Bind the SVG explicitly to the button with `color: inherit` and
  `stroke: currentColor` and render it as a block.
- On hoverless/coarse-pointer devices, use the base text-color authority rather
  than the muted token and enlarge the chevron to `1rem`.
- Give dark touch mode an explicit readable fallback while preserving
  forced-colors `ButtonText` authority.
- Preserve `aria-expanded`-driven rotation and all collapse/expand JavaScript.

## Verification

- focused mobile speak-toggle visibility: **14/14**;
- speak-hint collapse/expand neighbor: **37/37**;
- registered Node/UI harness plane: **162/162**;
- mutation catalogue: **479/479**;
- JavaScript syntax: GREEN;
- maintenance core: **35/35**;
- family maintenance: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**.

## Prevention

A touch-only affordance must have a readable resting state without requiring
hover. CSS regressions must validate effective declarations, including
shorthand properties that can reset an earlier longhand; merely asserting that
a desired declaration appears in source is not sufficient.
