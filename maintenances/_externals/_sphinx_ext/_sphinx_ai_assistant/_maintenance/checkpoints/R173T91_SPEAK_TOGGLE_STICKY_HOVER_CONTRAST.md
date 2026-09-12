# R173T91 — Speak-toggle sticky-hover contrast

Status: COMPLETE

Base: R173T90 artifact Download mobile compaction.

## Reported UI issue

On mobile/touch, the speak-hint disclosure button remained clickable after a
press but its chevron could become visually indistinguishable from the button
surface. The symptom could occur in either `aria-expanded="false"` or
`aria-expanded="true"` because those states share the same SVG and differ only
by rotation.

## Root cause

R173T89 repaired the resting state, but one higher-specificity interaction rule
still used `color: inherit` on `:hover` / `:focus-visible`. Some touch browsers
retain a sticky `:hover` state after a tap. That sticky state therefore
out-ranked the explicit coarse-pointer resting colour and replaced it with the
parent's inherited colour. On host themes where that colour is close to the
control surface, the SVG remained present and clickable but looked transparent.

A second token issue made the failure easier to trigger: the speak controls used
`--pst-color-on-background` as a background surface. In PyData-style token
semantics that `on-*` token is not a reliable surface authority and can resolve
to a value unsuitable as the control ground.

## Repair

- Keep the desktop resting muted colour, but make hover, focus-visible and
  active states use an explicit `--pst-color-text-base` foreground. Never use
  `color: inherit` for this touch-visible disclosure.
- Include `:active` in the same interaction contract so the pressed state does
  not momentarily fall through to native/host paint.
- Use `--pst-color-surface` for the speak banner and disclosure-button ground.
- Make the banner/mic SVG use the readable base-text foreground instead of
  assuming the host primary accent always contrasts with its surface.
- Preserve `stroke: currentColor`, touch sizing, dark-mode fallbacks,
  forced-colors authority, and `aria-expanded`-driven rotation.
- Do not add separate true/false colour rules: expanded state changes geometry
  (rotation) only, not visibility paint.

## Verification

- T91 sticky-hover/state-visibility contract: **19/19**;
- T89 resting mobile visibility neighbor: **14/14**;
- speak collapse/expand behavior: **37/37**;
- registered Node/UI harness plane: **161/161**;
- mutation catalogue metadata + unique anchors: **246/246**;
- deliberate mutation execution: **243/243 mutants caught**;
- JavaScript syntax: **GREEN**.

## Prevention

A mobile control must be readable in the entire tap lifecycle, not only at
rest. On touch-capable browsers, treat `:hover` as potentially sticky after a
press. Interaction selectors with greater specificity must repeat an explicit
contrast-safe foreground rather than reverting to inheritance. Visual state
selectors such as `aria-expanded` should not own colour unless the state truly
changes semantic emphasis.

## Maintenance closure

- maintenance core: **35/35**;
- `_sphinx_ext` family gate: **2/2 GREEN**;
- AI maintenance drift: **GREEN (repository)**;
- independent AI review: **PR_READY / release ELIGIBLE**.
