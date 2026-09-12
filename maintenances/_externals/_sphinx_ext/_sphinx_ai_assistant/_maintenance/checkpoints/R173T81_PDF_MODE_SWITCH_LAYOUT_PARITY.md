# R173T81 — PDF mode switch layout parity

Status: COMPLETE

Base: wide `_sphinx_ext` review-v3 candidate.

## Observed UI defect

The AI-panel trigger switch was the correct reference: a fixed 48px trailing
gutter, zero extra horizontal padding, and a flex-centered 34×18 track. The PDF
row had drifted from that family contract even though its JavaScript state
semantics were already correct.

Two CSS defects were present:

- `.ai-assistant-pdf-row` reserved three grid columns for only two children
  (action + mode switch);
- `.ai-assistant-pdf-mode-switch` retained stale `min-height` and horizontal
  padding instead of the established Panel/Copy 48px gutter geometry.

## Repair

The PDF row now uses `minmax(0, 1fr) auto`, and the PDF switch uses the same
container geometry as the working Panel and Copy controls: 48px width, no
extra padding/min-height, transparent neutral surface, pointer cursor, and
explicit flex centering.

No PDF JavaScript behavior changed. `aria-checked=true` still means prepared
PDF (`url`) mode, labels/tooltips remain synchronized, and print-only pages
remain disabled rather than pretending a second method exists.

## Regression evidence

A dependency-free static harness
`test_ai_assistant__pdf_toggle_layout.mjs` locks the geometry and JS-state
contract.

- PDF layout/parity harness: **16/16**
- Panel-trigger neighboring behavior: **100/100**
- Copy-mode neighboring behavior: **13/13**
- maintenance-core suite: **35/35**
- `_sphinx_ext` family maintenance gate: **GREEN (2/2)**
- AI maintenance drift checker: **GREEN (repository)**
- AI independent review: **PR_READY / release ELIGIBLE**

This checkpoint changes runtime CSS and adds one focused regression test; it
does not modify the PDF state machine or the YouTube runtime.
