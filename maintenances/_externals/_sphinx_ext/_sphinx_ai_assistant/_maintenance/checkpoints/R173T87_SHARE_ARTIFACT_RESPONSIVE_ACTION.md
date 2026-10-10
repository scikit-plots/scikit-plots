# R173T87 — Share artifact responsive action layout

Status: COMPLETE

Base: R173T86 workspace tab visual parity.

## Reported UI issue

Managed Share artifacts were acceptable at wide widths but compressed their
metadata too aggressively at narrow panel widths because action buttons stayed
on the same flex line. Global rows with four lifecycle actions were the worst
case.

## Root cause

Two layout defects combined:

1. `.ai-assistant-conv-share-artifact-text` was declared twice. The first rule
   set `flex: 1 1 12rem`, but the later `flex:1` shorthand reset the basis to
   `0%`, so the description collapsed before actions wrapped.
2. Every action button was a direct child of the artifact row. Flexbox could
   wrap buttons individually, but could not express the intended responsive
   structure: metadata first, action group second. The old fallback also used a
   viewport media query rather than the actual resizable Share surface width.

## Repair

- Give metadata one canonical `flex: 1 1 12rem` declaration.
- Render all lifecycle controls inside
  `.ai-assistant-conv-share-artifact-actions`.
- Make `.ai-assistant-conv-share-artifacts` an inline-size container.
- At `<=30rem` container width, stack metadata above the action group.
- Keep three-action rows wrapping naturally.
- At `<=21rem`, rows with four-or-more actions use a two-column grid, giving
  Global links a stable 2x2 layout instead of a squeezed description or lone
  trailing button.
- Preserve all Copy/Open/Check status/Revoke/Forget behavior.

## Verification

- T87 responsive layout contract: **20/20**;
- Share conversation contract: **147/147**;
- provider artifact lifecycle UI: **47/47**;
- feedback workspace neighbor: **19/19**;
- all registered Node/UI harnesses: **161/161**;
- mutation catalogue: **467/467**;
- maintenance core: **35/35**;
- family maintenance: **2/2 GREEN**;
- AI maintenance: **GREEN (repository)**;
- JavaScript syntax: GREEN.
