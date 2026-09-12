# R173T83 — PDF prepared-mode thumb optical alignment

Status: COMPLETE — SUPERSEDED BY R173T85

Base: R173T82 compact picker border continuity.

## Reported UI issue

In the prepared-PDF state (`aria-checked="true"`), the PDF toggle thumb reads slightly too far left inside its 34px track. The track itself and the unchecked/Print position are correct.

## Repair

The PDF checked-state transform moves from `translateX(16px)` to `translateX(17px)`. This is deliberately PDF-only and checked-state-only:

- track geometry stays 34px × 18px;
- thumb stays 14px × 14px at `left: 1px`;
- unchecked/Print position is unchanged;
- Panel, Copy, and Mic toggle geometry is unchanged;
- JavaScript mode/ARIA semantics are unchanged.

This is an optical correction, not a shared switch-token change. It supersedes only the visual parity conclusion in R173T42 for the checked PDF thumb; R173T42 remains historical evidence for the geometry investigation and forced-colours repair.

## Verification

- focused dependency-free PDF thumb optical regression;
- existing PDF layout parity regression;
- Panel/Copy neighbor regressions;
- centralized maintenance/review gates;
- packaged-byte replay before delivery.

## Supersession note

R173T85 found that the generic Mic checked-transform rule occurs later with equal specificity, so this 17px declaration was not the true computed-position authority. R173T85 removes the optical exception and centralizes checked travel through a shared token.
