# R173T82 — Compact picker border continuity

Status: COMPLETE

Base: wide `_sphinx_ext` UI/UX R173T81 candidate.

## Observed UI defect

At very small panel widths the footer model control contracts to the compact
representation `[model icon + effort] | [quick-model chevron]`. The primary
half removed its border in compact mode, while the chevron sibling retained
its top/right/bottom outline. A later joined-edge rule restored only the
middle separator, leaving the left/top/bottom outline absent around the
icon/effort half.

The result was a visually broken segmented control: the chevron appeared to
own an isolated three-sided box while the primary icon/effort section floated
without the matching outer edge.

## Repair

Compact mode now keeps the same 1px border token as the wide model picker.
The existing segmented-control rules remain authoritative:

- the primary half owns the left/top/bottom outer outline and the single
  shared middle separator;
- the chevron half owns the top/right/bottom outer outline and drops only its
  inline-start border;
- the legacy pseudo-element divider remains disabled, so the middle hairline
  is never doubled;
- compact representation remains panel-fit-owned, and the effort badge stays
  visible and bounded.

No JavaScript behavior or model/effort state changed.

## Regression evidence

A dependency-free static harness
`test_ai_assistant__compact_picker_border_parity.mjs` locks the compact
segmented-control geometry.

- compact border continuity: **12/12**
- footer panel-width fit neighbor: **22/22**
- compact/mobile effort neighbor: **13/13**
- quick-model / chevron family neighbor: **91/91**
- maintenance-core suite: **35/35**
- `_sphinx_ext` family maintenance gate: **GREEN (2/2)**
- AI maintenance drift checker: **GREEN (repository)**
- AI independent review: **PR_READY / release ELIGIBLE**

This checkpoint changes runtime CSS and adds one focused regression test. It
does not modify PDF behavior, model selection semantics, effort state, or the
YouTube runtime.
