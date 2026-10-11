---
title: "Generate one coverage matrix that doctor, docs and gallery all read"
status: open
kind: "docs-contract"
area: "scikitplot/cleanprompt"
discovered_during: "internal review 2026-10-10 (coverage/readiness diagnostics)"
release_note: "not-required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# Generate one coverage matrix that doctor, docs and gallery all read

## Summary

What is covered is spread over `kinds` (structural patterns), the entity
engines' canonical labels, pack field kinds and formats. Each surface states
its own subset by hand. The review recommends one generated matrix:
entity kind × surface (pattern, validator, field, entity engine) × locale ×
tier × default action × reversible × semantic risk.

## Why it matters

Hand-written lists drift (the README's command count did; `CP-025` and
`CP-099` were drifted constants). A matrix generated from the registries
cannot.

## Current evidence

`kinds`, `CANONICAL_LABELS`, `_catalog` field index, `doctor` — four sources,
no single view. After round 25, `doctor --ner` reports per-engine readiness;
the matrix would be its natural complement.

## Root cause / current understanding

Design gap.

## Expected behavior

`python -m scikitplot.cleanprompt kinds --matrix --format json` (or a new
`coverage` command) derived from the live registries and readiness; a static
copy generated into the user guide by a tool, with a drift test like
`test_user_guide_sync.py`.

## Affected paths and ownership

`_diagnostics.py`, `_cli.py`, `_catalog.py`, docs generator under `tools/` or
the maintenance plane.

## Constraints and non-goals

No import of optional engines to build it (readiness via
`_engines.engine_readiness`, metadata only by default).

## Edge cases to cover

Absent tiers, a pack kind that is also a pattern kind, custom packs.

## Proposed direction

Implement after `per-kind-action-policy.md`, so the action column is real.

## Verification / acceptance criteria

Matrix equals the registries in tests; docs copy regenerated with no diff.

## Documentation impact

A "Coverage" section in `how_it_works.rst`.

## Release-note promotion

Not required for the docs copy; the CLI addition would take an
`enhancement` fragment.
