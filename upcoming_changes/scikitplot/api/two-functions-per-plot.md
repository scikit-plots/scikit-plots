---
title: "api: ROC and precision-recall plots exist twice with different signatures"
status: open
kind: "api"
area: "scikitplot/api"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.api"
towncrier_type: "api"
towncrier_fragment: ""
---

# api: ROC and precision-recall plots exist twice with different signatures

## Summary

`scikitplot.api` exports `plot_roc` and `plot_roc_curve`, `plot_precision_recall` and `plot_precision_recall_curve`. The `*_curve` versions take `title` positionally and `curves=(...)`; the others take keyword-only `class_index`, `class_names`, `multi_class`.

## Why it matters

Two entry points for one plot double the documentation and tests, and users cannot tell which is current.

## Current evidence

- `python -m scikitplot api scikitplot.api` lists `plot_roc(y_true, y_probas, *, class_index=None, class_names=None, multi_class=None, ...)` and `plot_roc_curve(y_true, y_probas, title='ROC Curves', curves=('micro', 'macro', 'each_class'), ...)`.

## Root cause / current understanding

Legacy signatures kept next to the redesigned ones.

## Expected behavior

One function per plot (the keyword-only design); the legacy names removed.

## Affected paths and ownership

- `scikitplot/api/metrics/`

## Constraints and non-goals

API removal (campaign: no backward compatibility) with fragment and API reference.

## Edge cases to cover

- Galleries and docs examples using the old names

## Proposed direction

Remove the `*_curve` variants after moving any feature they have that the others lack.

## Verification / acceptance criteria

- One public function per plot in `scikitplot.api`

## Documentation impact

API reference, gallery.

## Release-note promotion

Required: `scikitplot.api` / `api`, one bullet, when the change lands.
