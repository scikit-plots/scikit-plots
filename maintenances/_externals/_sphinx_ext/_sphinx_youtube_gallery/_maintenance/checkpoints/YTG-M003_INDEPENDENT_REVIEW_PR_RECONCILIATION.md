# YTG-M003 — Independent review and PR reconciliation

## Purpose

Add a maintenance-only review plane that can inspect the YouTube/Sphinx runtime family
package-by-package, run independent review lenses concurrently, and reconcile findings
without giving free-form agent output or review metadata execution authority.

## Contracts

- `MAINTENANCE.json` schema v3 requires `_maintenance/REVIEW.json`.
- Review profiles may select registered deterministic checks only; executable fields such
  as shell commands, Python snippets, environment overrides, scripts, or working
  directories are rejected.
- `_sphinx_youtube_gallery` reviews six package targets independently:
  `_sphinx_youtube_gallery`, `_sphinx_youtube_core`, `_sphinx_collection`,
  `_sphinx_gallery_grid`, `_sphinxcontrib_youtube`, and `_pydata_component_list`.
- `_sphinx_ai_assistant` has an independent profile and package review; family review
  reconciles only after subsystem review completes.
- `PR_READY` is distinct from release promotion. Optional verification reported as
  `UNAVAILABLE` does not become a false failure or false pass; it can permit PR review
  while still blocking release promotion.
- Skills and handoffs share the canonical fresh-chat authority order. This review found
  and repaired the AI skill's missing `TRACKER.json` routing entry.

## Evidence

The maintenance-core suite includes mutation coverage for unsafe review metadata,
unknown registered checks, package path escape, missing package targets, runtime plane
leakage, fresh-chat skill drift, unavailable release evidence, and serial/parallel
review determinism.

Current locally available result:

- maintenance core: 35/35 tests GREEN;
- family independent review: 2/2 subsystems PR_READY;
- package review: 7 runtime package lanes GREEN;
- deterministic findings: 0 ERROR, 0 WARNING, 4 UNAVAILABLE, 0 INFO;
- AI review: PR_READY / release ELIGIBLE in locally represented evidence;
- YouTube review: PR_READY / release BLOCKED by the already-known Sphinx/jsdom layers;
- runtime source changed by this checkpoint: no.

## Rollback

This checkpoint is maintenance/skill-only. Roll back the manifest-v3 review-profile
fields, `_maintenance/REVIEW.json` files, review tools/schemas/tests/docs, and the AI
skill authority-order repair. No runtime package rollback is required.
