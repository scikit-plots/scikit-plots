---
title: "Synchronize maintainer changelog and release workflow with Scikit-Plots Towncrier"
status: open
kind: "docs-contract"
area: "docs/source/devel/maintainers"
discovered_during: "Towncrier section ownership and next-release tracking review"
release_note: "not-required"
towncrier_section: "documentation"
towncrier_type: "other"
towncrier_fragment: ""
---

# Synchronize maintainer changelog and release workflow with Scikit-Plots Towncrier

## Summary

The maintainer documentation still contains an Astropy changelog workflow and a
release example that targets `CHANGES.rst`/Astropy paths rather than the current
Scikit-Plots Towncrier configuration in `pyproject.toml`.

## Why it matters

A maintainer following those pages can render or commit the wrong file, look for
nonexistent Astropy documentation, or assume the wrong release procedure. That
conflicts with the repository-specific fragment workflow under
`docs/source/whats_new/upcoming_changes/`.

## Current evidence

- `docs/source/devel/maintainers/maintainer_workflow.rst` says "The Astropy
  changelog" and links to Astropy `docs/changes/README.rst`.
- `docs/source/devel/maintainers/releasing.rst` instructs maintainers to run an
  Astropy-style `towncrier build --version 6.0.0`, inspect `CHANGES.rst`, and
  commit `CHANGES.rst`.
- `pyproject.toml` currently configures Towncrier to write
  `docs/source/whats_new/v0.5.rst` from
  `docs/source/whats_new/upcoming_changes/`.

## Root cause / current understanding

The maintainer pages retain upstream/copied release-process prose that was not
synchronized when Scikit-Plots adopted its current Towncrier target and fragment
layout.

## Expected behavior

Maintainer documentation should describe only the current Scikit-Plots
fragment, preview, final rendering, review, and commit workflow and should refer
to the configured target/version rather than unrelated upstream paths.

## Affected paths and ownership

- `docs/source/devel/maintainers/maintainer_workflow.rst`
- `docs/source/devel/maintainers/releasing.rst`
- `pyproject.toml` (`[tool.towncrier]`)
- `docs/source/whats_new/upcoming_changes/README.md`

## Constraints and non-goals

Do not redesign the release process merely to make the copied prose fit. First
verify the actual release branch/tag workflow and current automation, then make
the maintainer guide reflect that verified process.

## Edge cases to cover

- release candidates versus final releases;
- previewing without consuming fragments;
- building on release branches without unintentionally consuming fragments for
  a later release;
- version changes after `0.5`;
- fragment cleanup after final rendering.

## Proposed direction

Replace the upstream-specific prose with a Scikit-Plots workflow grounded in
`pyproject.toml`, the changelog-fragment README, and the repository's actual
release automation.

## Verification / acceptance criteria

- No Astropy changelog paths/commands remain in the Scikit-Plots maintainer
  changelog procedure.
- Commands operate on the configured Scikit-Plots target.
- Preview and final-build semantics are distinguished clearly.
- Release docs and `docs/source/whats_new/upcoming_changes/README.md` agree.

## Documentation impact

This is a maintainer-documentation synchronization task.

## Release-note promotion

No Towncrier fragment is required for correcting internal maintainer guidance.
