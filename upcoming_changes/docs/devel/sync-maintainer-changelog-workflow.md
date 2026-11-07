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

## Additional evidence (PR 907 review, deferred by the maintainer)

- The whole `docs/source/devel/maintainers/` folder is Astropy's maintainer
  guide copied as-is: `index.rst` ("Maintaining astropy and affiliated
  packages"), `maintainer_workflow.rst`, `releasing.rst`, `testhelpers.rst`
  ("Astropy Testing Tools") and `astropy-package-template.rst`.
- `maintainers/index.rst` is `:orphan:` and no page links to it, so no
  navigation reaches these pages; Sphinx still builds and publishes them.
- Only three of their labels are referenced from other pages:
  `github-squash-and-merge` (`guide_devel_details.rst`, twice),
  `pre-commit_bot` (`guide_qu_contribute.rst`, `guide_devel_details.rst`
  twice) and `benchmarks` (`guide_test.rst`). The other 19 labels have no
  inbound reference. Removing or replacing the folder must keep those three
  targets.
- What the repository actually does with the changelog (verified):
  `[tool.towncrier]` renders `docs/source/whats_new/v0.5.rst` (single file)
  from `docs/source/whats_new/upcoming_changes/` with
  `upcoming_changes/towncrier_template.rst.jinja2`; the fragment README
  documents the preview `towncrier build --draft --version 0.5.0`; CircleCI's
  docs job runs `towncrier build --yes` (no `--version`: the version comes
  from the `scikitplot` package) in its own checkout before `make html`, so
  the published docs always include unreleased fragments; fragment names are
  checked by `tools/maint_tools/generate_towncrier_sections.py check` and PRs
  by `pr_check_changelog.yml`.
- No verified final-release step (who renders and commits `v0.5.rst`, and
  when fragments are deleted) is documented anywhere in the repository.

## Related

`upcoming_changes/docs/devel/review-astropy-affiliated-language.md` covers the
inherited Astropy wording in the other developer guides; review them together.

`upcoming_changes/docs/devel/rewrite-release-guide-from-copied-text.md`
covers `docs/source/devel/guide_release.rst`, the release guide the
navigation actually links to.
