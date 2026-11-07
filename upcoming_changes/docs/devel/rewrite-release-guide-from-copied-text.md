---
title: "The linked release guide is Matplotlib's release-manager text"
status: open
kind: "docs-contract"
area: "docs/source/devel"
discovered_during: "PR 909 review of the maintainer changelog workflow"
release_note: "not-required"
towncrier_section: "documentation"
towncrier_type: "other"
towncrier_fragment: ""
---

# The linked release guide is Matplotlib's release-manager text

## Summary

`docs/source/devel/guide_release.rst` (label `release-guide`) is the release
guide the developer index links to (`devel/index.rst.template`), but its
procedure is Matplotlib's: it describes Matplotlib's versioning, CI, test
notebooks, GitHub statistics files and what's-new layout. A scikit-plots
release manager following it would run commands against paths that do not
exist here.

## Why it matters

It is the only reachable release procedure. The orphaned Astropy copy under
`docs/source/devel/maintainers/` is covered by
`upcoming_changes/docs/devel/sync-maintainer-changelog-workflow.md`.

## Current evidence

- Line 15: "Template for further usage, template belong to matplotlib."
- Line 18: "This document is only relevant for Matplotlib release managers."
- Line 27: "Matplotlib follows the Intended Effort Versioning (EffVer)".
- Line 106: CI link to `github.com/matplotlib/matplotlib/actions`.
- Lines 119-122: `lib/matplotlib/backends/web_backend/nbagg_uat.ipynb`,
  `%matplotlib widget`.
- Lines 136-146: `doc/users/github_stats.rst` and
  `doc/users/prev_whats_new/`; lines 205-217: `doc/users/next_whats_new/` and
  `doc/api/next_api_changes/` (scikit-plots uses Towncrier fragments under
  `docs/source/whats_new/upcoming_changes/`).
- 43 lines name Matplotlib or its paths. A few lines are already
  scikit-plots specific (line 350, the scikit-plots releases page; Zenodo DOI).
- What the repository's automation shows: publishing to PyPI/TestPyPI is
  `workflow_dispatch` with a chosen distribution, repository and version
  (`.github/workflows/ci_wheels_(test)pypi.yml`); conda has its own workflows
  (`ci_wheels_conda*.yml`); the version is declared in
  `scikitplot/__init__.py`; the citation files are regenerated with
  `tools/citation/citation_ref.py sync --apply`; release DOIs go into
  `VERSION_DOIS` of that tool.

## Root cause / current understanding

The page was copied as a template and never rewritten for scikit-plots.

## Expected behavior

The release guide describes only scikit-plots' verified procedure: version
bump, release-note rendering, citation regeneration, tagging, building and
publishing (PyPI, conda, partial distributions under `libs/`), the Zenodo DOI,
and the post-release steps.

## Affected paths and ownership

`docs/source/devel/guide_release.rst`; it should agree with
`docs/source/whats_new/upcoming_changes/README.md`, `tools/citation/README.md`
and the publishing workflows.

## Constraints and non-goals

Do not invent a process: the branch/tag naming, who renders and commits the
final `whats_new` file, and when fragments are deleted must be confirmed by the
maintainer first. Keep the `release-guide` label (linked from the index).

## Edge cases to cover

Release candidates versus final releases; post-releases (`0.4.0.post12` is the
workflow default); partial distributions released on their own; a release
whose Zenodo DOI is minted after the tag.

## Proposed direction

Rewrite the page section by section from the verified facts above plus the
maintainer's confirmed steps; remove the Matplotlib-only sections.

## Verification / acceptance criteria

No Matplotlib paths, commands or links remain; every command exists in this
repository; the page agrees with the fragment README and the citation README.

## Documentation impact

Developer guide, release section.

## Release-note promotion

Not required (internal maintainer guidance).

## Related

`upcoming_changes/docs/devel/sync-maintainer-changelog-workflow.md` and
`upcoming_changes/docs/devel/review-astropy-affiliated-language.md`.
