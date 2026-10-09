---
title: "Verify provenance and content of existing 0.5 changelog fragments"
status: open
kind: "docs-contract"
area: "docs/source/whats_new/upcoming_changes"
discovered_during: "repository maintenance / skills / upcoming-changes workflow bootstrap"
release_note: "not-required"
towncrier_section: "documentation"
towncrier_type: "other"
towncrier_fragment: ""
---

# Verify provenance and content of existing 0.5 changelog fragments

## Summary

The current Towncrier fragment directory contains three populated fragments that
predate the repository-specific workflow documented in this review. Their
provenance/content should be verified before they are included in the next
release draft rather than being silently treated as current release truth.

The files are:

- `docs/source/whats_new/upcoming_changes/array-api/29639.other.rst`
- `docs/source/whats_new/upcoming_changes/documentation/456.feature.rst`
- `docs/source/whats_new/upcoming_changes/documentation/510.feature.rst`

## Why it matters

Towncrier aggregates these files into the next release notes. A stale, copied or
misclassified fragment can make the published changelog claim a change that did
not occur in the associated Scikit-Plots pull request, or can expose internal
file-level implementation detail instead of useful user-facing behavior.

## Current evidence

### `29639.other.rst`

The fragment describes removal of `cupy.array_api` support and credits
`Olivier Grisel <ogrisel>`. Pull request number `29639` is far outside the
current Scikit-Plots PR range visible during this review, and the text resembles
an upstream/scikit-learn Array API changelog entry. Its Scikit-Plots provenance
was not established in this pass.

### `456.feature.rst`

The fragment says a gallery/example tagging directive was introduced. The
public Scikit-Plots PR at
`https://github.com/scikit-plots/scikit-plots/pull/456` is titled
`Modified Annoy cpp`. The fragment may still correspond to a change included in
that PR, but that relationship must be verified from the PR diff/commit history
before release.

### `510.feature.rst`

The fragment currently consists primarily of the internal path
`docs/source/_sphinx_ext/skplt_ext/sphinx_tabs_patch.py` plus a source link. It
does not explain a user-visible change. The public Scikit-Plots PR at
`https://github.com/scikit-plots/scikit-plots/pull/510` is titled
`Subpackage bug fix`; the exact relationship between that PR and this fragment
should be verified.

## Root cause / current understanding

The `upcoming_changes` directory previously contained copied/general changelog
instructions and no clear separation between engineering follow-up notes and
release fragments. These entries may have been created under older conventions.
This review does not have enough provenance to decide safely whether each entry
should be rewritten, moved, retained, or removed.

## Expected behavior

Every fragment included in the next release draft should:

- map to the Scikit-Plots pull request named in its filename;
- describe verified user-visible behavior (or a justified release-note item);
- use a configured Towncrier section and type;
- contain one coherent ReStructuredText bullet;
- avoid presenting an internal file path as the change itself.

## Affected paths and ownership

Primary paths:

- `docs/source/whats_new/upcoming_changes/`
- `pyproject.toml` (`[tool.towncrier]`)

This is release/documentation maintenance; it does not require a runtime code
change unless provenance review uncovers a missing implementation/documentation
sync.

## Constraints and non-goals

- Do not delete a valid unreleased fragment merely because its PR title is
  broad or different.
- Do not invent replacement release-note text without reading the associated
  PR/files and verifying the delivered behavior.
- Do not convert historical provenance uncertainty into a runtime defect.

## Edge cases to cover

- A PR may legitimately contain several unrelated commits despite its title.
- A merged PR may already have been released, in which case its fragment should
  not remain in the next-release queue.
- A change may be internal and correctly require no changelog entry.
- A single PR may require more than one fragment when distinct configured types
  are genuinely needed.

## Proposed direction

For each populated fragment, inspect the Scikit-Plots PR/commit diff and the
current release history, then choose one of: retain as-is, rewrite to the
verified user-facing change, reclassify section/type, move into an already
released changelog, or remove if no release fragment is warranted.

## Verification / acceptance criteria

- Every remaining populated fragment has verified Scikit-Plots provenance.
- Filename PR numbers correspond to the pull request that delivered the change.
- Fragment text satisfies
  `docs/source/whats_new/upcoming_changes/README.md`.
- `towncrier build --draft --version <next-version>` renders the resulting
  release notes without stale/copied entries.

## Documentation impact

Update release-note fragments only. If provenance review reveals a public
feature missing from the user guide/gallery, create a separate focused docs
task.

## Release-note promotion

No fragment is required for this maintenance cleanup itself. The purpose of the
work is to make the existing next-release fragments trustworthy.
