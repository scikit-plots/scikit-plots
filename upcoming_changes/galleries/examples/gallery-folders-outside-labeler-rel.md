---
title: "galleries/examples/env and ipymagic do not get their module label"
status: open
kind: "other"
area: "galleries/examples"
discovered_during: "PR 891, gallery examples"
release_note: "not-required"
towncrier_section: "documentation"
towncrier_type: "other"
towncrier_fragment: ""
---

# galleries/examples/env and ipymagic do not get their module label

## Summary

The labeler's satellite rule maps a module's gallery to
`galleries/examples/{rel}/**/*`, where `{rel}` is the path below
`scikitplot/` (`_env`, `_ipymagic`). The galleries of these two modules live
in `galleries/examples/env` and `galleries/examples/ipymagic` (the names asked
for in PR 891), so a pull request that changes only them gets no
`scikitplot/_env` or `scikitplot/_ipymagic` label.

## Why it matters

Reviewers who follow a module label miss gallery-only changes.

## Current evidence

`[tool.scikitplot.maintenance.labeler.satellites]` in `pyproject.toml`;
`python tools/maint_tools/generate_pr_labeler.py sync` reports no change after
the two folders were added.

## Root cause / current understanding

The user guide uses the private path (`docs/source/user_guide/_env`); the
gallery uses the public name. Both conventions are in use.

## Expected behavior

Either the folders follow `{rel}` (`galleries/examples/_env`,
`galleries/examples/_ipymagic`, updating the README labels' targets), or the
satellite rule gains a placeholder for the name without the leading `_`.

## Affected paths and ownership

`galleries/examples/env/`, `galleries/examples/ipymagic/`, `pyproject.toml`
labeler satellites, `tools/maint_tools/generate_pr_labeler.py`.

## Constraints and non-goals

Do not break the gallery labels `env__examples` and `ipymagic__examples`.

## Edge cases to cover

A module whose public name collides with another folder (`externals` and
`_externals`).

## Proposed direction

Rename the folders to `{rel}`; it needs no tool change.

## Verification / acceptance criteria

`generate_pr_labeler.py explain galleries/examples/<folder>/README.txt` lists
the module label.

## Documentation impact

none (labels only).

## Release-note promotion

Not required.
