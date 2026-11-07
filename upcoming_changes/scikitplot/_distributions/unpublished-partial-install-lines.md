---
title: "The partial distributions' names are unclaimed on PyPI"
status: blocked
kind: "security"
area: "scikitplot/_distributions"
discovered_during: "PR 911: review of the published usage skills (scikit-plots/skills 05d1ab6)"
release_note: "required"
towncrier_section: "scikitplot"
towncrier_type: "fix"
towncrier_fragment: "docs/source/whats_new/upcoming_changes/scikitplot/911.fix.rst"
---

# The partial distributions' names are unclaimed on PyPI

## Summary

The partial distributions (`scikit-plots-skinny`, `-annoy`, `-cleanprompt`,
`-corpus`, `-cython`, `-mcp`, `-mlflow`, `-rank-bm25`, `-sphinx-ext`) are not
on PyPI, and the project cannot publish there yet. Their names are unclaimed:
a person who types `pip install scikit-plots-corpus`, or a tool that resolves
`scikit-plots-skinny` (every part depends on it) from PyPI, installs whatever
another party publishes under that name. Status `blocked`: the remaining step
is registering the names, which needs PyPI publishing.

## Why it matters

Dependency confusion. The project's own advice no longer names a part bare
(PR 911, below), but the names stay open to anyone until registered.

## Current evidence

- PyPI JSON API, 2026-11-07: 404 for all nine names; `scikit-plots` 200 at
  `0.4.0.post11` (no `agent`, `cleanprompt`, `feature_selection`,
  `levenshtein`, `logging`, `mcp`, `rank_bm25`).
- The staging index serves the ten `scikit-plots*` distributions at
  `0.5.dev0` and no third-party package.

## Root cause / current understanding

Fixed in PR 911. `scikitplot/_distributions.py` records which distributions
are on PyPI (`ON_PYPI`, today only `scikit-plots`) and builds every install
command:
- `requirement` / `pip_install` give a part as a quoted direct link to its
  `libs/` directory, with the core named;
- `staging_install` gives the two staging-index commands: packages
  `--no-deps` from the index, then dependencies from PyPI.

Each form was installed and imported in a clean environment from local builds
standing in for GitHub and the index:
- a part alone fails resolving `scikit-plots-skinny`;
- the core first, then a part alone, works;
- the two-step form resolves no project name from PyPI.

These now use the map:
- `install_hint` (root and CLI messages, including the delegate hint);
- `doctor`'s fixes;
- the `cleanprompt` and `mcp` corpus messages;
- the `libs/` READMEs (`libs/_tools`);
- the install guide, the root README and the rank-bm25 pages;
- the development skill `scikitplot-repo-dev`.

The usage plugin names only `pip install scikit-plots` and links the guide,
because the Claude directory refuses other sources in a plugin.

Not verified: the staging index itself (unreachable from the review
environment).

## Expected behavior

Each name is held by this project on PyPI; then `ON_PYPI` lists it and every
command names it bare.

## Affected paths and ownership

`scikitplot/_distributions.py` (`ON_PYPI`), `libs/_tools` (READMEs),
`docs/source/install/installation.rst`; owner `maintenances/_repo`
(`libs/`), skill `scikitplot-repo-dev`.

## Constraints and non-goals

No name is added to `ON_PYPI` before its first release is on PyPI.

## Edge cases to cover

`scikit-plots-skinny` published before a part (the part's command then names
the core bare and the part as a link); a part published and later yanked.

## Proposed direction

When publishing is possible: publish (or register with a first release) all
nine names, add each to `ON_PYPI`, run `python -m libs._tools generate`, and
update the install guide's commands.

## Verification / acceptance criteria

All nine names return 200 from PyPI's JSON API under this project's
account; `ON_PYPI` lists them; `scikitplot/tests/test__distributions.py`
and `python -m libs._tools check` pass.

## Documentation impact

`docs/source/install/installation.rst` and `README.md` when the names
reach PyPI.

## Release-note promotion

PR 911's advice change: `scikitplot/911.fix.rst`. Delete this note when the
acceptance criteria hold.
