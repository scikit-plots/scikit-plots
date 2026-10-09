---
title: "Remove runtime dependency on scikitplot._build_utils from vendored probscale"
status: open
kind: "build"
area: "scikitplot/externals/_probscale"
discovered_during: "API-reference generator maintenance review"
release_note: "unknown"
towncrier_section: "scikitplot.externals"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Remove runtime dependency on scikitplot._build_utils from vendored probscale

## Summary

`scikitplot/externals/_probscale/__init__.py` imports
`scikitplot._build_utils.gitversion.git_remote_version` at runtime even though the
imported function is not used to compute the currently hard-coded `__git_hash__`.
This crosses the repository's documented build-only boundary for
`scikitplot._build_utils`.

## Why it matters

Runtime packages should not depend on build-time tooling. Keeping the import can
pull build-only implementation and Git/version behavior into an ordinary import
of the vendored probability-scale module, increasing coupling and making future
build-tool changes capable of breaking runtime imports unnecessarily.

## Current evidence

- `scikitplot/externals/_probscale/__init__.py:48` imports
  `..._build_utils.gitversion.git_remote_version`.
- The immediately following dynamic hash expression is commented out and
  `__git_hash__` is assigned a fixed string.
- `git_remote_version` is then deleted without being called.
- `python maintenances/_build_utils/_maintenance/tools/check_contract.py --json`
  reports this as a build-only boundary violation.

## Root cause / current understanding

The vendored module retained a historical version-provenance import after the
runtime hash lookup was replaced by a fixed vendored revision. The dependency
therefore no longer serves the active runtime contract.

## Expected behavior

Importing `scikitplot.externals._probscale` must not import
`scikitplot._build_utils`. Vendored version/revision metadata should be static
runtime data generated or updated during vendoring/release maintenance.

## Affected paths and ownership

- Runtime: `scikitplot/externals/_probscale/__init__.py`
- Build-tool boundary owner: `scikitplot/_build_utils`
- Towncrier owner if user-visible: `scikitplot.externals`

## Constraints and non-goals

Do not redesign probscale behavior or its Matplotlib scale registration. Do not
reintroduce Git subprocess/network/version discovery at runtime. Preserve the
current public metadata values unless the vendored revision itself is being
updated.

## Edge cases to cover

- import from a normal installed wheel;
- import from an sdist/build without `.git` metadata;
- environments where Git is absent;
- repeated import remains idempotent;
- no runtime module outside `_build_utils` imports `_build_utils` after the fix.

## Proposed direction

Remove the unused runtime import/delete pair and keep vendored revision metadata
static. If revision metadata needs automation, generate it in the vendoring or
build/release path rather than at runtime.

## Verification / acceptance criteria

- `_probscale` imports successfully without importing any
  `scikitplot._build_utils` module.
- the `_build_utils` maintenance boundary checker no longer reports this path;
- focused `_probscale`/externals tests still pass;
- static version/hash metadata remains available with the intended values.

## Documentation impact

None expected unless version-provenance behavior is documented publicly.

## Release-note promotion

A Towncrier fragment is only required if the final change fixes a user-visible
import/build failure or otherwise meets the repository's release-note threshold.
