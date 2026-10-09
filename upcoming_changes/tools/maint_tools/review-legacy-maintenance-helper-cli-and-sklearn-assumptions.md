---
title: "Review legacy maintenance helper CLI side effects and scikit-learn assumptions"
status: open
kind: "maintenance-tooling"
area: "tools/maint_tools"
discovered_during: "Towncrier ownership helper and CLI policy review"
release_note: "unknown"
towncrier_section: "tools.maint_tools"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Review legacy maintenance helper CLI side effects and scikit-learn assumptions

## Summary

Several older scripts under `tools/maint_tools/` do not follow the repository's
current import-safe maintenance CLI policy, and at least two retain explicit
scikit-learn paths/names that are not valid Scikit-Plots ownership assumptions.
They should be reviewed one-by-one before being relied on for release automation.

## Current evidence

- `tools/maint_tools/update_tracking_issue.py` constructs an `ArgumentParser`,
  calls `parse_args()`, initializes GitHub state, and executes workflow logic at
  module import time.
- `tools/maint_tools/sort_whats_new.py` consumes `sys.stdin` and prints output at
  module import time. Its generated module heading is still
  `:mod:\`sklearn.<module>\`` and its parser explicitly recognizes `sklearn.`.
- `tools/maint_tools/check_xfailed_checks.py` imports scikit-learn private test
  helpers and executes estimator checks at module import time.
- `tools/maint_tools/bump-dependencies-versions.py` still uses
  `scikit_learn_release_date` terminology and calls
  `sklearn/_min_dependencies.py`, which is not the Scikit-Plots repository path.

## Why it matters

Import-time execution makes helpers difficult to test, compose, or safely call
from other maintenance automation. Stale `sklearn` paths can produce incorrect
output or fail only when a release maintainer needs the tool.

## Expected behavior

Each retained Python maintenance helper should have an explicit supported
purpose, avoid side effects on import, expose a testable `main(argv, ...)` style
entry point when it is a CLI, use Scikit-Plots paths/terminology, and fail with
actionable non-zero status instead of depending on hidden working-directory
assumptions.

## Constraints and non-goals

Do not mechanically rewrite every inherited helper. First determine whether each
script is still used; remove obsolete helpers rather than modernizing dead code.
Networked helpers should remain explicit network tools and must not run network
operations merely by being imported.

## Edge cases to cover

- execution from a working directory other than repository root;
- `--help` without optional/network dependencies where practical;
- imports by tests or other maintenance modules;
- missing GitHub credentials/network access;
- Python versions supported by `pyproject.toml`;
- stdin-driven tools receiving empty or malformed input.

## Proposed direction

Audit `tools/maint_tools/*.py`, classify each helper as retained/deprecated/
removed, then migrate retained CLIs to the policy documented in
`tools/maint_tools/README.md` and exercised by `generate_towncrier_sections.py`.

## Verification / acceptance criteria

- importing retained CLI modules does not parse argv, read stdin, perform
  network requests, or terminate the process;
- `--help` is deterministic and safe;
- supported helpers resolve repository paths correctly;
- no retained helper emits or invokes stale `sklearn` repository paths unless
  that upstream reference is intentional and documented;
- focused tests cover success, invalid input, and refused/unsafe operations.

## Documentation and release-note impact

Update `tools/maint_tools/README.md` as helper support changes. If the cleanup
materially changes a contributor/release workflow, add a fragment under
`docs/source/whats_new/upcoming_changes/tools.maint_tools/` after implementation
and a real pull-request number are available.
