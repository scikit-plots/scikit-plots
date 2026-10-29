---
title: "The numpy include directory is computed relative to tools/ and used relative to the project root"
status: open
kind: "build"
area: "build/meson (meson.build)"
discovered_during: "TP-1, building an editable install outside a top-level checkout, 2026-10-23"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "fix"
towncrier_fragment: ""
---

# The numpy include directory is computed relative to tools/ and used relative to the project root

## Summary

`meson.build` finds numpy's headers with a Python snippet that changes into
`tools/` and prints `os.path.relpath(np.get_include())`, then passes that text
to `include_directories()`, which Meson resolves relative to the directory of
`meson.build`, the project root. The relative path is therefore one `..` short,
and the build fails whenever numpy is outside the source tree and the checkout
is not directly below `/`.

## Why it matters

A contributor who builds from `~/src/scikit-plots` with a virtual environment
in `~/venvs/...` (the usual layout) cannot build at all; CI and the Docker
images work only because their checkouts are `/work`-like paths one level
below the root, where an extra `..` is absorbed by `/`.

## Current evidence

Checkout at `/home/claude/work/snap1`, Python 3.11 virtual environment at
`/home/claude/venv311`:

```text
pip install --no-build-isolation --no-deps -e .
../../meson.build:1422:13: ERROR: Include dir ../../../venv311/lib/python3.11/site-packages/numpy/_core/include does not exist.
```

From `tools/`, `../../../venv311` is `/home/claude/venv311` (correct); from the
project root it is `/home/venv311` (wrong). The same build succeeds with the
external property the snippet already honours:

```ini
[properties]
numpy-include-dir = '/home/claude/venv311/lib/python3.11/site-packages/numpy/_core/include'
```

passed as `--config-settings=setup-args=--native-file=<that file>`.

## Root cause / current understanding

Verified: the snippet's `os.chdir(os.path.join(".", "tools"))` makes the
relative path relative to `tools/`; `include_directories(_np_inc_dir)` reads
it relative to the project root (`meson.build`, the block that sets
`_np_inc_dir` around line 1380). The `os.path.exists` check passes because it
runs after the same `chdir`.

## Expected behavior

The include directory resolves to numpy's real headers for any location of
the checkout and of the environment, including a virtual environment inside
the source tree (the case the relative path exists for).

## Affected paths and ownership

`meson.build` (repository build, `maintenances/_repo`).

## Constraints and non-goals

Keep the relative form when numpy is inside the source tree (Meson rejects
absolute paths into the source tree); keep the `numpy-include-dir` external
property for cross builds.

## Edge cases to cover

- environment outside the tree, checkout at depth 1 and at depth 3 or more;
- environment inside the tree (`.venv/` in the checkout);
- a cross file that sets `numpy-include-dir`.

## Proposed direction

Compute the relative path from the directory Meson resolves it against: run
the snippet with `os.path.relpath(np.get_include(), <project root>)` (pass
`meson.project_source_root()` as an argument) instead of changing into
`tools/`; use the absolute path when it is outside the tree.

## Verification / acceptance criteria

An editable build succeeds in each edge case above; the configure log's
`numpy include dir` line names an existing directory.

## Documentation impact

None (build internals).

## Release-note promotion

Not required: no user-visible change in the installed package.
