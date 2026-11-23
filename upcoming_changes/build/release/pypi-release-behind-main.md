---
title: "The latest PyPI release fails to import scikitplot.corpus on a clean install and lags main by seven months"
status: open
kind: "packaging"
area: "build/release"
discovered_during: "Real-data testing with the latest published release (2026-11-23)"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# The latest PyPI release fails to import scikitplot.corpus on a clean install and lags main by seven months

## Summary

`pip install scikit-plots` gives `0.4.0.post11`, uploaded 2026-04-12. In a
clean virtual environment (CPython 3.13, Linux) `import scikitplot.corpus`
raises `ModuleNotFoundError: No module named 'typing_extensions'`, and
`python -m scikitplot` raises `ModuleNotFoundError: No module named
'click'`. A clean install of `main` (13b457d) has neither problem.

## Why it matters

Users who follow the documentation install the release, so every corpus
fix since April (office formats, ALTO, encodings, PDF stability) is
unavailable to them and the corpus submodule does not import at all.

## Current evidence

Same archive, documents only, `reader_snapshot.py snapshot --repeat 2`:

| Build | Files | Errors | Unstable | Seconds | Peak RSS |
|---|---|---|---|---|---|
| 0.4.0.post11 (+ typing_extensions) | 692 | 124 | 1 | 231 | 479 MiB |
| main 13b457d | 692 | 122 | 0 | 61 | 483 MiB |

The release also decodes UTF-8 Markdown with CJK text as Latin-1 (4 files),
reads ALTO metadata instead of page text, refuses `.pptx`/`.xlsx`, logs a
`.zip` registration clash at every import, and has one PDF whose text
differs between two reads. All are fixed on `main`.

## Root cause / current understanding

Verified: the release's `scikitplot/corpus/_base.py` imports
`typing_extensions` unconditionally, which its metadata declares only for
the `dev` extra; `main` imports it under `TYPE_CHECKING`. The CLI's `click`
import is likewise fixed on `main`. No release has been published since.

## Expected behavior

A clean `pip install scikit-plots` imports every submodule documented as
installable with base dependencies, and the CLI starts.

## Affected paths and ownership

Release process (`.github/workflows/ci_wheels_(test)pypi.yml`), owner
`maintenances/_repo`.

## Constraints and non-goals

No code change on `main` is needed for these symptoms; this note asks for a
release and for a gate that would have caught them.

## Edge cases to cover

Python 3.8 to 3.15; base install without any extra; each extra alone.

## Proposed direction

(a) Publish a release from `main`. (b) Add a wheel smoke job: install the
built wheel into a fresh environment without extras and run
`python -c "import scikitplot.<sub>"` for every public submodule plus
`python -m scikitplot --help`.

## Verification / acceptance criteria

The smoke job passes on the wheel uploaded to TestPyPI; `pip install
scikit-plots` in a clean environment imports `scikitplot.corpus`.

## Documentation impact

none

## Release-note promotion

Not required (the release itself carries the accumulated fragments).
