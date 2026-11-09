---
title: "Image reader tests fail instead of skipping when Pillow is absent"
status: open
kind: "reliability"
area: "scikitplot/corpus"
discovered_during: "PR 919: research self-test of corpus and mcp (finding CX-07)"
release_note: "not-required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# Image reader tests fail instead of skipping when Pillow is absent

## Summary

In an environment without Pillow, 25 tests in
`scikitplot/corpus/_readers/tests/test__image.py` fail with
`ModuleNotFoundError: No module named 'PIL'` instead of skipping. They mock
the reader with `unittest.mock.patch("PIL.Image.open", ...)`, and patching
a dotted target imports `PIL` first.

## Why it matters

The partial distribution `scikit-plots-corpus` does not depend on Pillow
(`libs/corpus/pyproject.toml`: `scikit-plots-skinny` and `numpy`); the full
distribution gets Pillow only through matplotlib. A test run of the corpus
part on its own therefore reports 25 failures that are not defects, which
hides real failures among them.

## Current evidence

- CPython 3.14.6 with numpy, scipy, scikit-learn, pydantic, mcp, lxml,
  pypdf, langdetect, regex, krippendorff, tenacity and no Pillow: 25 failed
  in `test__image.py`, the same set on `main` (`6734def`) and on PR 919.
- With matplotlib installed (Pillow present), the same tests pass.
- `scikitplot/corpus/_readers/_image.py` itself raises an `ImportError`
  with a message when Pillow is missing: only the tests are ungated.

## Root cause / current understanding

`patch("PIL.Image.open")` resolves its target by importing `PIL.Image`; the
test module has no Pillow gate (`pytest.importorskip("PIL")` or a
`skipif`).

## Expected behavior

Without Pillow, the tests that need it are skipped with a stated reason, and
the reader's own missing-Pillow `ImportError` is tested.

## Affected paths and ownership

- `scikitplot/corpus/_readers/tests/test__image.py`
- Owner: `maintenances/corpus/`

## Constraints and non-goals

Tests only; no runtime change. Do not make Pillow a dependency of the
partial distribution for this.

## Edge cases to cover

Pillow absent; Pillow present without `pytesseract` or `easyocr` (already
mocked through `sys.modules`).

## Proposed direction

Gate the Pillow-dependent tests on Pillow (module-level or per class), and
add one test that the reader raises its `ImportError` when Pillow is
absent.

## Verification / acceptance criteria

Without Pillow: 0 failures in `test__image.py`, the Pillow tests reported
as skipped. With Pillow: the same tests pass as before.

## Documentation impact

None.

## Release-note promotion

Not required (tests only).
