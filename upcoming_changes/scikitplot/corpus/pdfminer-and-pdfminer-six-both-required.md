---
title: "The corpus extra requires both pdfminer and pdfminer.six, which install the same package"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 930: installing the PDF backends on each supported interpreter"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# The corpus extra requires both pdfminer and pdfminer.six

## Summary

`pyproject.toml` (extra `corpus`), `libs/corpus/pyproject.toml` and
`requirements/corpus.txt` list `pdfminer` next to `pdfminer.six`. They are
two distributions that install the same import package, `pdfminer`
(finding CX-17).

## Why it matters

Installing both writes two sets of files into one directory; which modules
remain depends on the order the installer unpacks them. `PDFReader` needs
`pdfminer.high_level.extract_pages`, which only pdfminer.six has.

## Current evidence

- The `pdfminer` distribution's latest release, 20191125, is a source
  archive holding `pdfminer/__init__.py` and `pdfminer/layout.py` and no
  `pdfminer/high_level.py`.
- `PDFReader` imports `pdfminer.high_level`, `pdfminer.layout`,
  `pdfminer.pdfdocument`, `pdfminer.pdfpage` and `pdfminer.pdfparser`.
- Not reproduced here: an install of the extra that ends with a broken
  `pdfminer` package.

## Root cause / current understanding

The reader's install hint names pdfminer.six only; the bare name was
presumably added as the import name.

## Expected behavior

The extra requires pdfminer.six alone.

## Affected paths and ownership

- `pyproject.toml`, `libs/corpus/pyproject.toml`, `requirements/corpus.txt`
- Owner: `maintenances/corpus/`

## Constraints and non-goals

No version bound is added without a tested reason.

## Edge cases to cover

Whether `libs/corpus/pyproject.toml` is generated; other files that repeat
the list.

## Proposed direction

Remove the `pdfminer` line from the three files; install the extra in a
fresh environment and read a PDF.

## Verification / acceptance criteria

`pip install` of the extra installs one distribution providing `pdfminer`;
the PDF tests pass in that environment.

## Documentation impact

None.

## Release-note promotion

Required: `scikitplot.corpus`, type `fix`.
