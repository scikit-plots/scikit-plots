---
title: "ALTOReader skips a malformed page without lxml and fails the archive with it (CX-42)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 947: reading the other callers of the hardened lxml parser"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# ALTOReader skips a malformed page without lxml and fails the archive with it (CX-42)

## Summary

`ALTOReader`'s page loop catches `ValueError` around `_parse_xml_bytes`,
logs a warning and goes on to the next page. Without lxml a malformed page
raises `ValueError` and is skipped. With lxml it raises
`lxml.etree.XMLSyntaxError`, a `SyntaxError`, which the loop does not catch:
one bad page ends the whole archive.

## Why it matters

The same archive gives every good page on one machine and an exception on
another; the machine with the better parser installed gets less.

## Current evidence

- `_parse_xml_bytes(b"<alto><Page>")`, tree of PR 947, CPython 3.13: with
  lxml 6.1 `XMLSyntaxError` (`isinstance(exc, ValueError)` is `False`);
  with lxml blocked `ValueError` ("ALTOReader: could not parse XML: ...").
- `scikitplot/corpus/_readers/_alto.py`: `_parse_xml_bytes` wraps only the
  stdlib path; the loop reads `except ValueError as exc:` then `continue`.
- No `test__alto.py` exists under `_readers/tests/`.

## Root cause / current understanding

The same as CX-23, in a second function: the stdlib path converts its
errors to `ValueError`, the lxml path does not.

## Expected behavior

A malformed page is skipped with the same warning whichever parser ran.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_alto.py` (`_parse_xml_bytes`); new tests in
  `scikitplot/corpus/_readers/tests/`.
- Owner: `maintenances/corpus/`

## Constraints and non-goals

Submodule files stay independent; the hardened parsers stay as they are.
Whether `_alto.py` and `_xml.py` should share one parse function is a
separate question.

## Edge cases to cover

An archive with one malformed page among good ones, with and without lxml;
an archive whose every page is malformed; an empty page file.

## Verification / acceptance criteria

A test archive with one malformed page yields the other pages and one
warning under both parsers; the test fails on the tree of PR 947 with lxml.

## Documentation impact

None expected.

## Release-note promotion

Required: `scikitplot.corpus`, type `fix`.
