---
title: "XMLReader's error text for bad XML differs with and without lxml (CX-23, rest)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 932: reader snapshots of the same files under two interpreters"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# XMLReader's error text for bad XML differs with and without lxml (CX-23, rest)

## Summary

PR 946 made the type one: `XMLReader` and `TEIReader` raise `ValueError`
("XMLReader: could not parse XML content: ...") for unparsable XML whichever
parser ran, the parser's error as `__cause__`. The text after the prefix is
still the parser's own, so the same file gives two messages on two machines.

## Why it matters

A batch log or a reader snapshot of the same folder differs between a
machine with lxml and one without, although nothing about the files differs.

## Current evidence

`reader_snapshot.py` on the same four files, same tree (PR 946), CPython
3.13 with lxml 6.1 against CPython 3.9 without lxml, three differences:

- empty file: `Document is empty, line 1, column 1 (<string>, line 1)`
  against `no element found: line 1, column 0`;
- plain text: `Start tag expected, '<' not found, line 1, column 1
  (<string>, line 1)` against `syntax error: line 1, column 0`;
- truncated XML: `Premature end of data in tag a line 1, line 1, column 12
  (<string>, line 1)` against `no element found: line 1, column 11`.

## Root cause / current understanding

libxml2 and expat word and locate the same error differently (columns count
from 1 and from 0), and `_unparsable_xml` puts the parser's text in the
message, as the first version of this note asked ("the parser's message
kept").

## Expected behavior

A decision of the maintainer (options, recommended first): (a) keep the
parser's text in the message and accept the difference: close this note and
change the acceptance line below to "the type is the same"; (b) a fixed
message with no parser text (the detail only in `__cause__`), so snapshots
agree byte for byte and the message alone no longer says what is wrong.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_xml.py` (`_unparsable_xml`)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

The hardened parsers and the XML safety tests stay as they are. No mapping
of one parser's wording to the other's.

## Edge cases to cover

Empty file, text that is not XML, truncated XML, a mismatched tag
(`TestUnparsableXmlRaisesOneType` in `_readers/tests/test__xml.py`).

## Verification / acceptance criteria

The snapshot comparison of the two interpreters shows no XML difference.

## Documentation impact

`XMLReader` docstring (Raises), if the message changes.

## Release-note promotion

Required only for (b): `scikitplot.corpus`, type `fix`.
