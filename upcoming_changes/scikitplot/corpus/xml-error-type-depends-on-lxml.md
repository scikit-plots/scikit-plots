---
title: "XMLReader raises a different exception type for bad XML with and without lxml"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 932: reader snapshots of the same files under two interpreters"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# XMLReader raises a different exception type for bad XML with and without lxml

## Summary

For an empty or malformed `.xml` file, `XMLReader` raises
`lxml.etree.XMLSyntaxError` where lxml is installed and `ValueError`
("XMLReader: could not parse XML content: ...") where it is not (finding
CX-23).

## Why it matters

Code that catches the error must catch both types, and only one of them can
be tested on a given machine. The same file gives a different error message
in a batch log on two machines.

## Current evidence

- `reader_snapshot.py` on the same folder, same tree: CPython 3.13 with lxml
  `XMLSyntaxError: Document is empty, line 1, column 1`; CPython 3.12 without
  lxml `ValueError: XMLReader: could not parse XML content: no element
  found: line 1, column 0`. The same for a file of plain text.
- `_parse_xml`'s docstring states that a parse error from an installed lxml
  "is still authoritative and propagates, preserving the historical parser
  contract": the difference is deliberate.

## Root cause / current understanding

The stdlib path wraps its errors in `ValueError`; the lxml path does not.

## Expected behavior

One exception type for unparsable XML, whichever parser ran, with the
parser's message kept and the original as `__cause__`.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_xml.py` (`_parse_xml`)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

The hardened parsers and the XML safety tests stay as they are.

## Edge cases to cover

Empty file, text that is not XML, truncated XML, an entity expansion attack
(the safety tests catch `XMLSyntaxError` from the parser directly).

## Proposed direction

A decision: (a) wrap lxml's `XMLSyntaxError` in the same `ValueError`
(`XMLSyntaxError` would remain the `__cause__`); (b) keep the historical
contract and document both types.

## Verification / acceptance criteria

The snapshot comparison of the two interpreters shows no XML difference.

## Documentation impact

`XMLReader` docstring (Raises).

## Release-note promotion

Required: `scikitplot.corpus`, type `fix`.
