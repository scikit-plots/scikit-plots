---
title: "PHONE: redact international groupings whole, and stop matching year ranges"
status: open
kind: "security"
area: "scikitplot/cleanprompt"
discovered_during: "cleanprompt round 25 (independent review of the detection view)"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "fix"
towncrier_fragment: ""
---

# PHONE: redact international groupings whole, and stop matching year ranges

## Summary

Two `PHONE` behaviours, both present on the uploaded tree with plain ASCII
(the round-25 detection view only extends them to Unicode dashes and spaces,
as designed):

```text
encode("Tel : +33 1 23 45 67 89").text        -> 'Tel : +33 1 [PHONE-1]'   (prefix sent)
encode("Revenue grew 12% in 2019-2024").text  -> 'Revenue grew 12% in [PHONE-1]'
```

## Why it matters

The first is a partial redaction: the country and area code go to the model,
and a placeholder that covers only part of a value is the leak shape
`CP-016` described. The second replaces information the model needs and
teaches users to distrust the output.

## Current evidence

`scikitplot/cleanprompt/_patterns.py`, `PHONE`, and its examples; outputs above
measured on both the uploaded tree and round 25's.

## Root cause / current understanding

The pattern's grouping alternatives do not include the two-digit European
grouping after a one-digit area code, so the match starts at the first group
it accepts; a nine-digit run with one hyphen fits a North American shape.

## Expected behavior

An international number in common national groupings is matched from its `+`
or leading digits to its last group; a `YYYY-YYYY` year range is not a phone
number.

## Affected paths and ownership

`_patterns.py` (`PHONE` pattern, a validator if needed), `tests/test__patterns.py`.

## Constraints and non-goals

Keep every current `examples_yes`; run new ones through the twelve sentence
positions; no locale database (that is the locale packs' job).

## Edge cases to cover

French, German, UK and Turkish groupings; extensions; a year range; a page
range; ISBNs; dates written with hyphens.

## Proposed direction

Add the missing grouping alternatives, and a validator that rejects a match
consisting of two four-digit years in plausible range joined by a dash.

## Verification / acceptance criteria

Every edge case asserted; scale probe extended; fuzz probe unchanged.

## Documentation impact

None beyond the pattern list.

## Release-note promotion

Required.
