---
title: "Restoration must not rewrite placeholder-like text that was in the source"
status: open
kind: "bug"
area: "scikitplot/cleanprompt"
discovered_during: "cleanprompt round 25 (independent review)"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Restoration must not rewrite placeholder-like text that was in the source

## Summary

If the original text already contains a spelling the lenient restorer
accepts as a placeholder variant, restoring the redacted text replaces it
with a vault value, so the round trip is not exact. Present on the uploaded
tree:

```text
r = Redactor().redact("[EMAIL\u2028-1] ada@example.com")
r.text                        -> '[EMAIL\u2028-1] [EMAIL-1]'
restore(r.text, r.vault).text -> 'ada@example.com ada@example.com'
```

## Why it matters

Invariant I1 (exact round trip) is stated for text containing no *literal*
placeholder; a lenient variant is not literal, so the invariant is broken in
a case it claims to cover, and the restored text puts a value where the user
had written something else.

## Current evidence

`_engine.reserved_label_spans` reserves exact-grammar placeholders found in
the source, so they survive; lenient variants are not reserved, and
`restore(lenient=True)` repairs them.

## Root cause / current understanding

Reservation (encode side) and leniency (restore side) use different grammars.

## Expected behavior

Either the encode side reserves every lenient variant present in the source
(and records them so restoration leaves them), or the restorer is told which
spans of the reply are known source text. The first is local to one call and
preferred.

## Affected paths and ownership

`_engine.py` (`reserved_label_spans`, `restore`), `_policy.TagStyle`
(`lenient_pattern`), `tests/test__engine.py`.

## Constraints and non-goals

Keep `CP-042`'s repairs for model rewrites; keep `[note 2]`-style prose
untouched.

## Edge cases to cover

Each lenient shape from the `CP-042` table appearing in the source; the same
shape appearing in a model reply (must still be repaired); surrogate style.

## Proposed direction

Reserve lenient matches in the source at encode time and store their labels
as "literal in source" in the result so `restore` skips them.

## Verification / acceptance criteria

The randomized round-trip probe gains lenient variants in its cold
fragments, with 0 failures.

## Documentation impact

`how_it_works.rst` (restore section).

## Release-note promotion

Required.
