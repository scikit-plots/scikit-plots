---
title: "Read key-value, code and JSON-string names through the detection view"
status: open
kind: "security"
area: "scikitplot/cleanprompt"
discovered_during: "cleanprompt round 25 (CP-102/CP-103 verification)"
release_note: "required"
towncrier_section: "security"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Read key-value, code and JSON-string names through the detection view

## Summary

Round 25 made the detection view (`CP-098`) and field-name normalisation
(`CP-103`) catch invisible and compatibility characters in values and in
whole-string keys (CSV/TSV headers, JSON keys, email headers). Three surfaces
still read names as written, so one invisible character in a *name* hides the
value it labels. Measured on round 25's final tree:

```text
encode_text('export DB_PASS\u200bWORD=hunter2hunter2\n', "shell")  -> value unchanged
encode_text('const pass\u200dword = "hunter2hunter2";\n', "javascript") -> value unchanged
encode_text('API_\u200bKEY = "abc123def456"\n', "python")       -> value unchanged
encode_text('{"z": "bob\u200b@example.com"}', "json")           -> value unchanged
```

`\u200d` (ZWJ) is legal inside a JavaScript identifier, so the second line is
valid code.

## Why it matters

The value reaches the model. Secrets in shell exports and code are exactly
what the `secrets` pack exists for.

## Current evidence

- The key-value and code splitters (`_structured.field_regions`,
  `_code.discover`) find names with an identifier character class (or
  `ast`/`tokenize`, which reject the character in Python), so the name is cut
  at the invisible character and never reaches `normalise_field`.
- JSON values are scanned through `_TokenBound`, which clips spans to tokens
  computed from the original text; it is document-bound and so, correctly,
  does not read the view (`CP-102`).
- `tests/test__runtime.TestInvisibleCharactersKeepStructure.NOT_YET` lists the
  formats where this is still open.

## Root cause / current understanding

Regions and tokens are computed from the original text only. The view exists
but nothing computes regions on it.

## Expected behavior

A name or a JSON string with an invisible character inside it is read as if
written plainly; offsets still index the original; the round trip is exact;
a document's structure (lines, separators, JSON validity) is unchanged.

## Affected paths and ownership

`_structured.py` (regions), `_code.py` (discovery), `_runtime.py`
(`_encode_native`, `_TokenBound`), `_canonical.py` (the view).

## Constraints and non-goals

Never rewrite the source; the vault keeps original surfaces. Python code that
cannot parse because of the character keeps its current fallback, but its
assignments must still be found by the text-level secret patterns.

## Edge cases to cover

Each line above; a salted key whose value is also salted; a JSON string with
escapes *and* a zero-width character (escape offsets via `_string_offsets`);
CRLF files; the twelve sentence positions for prose fallbacks.

## Proposed direction

Compute regions (and JSON tokens) on `detection_view(text)` when it differs,
then map every region and token boundary back with `DetectionView.source_span`
before any detector runs. Then `FieldDetector` and `_TokenBound` stay
document-bound but bound to correct original offsets. Remove each format from
`NOT_YET` as it is fixed.

## Verification / acceptance criteria

`TestInvisibleCharactersKeepStructure` passes for every sample with
`NOT_YET` empty; a salted-scale probe over the code and key-value formats
reports 0 leaks and 0 structural changes.

## Documentation impact

Remove the corresponding limit from
`docs/source/user_guide/cleanprompt/security_and_limits.rst`.

## Release-note promotion

Required: a `security` fix fragment.
