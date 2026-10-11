---
title: "Detect email addresses that contain non-ASCII letters"
status: open
kind: "security"
area: "scikitplot/cleanprompt"
discovered_during: "cleanprompt round 25, writing the security-and-limits guide page"
release_note: "required"
towncrier_section: "security"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Detect email addresses that contain non-ASCII letters

## Summary

An email address with one non-ASCII letter anywhere in it is not redacted at
all — not partly. Measured on round 25's tree:

```text
encode("mail josé@example.com").text   -> 'mail josé@example.com'
encode("mail ada@exämple.com").text    -> 'mail ada@exämple.com'
encode("mail \u0430da@example.com").text -> unchanged (Cyrillic U+0430 first)
```

## Why it matters

The whole address reaches the model. Internationalised addresses (RFC 6531
local parts, IDN domains) are ordinary for many users, and a look-alike
letter at the start of an address hides it from detection completely. This
is the "returns successfully having redacted less" class the maintenance
plane ranks first.

## Current evidence

- `scikitplot/cleanprompt/_patterns.py`, `EMAIL`: local part
  `[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+`, domain labels `[A-Za-z0-9]…`, leading
  `\b`. Its stated intent is the RFC 5322 dot-atom with an RFC 1035 domain.
- `\b` between a Unicode letter and an ASCII letter is not a boundary, so
  `\u0430da@example.com` does not even match `da@example.com`.
- The round-25 detection view does not help: it folds compatibility forms
  only, and `é`, `ä` and Cyrillic `\u0430` are not compatibility forms of ASCII.

## Root cause / current understanding

The pattern's intent is ASCII-only and its leading boundary is a word
boundary; both are deliberate in the current design, and together they mean a
single non-ASCII letter suppresses the whole match.

## Expected behavior

An address with Unicode letters in the local part or domain is detected as
`EMAIL` and restored exactly. A look-alike letter adjacent to an address
cannot hide the ASCII part of it.

## Affected paths and ownership

`scikitplot/cleanprompt/_patterns.py` (`EMAIL`), its examples, and
`tests/test__patterns.py`; owner `skills/cleanprompt/SKILL.md` ("Adding a
pattern").

## Constraints and non-goals

- Keep the `CP-005` guarantee: no match spans a `|`.
- Keep `CP-028`/`CP-029` behaviour in sentence positions (the twelve-position
  class test must stay green).
- Do not trim spans: a partial address is a leak.
- Base tier only: `re` with Unicode classes, no `regex` package.

## Edge cases to cover

`josé@example.com`, `ada@exämple.com`, `用户@例子.广告`, `\u0430da@example.com`
(the address must be found whole or the Cyrillic letter included), an address
followed by a full stop or a closing bracket, `a@b.a|b`, prose with `@` that
is not an address.

## Proposed direction

Replace the ASCII local-part class with `[^\W_]` plus the RFC 5322 specials,
the leading `\b` with a look-behind that rejects only local-part characters,
and allow Unicode letters in domain labels with the same label-length rules.
Add the edge cases above as `examples_yes`/`examples_no`, run them in the
twelve sentence positions, and add a salted-scale probe like `V25`.

## Verification / acceptance criteria

- every edge case above behaves as stated, through `encode`, `Redactor` and
  the JSON format;
- the full suite and `probe_negative.py` stay green;
- `probe_fuzz.py` reports no new failures.

## Documentation impact

Remove the `EMAIL` paragraph from
`docs/source/user_guide/cleanprompt/security_and_limits.rst` and the matching
FAQ entry in `troubleshooting.rst` once fixed.

## Release-note promotion

Required: a `security` fix fragment.
