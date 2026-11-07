---
title: "Add locale-specific national identifier packs, one jurisdiction per pack"
status: open
kind: "other"
area: "scikitplot/cleanprompt"
discovered_during: "internal review 2026-10-10 (entity coverage) and external comparison 2026-10-09"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# Add locale-specific national identifier packs, one jurisdiction per pack

## Summary

National and regional identifiers beyond `SSN_US` are covered only where a
built-in pack already carries them: `TCKN` (Turkish identity number, with its
checksum) and `NINO` (UK National Insurance number, in its issued letter
ranges) in `personal`, `NHS_NUMBER` (mod 11) in `patient`, `TAX_ID_DE`
(German tax identification number) in `de`, and field names such as
`national_id`, `tax_id` and `passport_number`. The vehicle half of this note
shipped as `CP-117`; the first jurisdiction pack, `de`, as `CP-119`
(`de_tax_id` validator from the published check specification). Their shape
is the template for the remaining jurisdictions.

## Why it matters

National identifiers appear in HR exports, KYC files and support tickets;
each identifies a person.

## Current evidence

`python -m scikitplot.cleanprompt kinds --matrix` lists every kind and the
pack that finds it; `de` is the only jurisdiction pack. The UK National
Insurance number needs no `uk` pack: `personal` already finds it with the
prefix rules of HMRC's National Insurance Manual (NIM39110).

## Root cause / current understanding

Coverage gap. Each identifier needs its published check rule and the label
words used in that jurisdiction, which is domain knowledge per country.

## Expected behavior

One pack per jurisdiction (`tr`, `de`, `uk`, ...), each identifier with its
published checksum as a named validator in `_hooks.VALIDATORS` where one
exists, a prose pattern only when the checksum makes it precise, and a
label-anchored pattern otherwise (as `vehicle`'s `VIN` does: one pattern with
an optional `label` group, the validator relaxing when the label is present).

## Affected paths and ownership

`_config/packs/<locale>.yaml`, `_hooks.py`, `_config/_compiled.json` via
`packs --compile`, `tests/test__hooks.py`, `tests/test__packs.py`, the
coverage table (`tools/generate_coverage_docs.py`).

## Constraints and non-goals

Opt-in packs (not `auto`), so no default output changes. Every check rule
from the identifier's published specification, cited in the validator's
developer notes. No global prose pattern for an identifier whose shape is
ambiguous without a label.

## Edge cases to cover

Valid and invalid check digits; separators; the twelve sentence positions;
the label-anchored form across CSV, JSON and `.env`.

## Proposed direction

One locale pack per contributor with domain knowledge, where a published
check rule exists. Done: `de` (Steuer-ID). Candidates still needing a
published specification each: further jurisdictions' tax and identity
numbers, on a concrete request.

## Verification / acceptance criteria

`packs --check` green; compiled == YAML; every example executed; the scale
probe extended; the coverage table regenerated.

## Documentation impact

`files_and_packs.rst` pack list; the coverage table.

## Release-note promotion

Required: an `enhancement` fragment.
