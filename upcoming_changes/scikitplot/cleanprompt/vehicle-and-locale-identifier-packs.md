---
title: "Add validated vehicle (VIN) and locale-specific identifier patterns as packs"
status: open
kind: "other"
area: "scikitplot/cleanprompt"
discovered_during: "internal review 2026-10-10 (entity coverage) and external comparison 2026-10-09"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# Add validated vehicle and locale-specific identifier patterns as packs

## Summary

There is no recogniser for vehicle identification numbers, and national or
regional identifiers beyond `SSN_US` live only as structured field names.
The review ranks deterministic, validated identifiers as the first coverage
expansion, ahead of any semantic detector.

## Why it matters

VINs and national identifiers appear in support tickets, logs and datasets;
they identify a person or their property.

## Current evidence

- `python -m scikitplot.cleanprompt kinds` lists no VIN kind.
- `_config/packs/*.yaml` carry field names for several identifiers; the
  structural library carries `SSN_US` only among national IDs.

## Root cause / current understanding

Coverage gap.

## Expected behavior

A `vehicle` pack: field names (`vin`, `vehicle_id`, `chassis_number`) and a
`VIN` pattern — 17 characters from `[A-HJ-NPR-Z0-9]` with the ISO 3779 /
North American check digit as a named validator in `_hooks.VALIDATORS`.
Because the check digit is mandatory only in North America, two patterns: a
validated prose pattern (high precision) and a label-anchored one
(`VIN: <value>`, `(?P<value>...)`) that accepts any well-formed VIN. Locale
packs (`tr`, `de`, `uk`, …) follow the same shape, one jurisdiction per pack,
each identifier with its published checksum where one exists (TC Kimlik,
Steuer-ID, NHS number already exists as a field).

## Affected paths and ownership

`_config/packs/vehicle.yaml` (+ locale packs), `_hooks.py`,
`_config/_compiled.json` via `packs --compile`, `tests/test__packs.py`,
`tests/test__hooks.py`, `tests/test__runtime.SAMPLES` if a format changes.

## Constraints and non-goals

Positive examples that a `secrets`-pack pattern accepts are written as
fragments (`I14`). No global prose pattern for an identifier whose shape is
ambiguous without a label.

## Edge cases to cover

Valid and invalid check digits; a 17-character product code that is not a
VIN; lowercase VINs; a VIN adjacent to punctuation (twelve sentence
positions); the label-anchored form across CSV, JSON and `.env`.

## Proposed direction

Vehicle pack first (one validator, measurable precision), then one locale
pack per contributor with domain knowledge.

## Verification / acceptance criteria

`packs --check` green; compiled == YAML; every example executed; the scale
probe extended with VIN fragments.

## Documentation impact

`files_and_packs.rst` pack list; the packs gallery example.

## Release-note promotion

Required: an `enhancement` fragment.
