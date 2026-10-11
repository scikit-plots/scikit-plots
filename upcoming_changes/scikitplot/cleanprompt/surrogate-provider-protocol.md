---
title: "Let Python code propose surrogate names (provider protocol, slice B)"
status: open
kind: "api"
area: "scikitplot/cleanprompt"
discovered_during: "cleanprompt round 26 (slice B of customizable-surrogate-generator)"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Let Python code propose surrogate names (provider protocol, slice B)

## Summary

Round 26 shipped data-only surrogate sets (slice A). Slice B lets a Python
object propose names through the same core loop, for needs a data file
cannot express (for example a generated, locale-aware cast).

## Why it matters

Only when a concrete request needs it. `GENERATOR_DESIGN.md` §6 recommends
waiting for one; this note keeps the design ready so the decision is cheap.

## Current evidence

- `_surrogates.surrogate_for(..., provider=)` already asks any object with
  `candidate(kind, index)`; `SurrogateSet` is the only implementation.
- The run-time floor (`entry_problem`) already applies to every proposal, and
  `EMAIL`/`PHONE`/`URL` never reach a provider (`test__surrogate_sets`
  `TestCoreLoop`).

## Root cause / current understanding

Not a defect. `TagStyle._check_surrogates` requires an `identity` string,
which a provider would also have to declare.

## Expected behavior

`TagStyle(style="surrogate", surrogate_set=provider)` accepts any object with
`identity: str` and `candidate(kind, index) -> str | None`. A provider that
raises stops the pass with an error naming the provider; a malformed proposal
is skipped; the identity is recorded exactly as for a set.

## Affected paths and ownership

`_surrogate_sets.py` (a `SurrogateProvider` protocol), `_policy.py`
(`TagStyle` docs), `python_api.rst`, `GENERATOR_DESIGN.md`.

## Constraints and non-goals

Python API only (no plugin loading from files or entry points: that would run
code from a configuration file). Deterministic output is the provider's
contract and must be documented; a test checks it for the reference
implementation.

## Edge cases to cover

A provider returning a non-string; returning a held value written
differently (`CP-071`, `CP-105`); returning the same name forever (bounded
search, placeholder fallback); raising.

## Proposed direction

Formalise the protocol; add the error wrapping that names the provider; a
gallery example with a tiny provider.

## Verification / acceptance criteria

Every edge case as a test; round-trip and no-leakage over randomised
documents with a provider.

## Documentation impact

`python_api.rst`, `how_it_works.rst` (surrogate sets section).

## Release-note promotion

Required: a `feature` fragment under `scikitplot.cleanprompt`.
