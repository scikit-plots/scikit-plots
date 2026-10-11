---
title: "Add a per-kind action layer (report, tokenize, surrogate, block) to the policy"
status: open
kind: "api"
area: "scikitplot/cleanprompt"
discovered_during: "internal review 2026-10-10 (policy direction) and external comparison 2026-10-09"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Add a per-kind action layer to the policy

## Summary

A policy decides *what is detected* (`kinds`, profiles, packs, engines) and,
through `TagStyle.style`, one global way of replacing it. It cannot say "for
`EMAIL` use a surrogate, for `CREDIT_CARD` a placeholder, for `MEDICATION`
refuse to send, for `TITLE_CASE` only report". Richer entity classes (packs,
a future semantic detector) need that second dimension before they are safe
to add.

## Why it matters

Different categories need different handling: naturalistic substitution of a
diagnosis or a religion changes meaning dangerously; some values must block
the send outright; some detections are too noisy to redact automatically but
worth reporting.

## Current evidence

- `RedactionPolicy` fields: `kinds`, `allow`, `tag_style`, `overlap`,
  `limits`, `case_insensitive`, `min_confidence`, `preserve_placeholders`.
- `TagStyle.style` is one of `placeholder`, `surrogate`, global.
- `SURROGATE_KINDS` hard-codes which kinds a surrogate may replace.

## Root cause / current understanding

Design gap, not a defect.

## Expected behavior

`RedactionPolicy.actions: Mapping[str, Action]` with
`Action ∈ {tokenize, surrogate, report, block}` (`preserve` is already
`allow`). Defaults reproduce today's behaviour exactly, and the three
profiles keep their meaning (the external review's "do not redefine strict").
`block` raises a typed error before anything is returned, naming kinds and
counts; `report` leaves the value and lists it as a finding (and `Guard`
still refuses to send it unless allowed). `surrogate` on a kind outside the
safety floor is refused at policy construction.

## Affected paths and ownership

`_policy.py` (field, validation, fingerprint), `_engine.py` (assign/rewrite),
`_diagnostics.py` (report outcome), `_guard.py`, `_cli.py` (`--action
KIND=ACTION`, repeatable, both frontends), `_plan.py`.

## Constraints and non-goals

Invariants I1–I9 hold for every action except `report` (which by design
leaves a value; it must be impossible to configure silently — `doctor`
lists reported kinds as blind spots of severity high). No change to the
default output.

## Edge cases to cover

Overlapping spans with different actions (the merged span takes the
strictest action: block > tokenize > surrogate > report); `block` inside a
batch run (the file is refused, not partly written); actions on pack kinds;
fingerprint change when actions change.

## Proposed direction

Implement after `customizable-surrogate-generator.md` slice A, since the
safety floor is shared.

## Verification / acceptance criteria

Default-equivalence test over randomised documents (byte-identical to the
current output); per-action tests; parity tests for `--action`.

## Documentation impact

`how_it_works.rst`, `python_api.rst`, `security_and_limits.rst`.

## Release-note promotion

Required: a `feature` fragment.
