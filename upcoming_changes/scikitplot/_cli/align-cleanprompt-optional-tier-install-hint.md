---
title: "Align the project CLI CleanPrompt install hint with current optional tiers"
status: open
kind: "docs-contract"
area: "scikitplot/_cli"
discovered_during: "source-grounded CleanPrompt user-guide synchronization"
release_note: "unknown"
towncrier_section: "scikitplot._cli"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Align the project CLI CleanPrompt install hint with current optional tiers

## Summary

The project-wide CleanPrompt command registration advertises an optional-tier
install hint that omits the NLTK tier and describes the crypto extra as required
for "vault encryption", while current CleanPrompt supports a base-tier portable
encrypted vault and uses ``cryptography`` only for the optional Fernet cipher.

## Why it matters

The hint is user-facing recovery guidance. A user can be sent toward a larger or
incorrect dependency set and may not discover the supported NLTK-only entity
path.

## Current evidence

- ``scikitplot/_cli/registry.py`` registers ``cleanprompt`` and lists
  ``cleanprompt-ner``, ``cleanprompt-web`` and ``cleanprompt-crypto`` only.
- ``scikitplot/cleanprompt/_capabilities.py`` defines four optional tiers:
  ``ner``, ``nltk``, ``web`` and ``crypto``.
- the ``crypto`` tier purpose explicitly states that it adds the Fernet vault
  cipher and that encryption itself needs no optional tier.
- root ``pyproject.toml`` defines ``cleanprompt-nltk`` as a first-class extra.

## Root cause / current understanding

The central CLI registration text predates the NLTK tier and the portable
standard-library vault cipher.

## Expected behavior

The central CLI hint should enumerate the supported project extras consistently
with the CleanPrompt capability table and describe ``cleanprompt-crypto`` as the
Fernet option rather than encryption in general.

## Affected paths and ownership

- ``scikitplot/_cli/registry.py``
- ``scikitplot/cleanprompt/_capabilities.py``
- ``pyproject.toml`` optional extras

## Constraints and non-goals

Do not gate the base ``cleanprompt`` command on optional dependencies. Preserve
the deliberate design that the base CLI remains available without any optional
tier.

## Edge cases to cover

- no optional dependencies installed;
- only NLTK installed;
- only spaCy installed;
- portable encrypted vault without ``cryptography``;
- explicit Fernet request without ``cryptography``.

## Proposed direction

Generate or validate the central hint against CleanPrompt's tier metadata, or
at minimum update the text and add a focused drift test.

## Verification / acceptance criteria

- NLTK extra is discoverable from the project CLI guidance;
- encryption is not described as unavailable without ``cryptography``;
- Fernet still points to ``cleanprompt-crypto``;
- base CleanPrompt command remains ungated.

## Documentation impact

Keep ``docs/source/user_guide/cleanprompt/index.rst`` capability wording aligned
with the corrected hint.

## Release-note promotion

Decide during implementation whether correcting CLI recovery guidance warrants a
``fix`` fragment under ``scikitplot._cli``.
