---
title: "Define one llms.txt artifact owner when AI Assistant and Sphinx LLM coexist"
status: open
kind: "docs-contract"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_llm"
discovered_during: "source-grounded Sphinx extension user-guide synchronization"
release_note: "required"
towncrier_section: "scikitplot._externals"
towncrier_type: "api"
towncrier_fragment: ""
---

# Define one llms.txt artifact owner when AI Assistant and Sphinx LLM coexist

## Summary

Both ``_sphinx_ai_assistant`` and ``_sphinx_llm`` can write ``llms.txt`` into
the Sphinx output directory.  Their default/illustrative configurations can
therefore overlap if both extensions are enabled without explicitly disabling
one artifact pipeline.

## Why it matters

Two build-finished/artifact pipelines targeting the same public filename make
output depend on extension/event ordering and can mix two different semantic
models.  A deployment should have one explicit authority for canonical machine
artifacts.

## Current evidence

- ``_sphinx_ai_assistant`` registers ``ai_assistant_generate_llms_txt`` with
  default ``True`` and writes ``outdir / "llms.txt"``.
- ``_sphinx_llm/core/generator.py`` writes ``outdir / "llms.txt"``.
- the ``_sphinx_llm`` module example shows it alongside
  ``_sphinx_ai_assistant`` without an explicit instruction to disable the
  assistant's overlapping artifact writer.

## Root cause / current understanding

``_sphinx_llm`` is evolving toward canonical semantic machine artifacts while
AI Assistant predates it and already owns a simpler Markdown/``llms.txt``
publication path.  The migration/coexistence contract is not yet explicit.

## Expected behavior

Enabling both extensions must have deterministic documented ownership.  Either
one extension delegates to the other, setup rejects conflicting writers, or a
clear configuration disables one pipeline automatically/explicitly.

## Affected paths and ownership

- ``scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/``
- ``scikitplot/_externals/_sphinx_ext/_sphinx_llm/``
- corresponding maintenance/skills
- both user-guide pages

## Constraints and non-goals

Do not silently change existing published artifact formats during a maintenance
release.  Preserve static-only AI Assistant deployments that do not opt into
``_sphinx_llm``.

## Edge cases to cover

- Assistant only;
- Sphinx LLM only;
- both enabled with defaults;
- one writer explicitly disabled;
- html versus dirhtml;
- failed/partial build cleanup;
- incremental rebuilds and artifact provenance.

## Proposed direction

Define an explicit machine-artifact authority setting or setup-time conflict
check.  Prefer one canonical writer and let consumers (including the Assistant)
read/discover those artifacts rather than maintaining parallel generation
pipelines.

## Verification / acceptance criteria

- a dual-extension build has exactly one deterministic owner for ``llms.txt``;
- tests prove extension ordering cannot change artifact bytes/ownership;
- logs identify the selected owner;
- user guides describe the supported coexistence configuration.

## Documentation impact

Update the AI Assistant and Sphinx LLM user guides once the coexistence contract
is implemented.

## Release-note promotion

Because this defines cross-extension public build behavior, promote it with an
``api`` fragment when implemented.
