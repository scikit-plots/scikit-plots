---
title: "<short descriptive title>"
status: open
kind: "<bug|security|reliability|compatibility|packaging|build|api|docs-contract|other>"
area: "<scikitplot/...|libs/...|docs/...|galleries/...|build/...>"
discovered_during: "<task or review scope>"
release_note: "<required|not-required|unknown>"
towncrier_section: "<configured section path or unknown>"
towncrier_type: "<major-feature|feature|efficiency|enhancement|fix|api|other|unknown>"
towncrier_fragment: ""
---

# <short descriptive title>

## Summary

State the concrete current problem in a few sentences. Describe observed
behavior, not an assumed cause.

## Why it matters

Explain user, contributor, compatibility, security, reliability or maintenance
impact. Name the affected public behavior when there is one.

## Current evidence

List the smallest reproducible evidence that establishes the finding:

- relevant file paths and symbols;
- focused commands/tests/probes;
- observed result;
- environment/version details only when they affect the result.

Do not paste large transient logs. Do not include secrets or sensitive data.

## Root cause / current understanding

State the verified root cause. If only part is verified, separate fact from
hypothesis explicitly.

## Expected behavior

Describe the contract the implementation should satisfy after the change.

## Affected paths and ownership

List the primary runtime/library/build paths and the matching maintenance/skill
owner when one exists.

## Constraints and non-goals

Record compatibility, API, security, performance, generated-source or ownership
constraints. State what this change must not accidentally redesign.

## Edge cases to cover

Include both ordinary and boundary cases relevant to the issue. Prefer cases
that can become tests or deterministic probes.

## Proposed direction

Describe a safe implementation direction without presenting an untested patch
as the only solution.

## Verification / acceptance criteria

Use measurable conditions. Include focused tests plus integration/platform
checks when the claim crosses those boundaries.

## Documentation impact

Name the docs/user-guide/gallery/API pages that need synchronization after the
runtime change. Use `none` only when verified.

## Release-note promotion

State whether a Towncrier fragment is required. When implemented, fill in the
configured section/type and final fragment path under
`docs/source/whats_new/upcoming_changes/`.
