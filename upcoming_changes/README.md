# Upcoming changes: development follow-up ledger

This root directory is a **current engineering follow-up ledger** for concrete
issues discovered during documentation, maintenance, compatibility or focused
subsystem review that are not being implemented in the current task.

It is deliberately separate from:

```text
docs/source/whats_new/upcoming_changes/
```

The latter is the Towncrier **release-note fragment** directory. Do not symlink
these directories and do not use one as a substitute for the other.

## What belongs here

Create a note when current evidence exposes a specific unresolved problem such
as:

- incorrect or inconsistent runtime logic;
- a security or privacy weakness;
- unreliable/error-prone behavior;
- a stale or broken compatibility contract;
- a packaging/build mismatch;
- a public API/documentation contract that cannot be made truthful without a
  runtime change;
- an important edge case that the current implementation mishandles.

Do not create notes for cosmetic preferences, unverified suspicions, broad
brainstorms, or historical issues that are already fixed.

## Path convention

Mirror the origin of the finding so ownership is obvious:

```text
upcoming_changes/scikitplot/<submodule>/<short-slug>.md
upcoming_changes/libs/<submodule>/<short-slug>.md
upcoming_changes/tools/<submodule>/<short-slug>.md
upcoming_changes/docs/<section>/<short-slug>.md
upcoming_changes/galleries/<section>/<short-slug>.md
upcoming_changes/build/<area>/<short-slug>.md
```

For a nested owner, keep enough path components to disambiguate it.

Examples:

```text
upcoming_changes/scikitplot/annoy/narrow-index-capacity.md
upcoming_changes/scikitplot/cexternals/_annoy/header-contract.md
upcoming_changes/libs/corpus/python-floor.md
upcoming_changes/tools/maint_tools/import-safe-cli.md
upcoming_changes/docs/user_guide/stale-install-matrix.md
upcoming_changes/galleries/examples/stale-public-workflow.md
```

Use lowercase kebab-case for the note filename. One note should describe one
coherent root cause or tightly coupled change.

## Required content

Copy `upcoming_changes/TEMPLATE.md` and fill it from **current evidence**. A good
note lets a contributor or AI agent start the implementation without needing
the chat that discovered it.

At minimum state:

- the problem and why it matters;
- evidence and reproduction/observation method;
- affected paths/public behavior;
- expected behavior;
- constraints and non-goals;
- important simple/complex edge cases;
- security/privacy implications when relevant;
- verification and acceptance criteria;
- documentation and release-note impact.

Prefer file paths, commands and observed behavior over narrative history.

## Status and lifecycle

Use the template field `status` with one of:

- `open` — verified finding, not yet scheduled;
- `planned` — accepted for an implementation task;
- `in-progress` — actively being implemented;
- `blocked` — accepted but waiting on named evidence/dependency;
- `promoted` — implementation is verified and any required Towncrier fragment
  has been created.

This directory is not intended to become a permanent historical log.

After a fix is implemented and verified:

1. decide whether the change is user-visible/release-note-worthy;
2. if yes, create the Towncrier fragment in the configured section under
   `docs/source/whats_new/upcoming_changes/` and record its path in the note;
3. mark the note `promoted` while the change is being reviewed if useful;
4. remove the root planning note once the implementation/release-note handoff
   no longer needs it.

The release fragment, PR and released changelog become the durable history.

## Relationship to documentation work

For documentation-focused reviews, use this ledger when the docs cannot be
made accurate without changing runtime/library/build behavior. Do not silently
change the runtime simply to make an existing documentation statement true.

A docs-only fix can be made directly under `docs/` and, when release-note
worthy, can have a `documentation` Towncrier fragment without creating a root
planning note.

## Promotion to release notes

The root note is detailed engineering context. A Towncrier fragment is a short
user-facing statement about a delivered change.

Do not copy the entire planning note into the changelog. Follow:

```text
docs/source/whats_new/upcoming_changes/README.md
```

The fragment should describe what changed for users, use the correct configured
section/type, and reference the pull request number required by the release-note
workflow.
