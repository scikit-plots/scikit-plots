# Fresh-chat handoff for repository maintenance

This file is intentionally short. It is the restart point when no prior chat,
run log, or contributor context should be assumed.

## Scope first

1. Read `docs/maintenances/MAINTAINING.md`.
2. Restate the requested scope in repository paths.
3. Do not expand a documentation task into a general code review.
4. Use source/tests/configuration only to verify the claims needed for that
   scope.

## Choose the owner

For a submodule such as `scikitplot/annoy`:

1. read `maintenances/annoy/MAINTAINING.md`;
2. read current state/evidence files named by that entry point;
3. read `skills/annoy/SKILL.md` if it exists;
4. inspect the runtime, docs, tests and build files that the task actually
   touches.

For nested owners, keep the same nested path where the repository provides one.
Do not assume a similarly named subsystem owns another subsystem's files.

If there is no matching maintenance or skill directory, continue from current
source/docs/tests and state that the maintenance coverage is absent instead of
inventing a contract.

## Current truth, not historical truth

Treat `STATE.json`, `EVIDENCE.json`, focused tests and current source as current
only when they apply to this checkout and can be reproduced. Treat history,
archived runs and old chat conclusions as leads to verify, not facts to repeat.

## Change discipline

Use:

```text
inspect -> verify -> explain root cause -> edit -> verify again
```

Do not edit while a verification process is still reading the same tree.
Prefer small, reviewable batches over a broad rewrite.

## Findings outside the requested edit scope

A concrete unresolved runtime/library/build/compatibility issue discovered while
reviewing docs belongs under:

```text
upcoming_changes/<origin path>/<short-slug>.md
```

Examples:

```text
upcoming_changes/scikitplot/annoy/narrow-index-capacity.md
upcoming_changes/libs/corpus/python-floor.md
```

Use `upcoming_changes/TEMPLATE.md`. Do not put speculative ideas there.

A user-visible fix that is actually implemented should also receive a Towncrier
fragment under `docs/source/whats_new/upcoming_changes/` according to that
directory's `README.md`.

## Handoff

End each completed run with:

- Added
- Changed
- Removed
- Verification performed
- Not verified / remaining risks
- New `upcoming_changes/` findings, if any

For repository handoff artifacts, provide the complete updated repository, not
only patches, when that is what the task requests.
