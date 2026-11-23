---
title: "orient.py reports a stale pull request number and no next id for corpus findings"
status: open
kind: "reliability"
area: "tools/maint_tools"
discovered_during: "Session start from the 2026-11-23 handoff"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# orient.py reports a stale pull request number and no next id for corpus findings

## Summary

On `main` at 13b457d, `python -B maintenances/_repo/_maintenance/tools/orient.py`
prints `pull request: 938` (the next pull request is 955) and lists no
`corpus` area under "areas with findings", although corpus findings run to
CX-45.

## Why it matters

`.claude/init` tells every session to start from `orient.py`. A session
that trusts it files fragments under the wrong pull request number and has
to search the ledger by hand for the next CX id (risk of reusing one).

## Current evidence

- `.claude/new-chat-template.md` FILL IN: `Pull request number for new
  fragments: 938` (last changed 2026-11-15). `orient.py` takes the number
  from there and derives the next one only when the line is missing or
  reads `auto`.
- `maintenances/corpus/REVIEW.json` has no `findings` key (keys:
  `schema_version`, `subsystem`, `lanes`, `release_gates`); CX ids exist
  only in `maintenances/corpus/todo/todo.md` and `upcoming_changes/` notes.
  `orient.py` counts ids from `REVIEW.json` only.

## Root cause / current understanding

Verified: the template's fixed number is not updated after each merge; the
corpus area records its findings outside the file `orient.py` reads.

## Expected behavior

`orient.py` reports the next pull request number by default and the next
free id for every series, wherever the area records it.

## Affected paths and ownership

`.claude/new-chat-template.md`, `maintenances/_repo/_maintenance/tools/orient.py`,
`maintenances/corpus/REVIEW.json`. Owner: `maintenances/_repo`.

## Constraints and non-goals

Generators must not import the installed package (lesson CI-12).

## Edge cases to cover

Template line `auto`, missing, or a number lower than the highest fragment
number (warn); ids found in both `REVIEW.json` and notes (take the maximum).

## Proposed direction

Set the FILL IN line to `auto`, and make `orient.py` warn when a fixed
number is not above the highest number in fragment names and recent commit
subjects. Either record CX findings in `corpus/REVIEW.json`, or let
`next_ids` also scan `(CX-NN)` tokens in the area's notes and ledger.

## Verification / acceptance criteria

`orient.py` prints `955 (derived ...)` at 13b457d and `corpus: ... next
CX-54` once this note's findings CX-46 to CX-53 are recorded; tests for the
warning.

## Documentation impact

`.claude/init` step 2 if the output format changes.

## Release-note promotion

Not required (maintenance tooling).
