---
title: "Warn when a surrogate stand-in is also an ordinary word"
status: open
kind: "reliability"
area: "scikitplot/cleanprompt"
discovered_during: "cleanprompt round 26 independent review (surrogate sets)"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# Warn when a surrogate stand-in is also an ordinary word

## Summary

`decode` restores a stand-in wherever it appears in the model's reply. A
surrogate set entry that is also an ordinary word (`The`, `Ash`) therefore
rewrites the reply's own words: the review loaded `GPE: [The, Ash, Ash-Vale]`
and `decode("The trip from The to Ash ...")` returned
`"Oslo trip from Oslo to Bergen ..."`.

## Why it matters

A wrong word in a restored answer is silent data corruption. The built-in
names are chosen to be unusual, but a user's set can contain anything that
reads as a name.

## Current evidence

- Round 26 review, reproduced (`scratchpad/review26/s.py` in that session).
- `GENERATOR_DESIGN.md` §2 records it as a limit, not an invariant; the
  guide (`how_it_works.rst`, surrogate sets) tells set authors to avoid
  ordinary words.

## Root cause / current understanding

Restoration is a literal replacement of issued stand-ins (by design: the
model's reply is free text). Nothing can tell "the model wrote `Ash` meaning
the stand-in" from "the model wrote the word ash".

## Expected behavior

At least one of, decided with the maintainer:

1. `decode` reports a stand-in that occurs in the reply more often than it
   was issued in the prompt (a count both sides know), as "ambiguous", and
   `--exact`-style strictness can refuse it;
2. set validation refuses entries shorter than a documented length or
   entirely lower-case (deterministic, but excludes some real names);
3. an optional, user-supplied stop-word list in the set file
   (`avoid: [...]`), checked at load.

## Affected paths and ownership

`_engine.restore` / `_types.RestorationResult` (option 1), `_surrogate_sets.py`
(options 2–3), docs.

## Constraints and non-goals

No shipped word list (a guess about other people's languages). Base tier
only.

## Edge cases to cover

A stand-in that is a word in one language only; a reply quoting the prompt;
streaming decode (`StreamDecoder`).

## Proposed direction

Option 1: it is language-independent and uses information restoration
already has.

## Verification / acceptance criteria

The review's `The`/`Ash` reproduction reports the ambiguity instead of
silently rewriting.

## Documentation impact

`how_it_works.rst` (surrogate sets), `troubleshooting.rst`.

## Release-note promotion

Required: an `enhancement` fragment under `scikitplot.cleanprompt`.
