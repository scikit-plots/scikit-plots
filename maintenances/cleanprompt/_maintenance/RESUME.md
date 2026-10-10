# Resume here

This file is the working log of the round in progress. It exists so that a
fresh chat — one with no history, no transcript and no memory of the last
session — can continue the work exactly where it stopped, without redoing a
step or trusting a claim nobody re-ran.

`FRESH_CHAT_HANDOFF.md` explains the subsystem. This file says **where the
work is**. Read this one first, then the handoff.

## How to use this file

1. Read **Current round** and **Step log**. A step is done only when its
   evidence column names a command that was run and what it printed.
2. Re-run the **Baseline commands** before touching anything. If a number
   differs from **Last verified numbers**, the tree changed since this file was
   written: find out why before continuing.
3. Do the **Next action**. When a step finishes, update its row *in the same
   change* as the code — a log written afterwards from memory is a claim with
   no evidence behind it.
4. Keep **Open ledger** equal to the notes under
   `upcoming_changes/scikitplot/cleanprompt/`. `tests/test_resume.py` fails
   when they disagree, so a note cannot be forgotten and a closed one cannot
   linger here.
5. At the end of a round, move the round's summary to `HISTORY.md`, reset the
   step log for the next round, and leave **Next action** pointing at it.

## Baseline commands

Run from the wide checkout root (the directory holding `scikitplot/`,
`maintenances/` and `skills/`).

```sh
python -B maintenances/cleanprompt/_maintenance/check_trackers.py --json
python -B -m pytest maintenances/cleanprompt/_maintenance/tests -q -p no:cacheprovider
python -B -m pytest scikitplot/cleanprompt/tests -q -p no:cacheprovider
python -B -m pytest docs/source/user_guide/cleanprompt -q -p no:cacheprovider
```

The package suite needs the parent `scikitplot` importable. In a source
checkout without a compiled build, run it under a stand-in parent: a directory
holding `scikitplot/__init__.py` (empty) and `scikitplot/cleanprompt` (a copy
or link of the real package), with a `pytest.ini` that sets
`filterwarnings = error`. That is how rounds 23–25 were verified, and
`tests/_isolated.py` measures import isolation the same way.

**Never edit the tree while a run reads it** (`tasks/lessons.md`, rules 1 and
22): verify against a frozen copy, or wait for the run to finish.

## Current round

Round twenty-six is **complete** (2026-10-10): *the user decides, and the
floor does not move* — a pattern-risk check for custom packs (warn by
default, tuned per pattern, run, team or machine), custom surrogate name
sets under a fixed safety floor, and the generator's design and growth plan.
Summary in `HISTORY.md`; plan and results in `tasks/todo.md` under
"cleanprompt round 26"; reasoning in `DESIGN.md` section 25 and
`GENERATOR_DESIGN.md`. Findings `CP-104` … `CP-107` are closed in
`REVIEW.json`. PR number for all round-25/26 fragments: **864**.

Maintainer decisions taken this round (do not reopen without asking):
pull request 864; risky custom patterns are **always warned about** with
quick options, never refused by default; generator scope was delegated
(data sets first, provider protocol later only on a concrete request).

## Step log

| # | Step | Status | Evidence |
|---|---|---|---|
| 1 | Round-25 fragments under PR 864; two notes promoted | done | `generate_towncrier_sections.py check` PASS |
| 2 | `_pattern_risk.py`: parser, overlap by public `re`, rules, policy surfaces (packs, plan, cleaner, CLI, env, `risk: accepted`) | done | `tests/test__pattern_risk.py`; `TestCP104`; CLI smoke in both frontends |
| 3 | `GENERATOR_DESIGN.md`; slice A `_surrogate_sets.py`, `TagStyle.surrogates`, `--surrogates` | done | `tests/test__surrogate_sets.py`; default digests pinned and equal to round 25 |
| 4 | `CP-105` (held name shown with a dot) found via the gallery | done | `TestCP105`; reproduced on the round-25 tree |
| 5 | Independent review (one agent, fresh): 12 findings, all reproduced and fixed or recorded | done | analyser (verbose, inline flags, escapes, equal units, bounded repetition, separators one group up, possessive), surrogate floor (invisible/look-alike entries, TITLE_CASE, pairs), `CP-106` append grammar, `-W error`; ordinary-word entries → new ledger note. A second review pass was cut off by a rate limit; its brief was re-run by hand (reviewer scripts, verbose edge cases, 14 scripts of names) and found `CP-107` and the NFKC over-refusal (Thai, Arabic) |
| 6 | Soundness fuzz of the analyser | done | `evidence/probe_round26_fuzz.py`: found 3 misses of the first fix (nullable groups, optional parts); after the fix 0 slow among passed patterns |
| 7 | Docs (5 pages), gallery (packs, moderate), README, fragments, ledger | done | guide sync 9 passed; every new example executed |
| 8 | Verification ladder on a frozen copy identical to the work tree | done | see **Last verified numbers** |
| 9 | Evidence refresh, maintenance plane, drop-in | done | `check_trackers.py`: maintenance PASS, runtime PASS, release UNVERIFIED |

## Last verified numbers

Final round-26 tree, 2026-10-10 (frozen copy identical to the work tree):

- package suite, no optional tier: 2888 passed, 88 skipped (2 runs, a
  shuffled order, a shuffled order under xdist, live logging under xdist)
- spaCy 3.8.16 and NLTK 3.10.3 installed, no model and no data: 2899 passed,
  77 skipped
- every tier (model and data present): 2966 passed, 10 skipped
- CPython 3.8, 3.9, 3.10, 3.11, 3.12, 3.13, 3.14 (pytest only): all green
- probes: negative (no tier, every tier), isolation, fuzz (4000 documents),
  scale, live engines, round 25, round 26 (22 measured verdicts), round-26
  fuzz (0 slow among passed patterns) — 0 failures; gallery 9/9 in both
  installations
- `check_trackers.py`: maintenance PASS, runtime PASS, release UNVERIFIED

## Next action

Round twenty-seven. Suggested order (each has a note with a full design):

1. `invisible-characters-in-keyvalue-and-code-names.md`,
   `email-addresses-with-non-ascii-letters.md`,
   `phone-pattern-groupings-and-ranges.md` — detectors and names together.
2. `restore-alters-near-placeholders-in-the-source.md` and
   `surrogate-stand-ins-that-are-ordinary-words.md` — restoration review
   (both are about what `decode` rewrites in a reply).
3. `bound-custom-pattern-runtime-with-a-timeout.md` (optional `regex` tier).
4. Capability notes: per-kind actions, vehicle/locale packs, look-alike
   characters, coverage matrix; `surrogate-provider-protocol.md` only on a
   concrete request.

Reset the step log above for round 27 before starting.

## Open ledger

Every note under `upcoming_changes/scikitplot/cleanprompt/`, with its status.
`tests/test_resume.py` compares this list with the directory. Promoted notes
are removed once PR 864 merges.

- `upcoming_changes/scikitplot/cleanprompt/bound-custom-pattern-runtime-with-a-timeout.md` — open
- `upcoming_changes/scikitplot/cleanprompt/customizable-surrogate-generator.md` — promoted
- `upcoming_changes/scikitplot/cleanprompt/email-addresses-with-non-ascii-letters.md` — open
- `upcoming_changes/scikitplot/cleanprompt/generated-coverage-matrix.md` — open
- `upcoming_changes/scikitplot/cleanprompt/guard-custom-pattern-runtime.md` — promoted
- `upcoming_changes/scikitplot/cleanprompt/invisible-characters-in-keyvalue-and-code-names.md` — open
- `upcoming_changes/scikitplot/cleanprompt/look-alike-characters-in-detection-view.md` — open
- `upcoming_changes/scikitplot/cleanprompt/per-kind-action-policy.md` — open
- `upcoming_changes/scikitplot/cleanprompt/phone-pattern-groupings-and-ranges.md` — open
- `upcoming_changes/scikitplot/cleanprompt/restore-alters-near-placeholders-in-the-source.md` — open
- `upcoming_changes/scikitplot/cleanprompt/round-25-release-fragments.md` — promoted
- `upcoming_changes/scikitplot/cleanprompt/surrogate-provider-protocol.md` — open
- `upcoming_changes/scikitplot/cleanprompt/surrogate-stand-ins-that-are-ordinary-words.md` — open
- `upcoming_changes/scikitplot/cleanprompt/vehicle-and-locale-identifier-packs.md` — open

Outside this directory but owned by these rounds:
`upcoming_changes/scikitplot/_cli/align-cleanprompt-optional-tier-install-hint.md`
(promoted: `scikitplot._cli/864.fix.rst`).

## Decisions waiting for the maintainer

1. A contributor GitHub handle for the `By :user:` lines of the PR-864
   fragments (not guessed; none added).
2. `surrogate-stand-ins-that-are-ordinary-words.md`: which of the three
   directions (report ambiguity on decode, a length/case rule, a per-set
   `avoid:` list).
3. `bound-custom-pattern-runtime-with-a-timeout.md`: whether an optional
   `regex` tier is wanted.
