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
`maintenances/` and `plugins/scikitplot-dev/skills/`).

```sh
python -B maintenances/cleanprompt/_maintenance/check_trackers.py --json
python -B -m pytest maintenances/cleanprompt/_maintenance/tests -q -p no:cacheprovider
python -B -m pytest scikitplot/cleanprompt/tests -q -p no:cacheprovider
python -B -m pytest docs/source/user_guide/cleanprompt -q -p no:cacheprovider
python -B maintenances/cleanprompt/_maintenance/tools/generate_coverage_docs.py --check
```

The package suite needs the parent `scikitplot` importable. In a source
checkout without a compiled build, run it under a stand-in parent: a directory
holding `scikitplot/__init__.py` (empty) and `scikitplot/cleanprompt` (a copy
or link of the real package), with a `pytest.ini` that sets
`filterwarnings = error`. That is how rounds 23–25 were verified, and
`tests/_isolated.py` measures import isolation the same way.

**Never edit the tree while a run reads it** (`plugins/_shared/skills/scikitplot-engineering-rules/rules/verification.md`, rules 1 and
22): verify against a frozen copy, or wait for the run to finish.

## Current round

Round twenty-eight is **in progress** (2026-11-04): the open ledger, one note
per delivery, PR **907**. Closed so far: the custom-pattern time limit
(`CP-108`, `DESIGN.md` D-28.1) and invisible characters inside names
(`CP-109`, D-28.2), and e-mail addresses in any script (`CP-110`,
D-28.3), and phone groupings and ranges (`CP-111`, D-28.4), and source spellings of labels (`CP-112`, D-28.5; across turns
`CP-113`, D-28.6), look-alike letters (`CP-114`, D-28.7), and per-kind
actions (`CP-115`, D-28.8), the coverage matrix (`CP-116`, D-28.9), the vehicle pack (`CP-117`,
D-28.10), avoided words in surrogate sets (`CP-118`, D-28.11), and the
`de` pack (`CP-119`, D-28.12); see
`HISTORY.md`. Their numbers are under **Last
verified numbers**.

Round twenty-six is **complete** (2026-10-10): *the user decides, and the
floor does not move* — a pattern-risk check for custom packs (warn by
default, tuned per pattern, run, team or machine), custom surrogate name
sets under a fixed safety floor, and the generator's design and growth plan.
Summary in `HISTORY.md`; plan and results in `history/r26_pattern_risk.md`
(archived from `tasks/todo.md`, "cleanprompt round 26"); reasoning in `DESIGN.md` section 25 and
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
| 10 | Round 27 (PR 864 CI): CodeQL `py/redos` on test fixtures, skinny `_cli` test reading cleanprompt, corpus ResourceWarning | done | `test_regex_fixtures.py` gate; `libs._tools verify scikit-plots-skinny` PASS (3.13, 3.8); corpus suite with `-W error::ResourceWarning` (DNS-dependent tests excepted, sandbox only); see `HISTORY.md` round 27 |

## Last verified numbers

Round twenty-eight, after `CP-108` (2026-11-04, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 2954 passed, 102 skipped
- with `regex` 2026.9.29 (CPython 3.13): 2970 passed, 86 skipped
- with `regex` 2020.1.8, the floor (CPython 3.8, stand-in parent): 2836
  passed, 220 skipped
- maintenance plane and user guide: 99 passed; `check_trackers.py`:
  maintenance PASS, runtime PASS, release UNVERIFIED
- mutation check of the new code: 15 of 15 mutants killed

After `CP-109` (same day, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 2991 passed, 102 skipped
- with `regex` (CPython 3.13): 3007 passed, 86 skipped
- CPython 3.8, stand-in parent: 2873 passed, 220 skipped
- maintenance plane and user guide: 99 passed; `check_trackers.py`:
  maintenance PASS, runtime PASS, release UNVERIFIED
- mutation check: 11 of 12 killed, the survivor equivalent (the ASCII fast
  path of `structure_view`)

After `CP-110` (same day, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 3042 passed, 102 skipped
- with `regex` (CPython 3.13): 3058 passed, 86 skipped
- CPython 3.8, stand-in parent: 2924 passed, 220 skipped
- `probe_negative.py`: 0 failures; `probe_fuzz.py`: 4000 documents, 0
  failures; maintenance plane and user guide: 99 passed
- mutation check of the pattern: 4 of 4 killed

After `CP-111` (same day, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 3096 passed, 102 skipped
- with `regex` (CPython 3.13): 3112 passed, 86 skipped
- CPython 3.8, stand-in parent: 2978 passed, 220 skipped
- `probe_scale.py` (2000 phone rows), `probe_negative.py`: 0 failures;
  `probe_fuzz.py`: 4000 documents, 0 failures
- mutation check: 5 of 6 killed; the survivor was a redundant clause, removed

After `CP-112` (same day, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 3113 passed, 102 skipped
- with `regex` (CPython 3.13): 3129 passed, 86 skipped
- CPython 3.8, stand-in parent: 2995 passed, 220 skipped
- `probe_negative.py` (I1 with lenient spellings in its cold fragments): 0
  failures; `probe_fuzz.py`: 4000 documents, 0 failures

After `CP-113` (2026-11-05, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 3121 passed, 102 skipped
- with `regex` (CPython 3.13): 3137 passed, 86 skipped
- CPython 3.8, stand-in parent: 3003 passed, 220 skipped
- `probe_scale.py`: 120000 rows in 9.9 s (10.1 s before); negative and fuzz
  probes: 0 failures

After `CP-114` (2026-11-05, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 3143 passed, 102 skipped
- with `regex` (CPython 3.13): 3159 passed, 86 skipped
- CPython 3.8 (unicodedata 12.1.0), stand-in parent: 3025 passed, 220 skipped
- `probe_negative.py` (with `V28`), `probe_fuzz.py`, `probe_scale.py`: 0
  failures; maintenance plane, lib tools and user guide: 802 passed
- the look-alike table is regenerated with
  `tools/generate_confusables.py PATH/confusables.txt` (`--check` to verify)

After `CP-115` (2026-11-06, frozen copy; `evidence/round28.log`):

- default output byte-identical to the earlier tree (1500 random documents)
- package suite, no optional tier (CPython 3.13): 3184 passed, 102 skipped
- with `regex` (CPython 3.13): 3200 passed, 86 skipped
- CPython 3.8, stand-in parent: 3066 passed, 220 skipped
- probes: 0 failures; maintenance plane, lib tools and user guide: 802 passed

After `CP-116` (2026-11-06, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 3190 passed, 102 skipped
- with `regex` (CPython 3.13): 3206 passed, 86 skipped
- CPython 3.8, stand-in parent: 3072 passed, 220 skipped
- probes: 0 failures; maintenance plane, lib tools and user guide: 804
  passed; the coverage table is current

After `CP-117` (2026-11-06, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 3217 passed, 102 skipped
- with `regex` (CPython 3.13): 3233 passed, 86 skipped
- CPython 3.8, stand-in parent: 3099 passed, 220 skipped
- probes (scale with 2000 VIN rows): 0 failures; `packs --check` and the
  coverage table current; maintenance plane, lib tools and user guide: 804
  passed

After `CP-118` (2026-11-07, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 3236 passed, 102 skipped
- with `regex` (CPython 3.13): 3252 passed, 86 skipped
- CPython 3.8, stand-in parent: 3118 passed, 220 skipped
- probes: 0 failures; default output unchanged (1500 documents); maintenance
  plane, lib tools and user guide: 804 passed

After `CP-119` (2026-11-07, frozen copy; `evidence/round28.log`):

- package suite, no optional tier (CPython 3.13): 3263 passed, 102 skipped
- with `regex` (CPython 3.13): 3279 passed, 86 skipped
- CPython 3.8, stand-in parent: 3145 passed, 220 skipped
- probes (scale with 2000 Steuer-ID rows): 0 failures; `packs --check` and
  the coverage table current; maintenance plane, lib tools and user guide:
  804 passed

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

Round twenty-eight, continuing. Suggested order (each has a note with a full design):

1. Capability notes: further locale identifier packs (one jurisdiction per
   pack, each from a published specification; `de` done); `surrogate-provider-protocol.md` only on a
   concrete request.

Reset the step log above for round 27 before starting.

## Open ledger

Every note under `upcoming_changes/scikitplot/cleanprompt/`, with its status.
`tests/test_resume.py` compares this list with the directory. The four
promoted PR 864 notes were removed on 2026-11-04 after the merge; their seven
release-note fragments (`864.*.rst` under `security`, `scikitplot.cleanprompt`
and `scikitplot._cli`), missing from the tree, were restored from the text the
notes recorded.

- `upcoming_changes/scikitplot/cleanprompt/locale-identifier-packs.md` — open
- `upcoming_changes/scikitplot/cleanprompt/surrogate-provider-protocol.md` — open

## Decisions waiting for the maintainer

1. A contributor GitHub handle for the `By :user:` lines of the PR-864
   fragments (not guessed; none added).
2. (Taken 2026-11-07: ordinary-word stand-ins are handled by a per-set
   `avoid` list, `CP-118`.)
